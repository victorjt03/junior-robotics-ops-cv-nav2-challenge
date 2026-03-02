#include "cv_yolov9_detector_cpp/onnx_runtime_detector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cv_yolov9_detector_cpp {

static inline float clampf(float v, float lo, float hi) {
  return std::max(lo, std::min(hi, v));
}

OnnxRuntimeDetector::OnnxRuntimeDetector(const std::string& model_path,
                                         int imgsz,
                                         float conf_thres,
                                         float iou_thres,
                                         int max_det)
: model_path_(model_path),
  imgsz_(imgsz),
  conf_thres_(conf_thres),
  iou_thres_(iou_thres),
  max_det_(max_det),
  env_(ORT_LOGGING_LEVEL_WARNING, "cv_yolov9_detector_cpp")
{
  if (model_path_.empty()) {
    throw std::runtime_error("model_path is empty");
  }
  if (imgsz_ <= 0) {
    throw std::runtime_error("imgsz must be > 0");
  }

  session_opts_.SetIntraOpNumThreads(1);
  session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  session_ = Ort::Session(env_, model_path_.c_str(), session_opts_);

  // ---- Cache I/O names (ORT 1.17+ API) ----
  Ort::AllocatorWithDefaultOptions allocator;

  const size_t num_inputs = session_.GetInputCount();
  input_names_str_.reserve(num_inputs);
  input_names_.reserve(num_inputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    Ort::AllocatedStringPtr name_ptr = session_.GetInputNameAllocated(i, allocator);
    input_names_str_.emplace_back(name_ptr.get());
  }
  for (auto& s : input_names_str_) input_names_.push_back(s.c_str());

  const size_t num_outputs = session_.GetOutputCount();
  output_names_str_.reserve(num_outputs);
  output_names_.reserve(num_outputs);

  for (size_t i = 0; i < num_outputs; ++i) {
    Ort::AllocatedStringPtr name_ptr = session_.GetOutputNameAllocated(i, allocator);
    output_names_str_.emplace_back(name_ptr.get());
  }
  for (auto& s : output_names_str_) output_names_.push_back(s.c_str());
}

cv::Mat OnnxRuntimeDetector::letterbox(const cv::Mat& src_bgr, int new_size, LetterboxInfo* info) {
  const int w = src_bgr.cols;
  const int h = src_bgr.rows;

  const float r = std::min(static_cast<float>(new_size) / static_cast<float>(w),
                           static_cast<float>(new_size) / static_cast<float>(h));
  const int new_w = static_cast<int>(std::round(w * r));
  const int new_h = static_cast<int>(std::round(h * r));

  cv::Mat resized;
  cv::resize(src_bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  const int pad_x = (new_size - new_w) / 2;
  const int pad_y = (new_size - new_h) / 2;

  cv::Mat out(new_size, new_size, src_bgr.type(), cv::Scalar(114, 114, 114));
  resized.copyTo(out(cv::Rect(pad_x, pad_y, new_w, new_h)));

  if (info) {
    info->r = r;
    info->pad_x = pad_x;
    info->pad_y = pad_y;
    info->new_w = new_w;
    info->new_h = new_h;
  }
  return out;
}

float OnnxRuntimeDetector::iou_xyxy(const Detection& a, const Detection& b) {
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);

  const float iw = std::max(0.0f, x2 - x1);
  const float ih = std::max(0.0f, y2 - y1);
  const float inter = iw * ih;

  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float uni = area_a + area_b - inter;

  return (uni > 0.0f) ? (inter / uni) : 0.0f;
}

std::vector<Detection> OnnxRuntimeDetector::nms(std::vector<Detection> dets, float iou_thres, int max_det) {
  std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) { return a.conf > b.conf; });

  std::vector<Detection> keep;
  keep.reserve(std::min<int>(static_cast<int>(dets.size()), max_det));

  for (const auto& d : dets) {
    bool ok = true;
    for (const auto& k : keep) {
      if (iou_xyxy(d, k) > iou_thres) { ok = false; break; }
    }
    if (ok) {
      keep.push_back(d);
      if (static_cast<int>(keep.size()) >= max_det) break;
    }
  }
  return keep;
}

std::vector<Detection> OnnxRuntimeDetector::infer(const cv::Mat& bgr) {
  if (bgr.empty()) return {};

  // 1) Letterbox
  LetterboxInfo lb;
  cv::Mat boxed = letterbox(bgr, imgsz_, &lb);

  // 2) BGR -> RGB
  cv::Mat rgb;
  cv::cvtColor(boxed, rgb, cv::COLOR_BGR2RGB);

  // 3) to float32 [0,1]
  cv::Mat f32;
  rgb.convertTo(f32, CV_32F, 1.0 / 255.0);

  // 4) HWC -> CHW
  std::vector<float> input_tensor(1 * 3 * imgsz_ * imgsz_);
  const int H = imgsz_;
  const int W = imgsz_;
  for (int y = 0; y < H; ++y) {
    const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);
    for (int x = 0; x < W; ++x) {
      const cv::Vec3f p = row[x];     // RGB
      input_tensor[0 * H * W + y * W + x] = p[0];  // R
      input_tensor[1 * H * W + y * W + x] = p[1];  // G
      input_tensor[2 * H * W + y * W + x] = p[2];  // B
    }
  }

  // 5) Run
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<int64_t, 4> in_shape{1, 3, imgsz_, imgsz_};

  Ort::Value in = Ort::Value::CreateTensor<float>(
    mem_info, input_tensor.data(), input_tensor.size(), in_shape.data(), in_shape.size());

  auto outs = session_.Run(
    Ort::RunOptions{nullptr},
    input_names_.data(),
    &in,
    1,
    output_names_.data(),
    output_names_.size());

  // Prefer output0 if exists
  int out_idx = 0;
  for (size_t i = 0; i < output_names_str_.size(); ++i) {
    if (output_names_str_[i] == "output0") { out_idx = static_cast<int>(i); break; }
  }

  Ort::Value& out = outs.at(out_idx);
  float* out_data = out.GetTensorMutableData<float>();

  // Expect [1,5,5376] channel-major
  constexpr int C = 5;
  const int N = 5376;

  std::vector<Detection> dets;
  dets.reserve(64);

  for (int i = 0; i < N; ++i) {
    const float cx = out_data[0 * N + i];
    const float cy = out_data[1 * N + i];
    const float w  = out_data[2 * N + i];
    const float h  = out_data[3 * N + i];
    const float conf = out_data[4 * N + i];

    if (conf < conf_thres_) continue;

    Detection d{};
    d.conf = conf;

    float x1 = cx - w * 0.5f;
    float y1 = cy - h * 0.5f;
    float x2 = cx + w * 0.5f;
    float y2 = cy + h * 0.5f;

    // Undo letterbox
    x1 = (x1 - static_cast<float>(lb.pad_x)) / lb.r;
    y1 = (y1 - static_cast<float>(lb.pad_y)) / lb.r;
    x2 = (x2 - static_cast<float>(lb.pad_x)) / lb.r;
    y2 = (y2 - static_cast<float>(lb.pad_y)) / lb.r;

    x1 = clampf(x1, 0.0f, static_cast<float>(bgr.cols - 1));
    y1 = clampf(y1, 0.0f, static_cast<float>(bgr.rows - 1));
    x2 = clampf(x2, 0.0f, static_cast<float>(bgr.cols - 1));
    y2 = clampf(y2, 0.0f, static_cast<float>(bgr.rows - 1));

    if (x2 <= x1 || y2 <= y1) continue;

    d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
    dets.push_back(d);
  }

  dets = nms(std::move(dets), iou_thres_, max_det_);
  return dets;
}

}  // namespace cv_yolov9_detector_cpp