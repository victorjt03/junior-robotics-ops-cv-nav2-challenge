#pragma once

#include "cv_yolov9_detector_cpp/detector.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace cv_yolov9_detector_cpp {

class OnnxRuntimeDetector : public IDetector {
public:
  OnnxRuntimeDetector(const std::string& model_path,
                      int imgsz,
                      float conf_thres,
                      float iou_thres,
                      int max_det);

  std::vector<Detection> infer(const cv::Mat& bgr) override;

private:
  struct LetterboxInfo {
    float r{1.0f};
    int pad_x{0};
    int pad_y{0};
    int new_w{0};
    int new_h{0};
  };

  static cv::Mat letterbox(const cv::Mat& src_bgr, int new_size, LetterboxInfo* info);
  static float iou_xyxy(const Detection& a, const Detection& b);
  static std::vector<Detection> nms(std::vector<Detection> dets, float iou_thres, int max_det);

  std::string model_path_;
  int imgsz_;
  float conf_thres_;
  float iou_thres_;
  int max_det_;

  Ort::Env env_;
  Ort::SessionOptions session_opts_;
  Ort::Session session_{nullptr};

  std::vector<std::string> input_names_str_;
  std::vector<std::string> output_names_str_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
};

}  // namespace cv_yolov9_detector_cpp