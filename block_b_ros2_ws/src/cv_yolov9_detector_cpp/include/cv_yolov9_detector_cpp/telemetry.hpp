#pragma once
#include <cstdint>
#include <string>

namespace cv_yolov9_detector_cpp {

struct Telemetry {
  uint64_t stamp_ns{};
  std::string frame_id{};
  int img_w{};
  int img_h{};
  std::string model_path{};
  std::string device{};
  float conf_thres{};
  float iou_thres{};
  float latency_ms{};
  float fps{};
  int num_detections{};
  float mean_conf{};
};

}  // namespace cv_yolov9_detector_cpp