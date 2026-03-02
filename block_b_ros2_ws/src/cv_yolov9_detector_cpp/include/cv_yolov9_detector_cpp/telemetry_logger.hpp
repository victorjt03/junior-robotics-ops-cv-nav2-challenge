#pragma once
#include <fstream>
#include <mutex>
#include <string>

namespace cv_yolov9_detector_cpp {

class TelemetryLogger {
public:
  explicit TelemetryLogger(const std::string& jsonl_path);
  void append_json_line(const std::string& json_line);

private:
  std::mutex mtx_;
  std::ofstream out_;
};

}  // namespace cv_yolov9_detector_cpp