#include "cv_yolov9_detector_cpp/telemetry_logger.hpp"
#include <stdexcept>

namespace cv_yolov9_detector_cpp {

TelemetryLogger::TelemetryLogger(const std::string& jsonl_path) {
  out_.open(jsonl_path, std::ios::app);
  if (!out_.is_open()) {
    throw std::runtime_error("Failed to open telemetry jsonl file: " + jsonl_path);
  }
}

void TelemetryLogger::append_json_line(const std::string& json_line) {
  std::lock_guard<std::mutex> lock(mtx_);
  out_ << json_line << "\n";
  out_.flush();
}

}  // namespace cv_yolov9_detector_cpp