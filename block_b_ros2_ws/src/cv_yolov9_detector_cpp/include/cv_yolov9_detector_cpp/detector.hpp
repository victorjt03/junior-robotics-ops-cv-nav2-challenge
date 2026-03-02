#pragma once
#include <vector>
#include <opencv2/core.hpp>

namespace cv_yolov9_detector_cpp {

struct Detection {
  float x1, y1, x2, y2;
  float conf;
  int class_id;
};

class IDetector {
public:
  virtual ~IDetector() = default;
  virtual std::vector<Detection> infer(const cv::Mat& bgr) = 0;
};

}  // namespace cv_yolov9_detector_cpp
