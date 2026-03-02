#pragma once
#include "cv_yolov9_detector_cpp/detector.hpp"
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <std_msgs/msg/header.hpp>
#include <string>
#include <vector>

namespace cv_yolov9_detector_cpp {

vision_msgs::msg::Detection2DArray to_detection2d_array(
  const std_msgs::msg::Header& header,
  const std::vector<Detection>& dets,
  int class_id,
  const std::string& class_name);

}  // namespace cv_yolov9_detector_cpp