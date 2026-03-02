#include "cv_yolov9_detector_cpp/vision_msgs_converter.hpp"
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <vision_msgs/msg/bounding_box2_d.hpp>

namespace cv_yolov9_detector_cpp {

vision_msgs::msg::Detection2DArray to_detection2d_array(
  const std_msgs::msg::Header& header,
  const std::vector<Detection>& dets,
  int class_id,
  const std::string& class_name) {

  vision_msgs::msg::Detection2DArray out;
  out.header = header;

  out.detections.reserve(dets.size());
  for (const auto& d : dets) {
    vision_msgs::msg::Detection2D det;

    vision_msgs::msg::BoundingBox2D bbox;
    const float cx = 0.5f * (d.x1 + d.x2);
    const float cy = 0.5f * (d.y1 + d.y2);
    const float w  = (d.x2 - d.x1);
    const float h  = (d.y2 - d.y1);

    bbox.center.position.x = cx;
    bbox.center.position.y = cy;
    bbox.size_x = w;
    bbox.size_y = h;
    det.bbox = bbox;

    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = class_name;      // string id
    hyp.hypothesis.score = d.conf;
    det.results.push_back(hyp);

    out.detections.push_back(det);
  }

  (void)class_id;  // reservado por si quieres mapear numeric id → string
  return out;
}

}  // namespace cv_yolov9_detector_cpp