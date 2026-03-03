#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <algorithm>
#include <chrono>
#include <string>

class EventMonitorNode : public rclcpp::Node {
public:
  EventMonitorNode() : Node("cv_event_monitor") {
    detections_topic_ = declare_parameter<std::string>("detections_topic", "/detections");
    score_thres_ = declare_parameter<double>("score_thres", 0.60);
    pause_seconds_ = declare_parameter<double>("pause_seconds", 3.0);

    sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
      detections_topic_, 10,
      std::bind(&EventMonitorNode::onDetections, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Listening '%s' score_thres=%.2f pause=%.1fs",
                detections_topic_.c_str(), score_thres_, pause_seconds_);
  }

private:
  void onDetections(const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
    double best = 0.0;
    for (const auto& det : msg->detections) {
      for (const auto& res : det.results) {
        best = std::max(best, static_cast<double>(res.hypothesis.score));
      }
    }

    if (best >= score_thres_) {
      RCLCPP_WARN(get_logger(), "CONE EVENT: detected score=%.3f (threshold=%.2f)",
                  best, score_thres_);

      const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(pause_seconds_));
      rclcpp::sleep_for(ns);
    }
  }

  std::string detections_topic_;
  double score_thres_;
  double pause_seconds_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr sub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EventMonitorNode>());
  rclcpp::shutdown();
  return 0;
}