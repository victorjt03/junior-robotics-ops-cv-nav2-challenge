// File: block_b_ros2_ws/src/cv_yolov9_detector_cpp/src/yolov9_detector_node.cpp

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <nlohmann/json.hpp>

#include "cv_yolov9_detector_cpp/onnx_runtime_detector.hpp"
#include "cv_yolov9_detector_cpp/telemetry_logger.hpp"
#include "cv_yolov9_detector_cpp/vision_msgs_converter.hpp"

namespace cv_yolov9_detector_cpp {

static uint64_t now_ns() {
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()
    ).count()
  );
}

class YoloV9DetectorNode : public rclcpp::Node {
public:
  YoloV9DetectorNode()
  : Node("yolov9_detector_cpp")
  {
    // -----------------------
    // Parameters
    // -----------------------
    image_topic_ = declare_parameter<std::string>("image_topic", "/camera/image_raw");
    detections_topic_ = declare_parameter<std::string>("detections_topic", "/detections");
    telemetry_topic_ = declare_parameter<std::string>("telemetry_topic", "/telemetry/cv");

    // NOTE: do NOT default to a relative path that depends on CWD
    // User should pass an absolute path, or leave empty (telemetry-only mode).
    model_path_ = declare_parameter<std::string>("model_path", "");

    device_ = declare_parameter<std::string>("device", "cpu");
    imgsz_ = declare_parameter<int>("imgsz", 512);
    conf_thres_ = declare_parameter<double>("conf_thres", 0.25);
    iou_thres_ = declare_parameter<double>("iou_thres", 0.7);
    max_det_ = declare_parameter<int>("max_det", 300);

    class_id_ = declare_parameter<int>("class_id", 0);
    class_name_ = declare_parameter<std::string>("class_name", "cone");

    // Telemetry path: default to ~/.ros/cv_telemetry.jsonl
    const char* home = std::getenv("HOME");
    const std::filesystem::path default_tp =
      (home ? std::filesystem::path(home) : std::filesystem::path(".")) / ".ros" / "cv_telemetry.jsonl";

    telemetry_jsonl_path_ = declare_parameter<std::string>("telemetry_path", default_tp.string());

    // Create parent directory if needed
    {
      const std::filesystem::path tp(telemetry_jsonl_path_);
      if (tp.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(tp.parent_path(), ec);
        if (ec) {
          RCLCPP_WARN(
            get_logger(),
            "Failed to create telemetry directory '%s': %s",
            tp.parent_path().string().c_str(),
            ec.message().c_str()
          );
        }
      }
    }

    telemetry_logger_ = std::make_unique<TelemetryLogger>(telemetry_jsonl_path_);

    // Optional detector init (lets you run without model)
    if (!model_path_.empty()) {
      detector_ = std::make_unique<OnnxRuntimeDetector>(
        model_path_, imgsz_,
        static_cast<float>(conf_thres_),
        static_cast<float>(iou_thres_),
        max_det_
      );
      RCLCPP_INFO(get_logger(), "Detector enabled. model_path='%s'", model_path_.c_str());
    } else {
      RCLCPP_WARN(
        get_logger(),
        "Parameter 'model_path' is empty. Node will run in telemetry-only mode (no inference). "
        "Provide: --ros-args -p model_path:=/absolute/path/to/model.onnx"
      );
    }

    // -----------------------
    // Publishers/Subscribers
    // -----------------------
    pub_det_ = create_publisher<vision_msgs::msg::Detection2DArray>(detections_topic_, 10);
    pub_tel_ = create_publisher<std_msgs::msg::String>(telemetry_topic_, 10);

    sub_img_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&YoloV9DetectorNode::on_image, this, std::placeholders::_1)
    );

    last_t_ns_ = 0;

    RCLCPP_INFO(
      get_logger(),
      "Started. image_topic='%s' detections_topic='%s' telemetry_topic='%s' telemetry_path='%s'",
      image_topic_.c_str(),
      detections_topic_.c_str(),
      telemetry_topic_.c_str(),
      telemetry_jsonl_path_.c_str()
    );
  }

private:
  void on_image(const sensor_msgs::msg::Image::SharedPtr msg) {
    const uint64_t t0 = now_ns();

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
      return;
    }

    const cv::Mat& bgr = cv_ptr->image;

    std::vector<Detection> dets;
    if (detector_) {
      try {
        dets = detector_->infer(bgr);
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "infer: %s", e.what());
        return;
      }
      pub_det_->publish(to_detection2d_array(msg->header, dets, class_id_, class_name_));
    } else {
      // Publish empty detections array to keep downstream consistent
      vision_msgs::msg::Detection2DArray empty;
      empty.header = msg->header;
      pub_det_->publish(empty);
    }

    const uint64_t t1 = now_ns();
    const float latency_ms = static_cast<float>((t1 - t0) / 1e6);

    float fps = 0.0f;
    if (last_t_ns_ != 0) {
      const double dt_s = static_cast<double>(t1 - last_t_ns_) / 1e9;
      fps = (dt_s > 0.0) ? static_cast<float>(1.0 / dt_s) : 0.0f;
    }
    last_t_ns_ = t1;

    float mean_conf = 0.0f;
    for (const auto& d : dets) mean_conf += d.conf;
    if (!dets.empty()) mean_conf /= static_cast<float>(dets.size());

    nlohmann::json j{
      {"stamp_ns", t1},
      {"frame_id", msg->header.frame_id},
      {"img_w", bgr.cols},
      {"img_h", bgr.rows},
      {"image_topic", image_topic_},
      {"model_path", model_path_},
      {"device", device_},
      {"imgsz", imgsz_},
      {"conf_thres", static_cast<float>(conf_thres_)},
      {"iou_thres", static_cast<float>(iou_thres_)},
      {"max_det", max_det_},
      {"latency_ms", latency_ms},
      {"fps", fps},
      {"num_detections", static_cast<int>(dets.size())},
      {"mean_conf", mean_conf}
    };

    std_msgs::msg::String tel_msg;
    tel_msg.data = j.dump();
    pub_tel_->publish(tel_msg);

    // TelemetryLogger writes JSONL
    try {
      telemetry_logger_->append_json_line(tel_msg.data);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "telemetry_logger: %s", e.what());
    }
  }

  std::string image_topic_;
  std::string detections_topic_;
  std::string telemetry_topic_;

  std::string model_path_;
  std::string device_;
  int imgsz_{512};
  double conf_thres_{0.25};
  double iou_thres_{0.7};
  int max_det_{300};

  int class_id_{0};
  std::string class_name_{"cone"};

  std::string telemetry_jsonl_path_;
  uint64_t last_t_ns_{0};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_det_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_tel_;

  std::unique_ptr<IDetector> detector_;
  std::unique_ptr<TelemetryLogger> telemetry_logger_;
};

}  // namespace cv_yolov9_detector_cpp

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cv_yolov9_detector_cpp::YoloV9DetectorNode>());
  rclcpp::shutdown();
  return 0;
}