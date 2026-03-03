from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    # Args
    use_sim_time = LaunchConfiguration("use_sim_time")
    detections_topic = LaunchConfiguration("detections_topic")
    conf_thres = LaunchConfiguration("conf_thres")
    pause_seconds = LaunchConfiguration("pause_seconds")
    waypoints_yaml = LaunchConfiguration("waypoints_yaml")

    # Default config file inside the package
    default_waypoints = PathJoinSubstitution([
        FindPackageShare("block_c_mission"),
        "config",
        "waypoints.yaml",
    ])

    # Event monitor node (listens to /detections and triggers event)
    event_monitor = Node(
        package="block_c_mission",
        executable="event_monitor_node",
        name="event_monitor",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "detections_topic": detections_topic,
            "conf_thres": conf_thres,
            "pause_seconds": pause_seconds,
            "waypoints_yaml": waypoints_yaml,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation clock",
        ),
        DeclareLaunchArgument(
            "detections_topic",
            default_value="/detections",
            description="vision_msgs/Detection2DArray topic",
        ),
        DeclareLaunchArgument(
            "conf_thres",
            default_value="0.5",
            description="Confidence threshold to trigger event",
        ),
        DeclareLaunchArgument(
            "pause_seconds",
            default_value="3.0",
            description="Optional pause duration when event triggers",
        ),
        DeclareLaunchArgument(
            "waypoints_yaml",
            default_value=default_waypoints,
            description="Waypoints YAML path",
        ),
        event_monitor,
    ])