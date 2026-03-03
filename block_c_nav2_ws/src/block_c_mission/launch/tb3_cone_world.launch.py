#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    # Robot spawn pose
    x_pose = LaunchConfiguration("x_pose")
    y_pose = LaunchConfiguration("y_pose")
    z_pose = LaunchConfiguration("z_pose")
    yaw = LaunchConfiguration("yaw")

    pkg_share = FindPackageShare("block_c_mission")
    tb3_gz_share = FindPackageShare("turtlebot3_gazebo")

    # --- Resource paths (TB3 models + tu cone_poster) ---
    tb3_models_dir = PathJoinSubstitution([tb3_gz_share, "models"])
    my_models_dir = PathJoinSubstitution([pkg_share, "models"])

    existing_gz_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    existing_ign_path = os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "")

    set_gz_resource = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[
            tb3_models_dir, TextSubstitution(text=":"),
            my_models_dir, TextSubstitution(text=":"),
            TextSubstitution(text=existing_gz_path),
        ],
    )
    set_ign_resource = SetEnvironmentVariable(
        name="IGN_GAZEBO_RESOURCE_PATH",
        value=[
            tb3_models_dir, TextSubstitution(text=":"),
            my_models_dir, TextSubstitution(text=":"),
            TextSubstitution(text=existing_ign_path),
        ],
    )

    # --- TB3 model (usa waffle_pi porque tiene cámara en model.sdf) ---
    tb3_model = os.environ.get("TURTLEBOT3_MODEL", "waffle_pi").strip()
    tb3_sdf = PathJoinSubstitution([tb3_gz_share, "models", f"turtlebot3_{tb3_model}", "model.sdf"])

    # --- Bridge YAML (tu archivo con camera/image_raw) ---
    bridge_yaml = PathJoinSubstitution([pkg_share, "params", "tb3_bridge_with_camera.yaml"])

    # 1) Gazebo estable (empty_world) -> aquí sí funciona el stack de sensores/render
    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([tb3_gz_share, "launch", "empty_world.launch.py"])
        ),
        launch_arguments={
            "gz_version": "8",
            "use_sim_time": use_sim_time,
        }.items(),
    )

    # 2) Spawn TB3 desde SDF (IMPORTANTE: esto mete sensores/cámara)
    spawn_tb3 = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name", "turtlebot3",
            "-file", tb3_sdf,
            "-x", x_pose,
            "-y", y_pose,
            "-z", z_pose,
            "-Y", yaw,
        ],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # 3) Bridge (clock/odom/tf/scan/camera_info/camera_image)
    parameter_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        parameters=[{"config_file": bridge_yaml, "use_sim_time": use_sim_time}],
    )

    # 4) Spawn poster (tu modelo) con delay
    poster_sdf = PathJoinSubstitution([pkg_share, "models", "cone_poster", "model.sdf"])
    spawn_poster = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name", "cone_poster",
            "-file", poster_sdf,
            "-x", "2.0",
            "-y", "0.0",
            "-z", "0.6",
            "-Y", "1.57079632679",
        ],
        parameters=[{"use_sim_time": use_sim_time}],
    )
    spawn_poster_delayed = TimerAction(period=3.0, actions=[spawn_poster])

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("x_pose", default_value="-1.5"),
        DeclareLaunchArgument("y_pose", default_value="0.0"),
        DeclareLaunchArgument("z_pose", default_value="0.01"),
        DeclareLaunchArgument("yaw", default_value="0.0"),

        set_gz_resource,
        set_ign_resource,

        gz_launch,
        # OJO: bridge DESPUÉS de arrancar gz, pero da igual; lo importante es que exista el robot SDF con cámara
        spawn_tb3,
        parameter_bridge,
        spawn_poster_delayed,
    ])