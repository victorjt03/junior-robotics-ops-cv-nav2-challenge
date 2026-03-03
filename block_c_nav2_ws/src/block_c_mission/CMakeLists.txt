cmake_minimum_required(VERSION 3.16)
project(block_c_mission)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(vision_msgs REQUIRED)

add_executable(event_monitor_node src/event_monitor_node.cpp)
ament_target_dependencies(event_monitor_node rclcpp vision_msgs)

install(TARGETS event_monitor_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install runtime assets used by launch files.
# IMPORTANT: include params/ so the bridge yaml is available after colcon install.
install(DIRECTORY
  launch
  worlds
  models
  params
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()