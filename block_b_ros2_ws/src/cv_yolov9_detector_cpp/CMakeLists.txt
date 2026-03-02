cmake_minimum_required(VERSION 3.16)
project(cv_yolov9_detector_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

# ---- nlohmann json (header-only) ----
# Ubuntu package: nlohmann-json3-dev (headers in /usr/include/nlohmann)
# We do not require a CMake config package to exist.
find_package(nlohmann_json CONFIG QUIET)
if(nlohmann_json_FOUND)
  message(STATUS "Found nlohmann_json via CMake config")
  set(HAVE_NLOHMANN_JSON 1)
else()
  message(STATUS "nlohmann_json CMake config not found; using header-only include (from nlohmann-json3-dev)")
  set(HAVE_NLOHMANN_JSON 1)
endif()
add_compile_definitions(CV_YOLOV9_HAS_NLOHMANN_JSON=${HAVE_NLOHMANN_JSON})

# ---- Optional: ONNX Runtime via pkg-config ----
find_package(PkgConfig REQUIRED)
pkg_check_modules(ONNXRUNTIME QUIET IMPORTED_TARGET onnxruntime)

if(ONNXRUNTIME_FOUND)
  message(STATUS "Found ONNX Runtime via pkg-config: ${ONNXRUNTIME_VERSION}")
  add_compile_definitions(CV_YOLOV9_HAS_ONNXRUNTIME=1)
else()
  message(WARNING "ONNX Runtime not found (pkg-config: onnxruntime). Build will continue without it.")
  add_compile_definitions(CV_YOLOV9_HAS_ONNXRUNTIME=0)
endif()

# ---- Core library ----
add_library(${PROJECT_NAME}_core
  src/onnx_runtime_detector.cpp
  src/vision_msgs_converter.cpp
  src/telemetry_logger.cpp
)

target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(${PROJECT_NAME}_core
  rclcpp
  sensor_msgs
  std_msgs
  vision_msgs
  cv_bridge
  image_transport
)

target_link_libraries(${PROJECT_NAME}_core
  ${OpenCV_LIBRARIES}
)

# nlohmann_json: link only if config package exists; otherwise header-only include is enough
if(nlohmann_json_FOUND)
  target_link_libraries(${PROJECT_NAME}_core nlohmann_json::nlohmann_json)
endif()

# ONNX Runtime: use imported target from pkg-config (adds -I and -L automatically)
if(ONNXRUNTIME_FOUND)
  target_link_libraries(${PROJECT_NAME}_core PkgConfig::ONNXRUNTIME)
endif()

# ---- Node executable ----
add_executable(yolov9_detector_node
  src/yolov9_detector_node.cpp
)

target_link_libraries(yolov9_detector_node
  ${PROJECT_NAME}_core
)

ament_target_dependencies(yolov9_detector_node
  rclcpp
  sensor_msgs
  std_msgs
  vision_msgs
  cv_bridge
  image_transport
)

# ---- Install ----
install(TARGETS
  ${PROJECT_NAME}_core
  yolov9_detector_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
  install(PROGRAMS
    scripts/record_bag.sh
    scripts/play_bag.sh
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

ament_package()