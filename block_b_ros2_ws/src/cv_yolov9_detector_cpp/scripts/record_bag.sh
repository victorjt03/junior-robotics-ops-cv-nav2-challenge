#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-block_b_ros2_ws/bags/cv_run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"
ros2 bag record -o "$OUT_DIR" /camera/image_raw /detections /telemetry/cv