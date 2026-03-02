#!/usr/bin/env bash
set -euo pipefail
BAG_DIR="${1:?Usage: play_bag.sh <bag_dir>}"
ros2 bag play "$BAG_DIR"