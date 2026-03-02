#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=42
DATASET="${DATASET:-block_a_cv/data/processed/cone_wdg6r_yolo/data.yaml}"
IMG=512
BATCH=4
EPOCHS=60
WORKERS=2

YOLO_DIR="block_a_cv/third_party/yolov9"
RUNS_DIR="block_a_cv/runs"
WEIGHTS="block_a_cv/weights/yolov9-s.pt"
CFG="block_a_cv/third_party/yolov9/models/detect/yolov9-s.yaml"
HYP="block_a_cv/third_party/yolov9/data/hyps/hyp.scratch-high.yaml"

python "${YOLO_DIR}/train_dual.py" \
  --data "${DATASET}" \
  --cfg "${CFG}" \
  --hyp "${HYP}" \
  --img "${IMG}" \
  --batch "${BATCH}" \
  --epochs "${EPOCHS}" \
  --weights "${WEIGHTS}" \
  --workers "${WORKERS}" \
  --project "${RUNS_DIR}" \
  --name "baseline_yolov9s_seed${SEED}_img${IMG}_b${BATCH}_e${EPOCHS}" \
  --seed "${SEED}"