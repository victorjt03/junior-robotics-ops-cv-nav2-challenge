# Metrics (object detection)

This project uses standard detection metrics computed on the validation split.

## Precision
**Precision = TP / (TP + FP)**  
Of all predicted cones, how many were correct.  
Operationally: prioritize precision when false positives are costly (e.g., unnecessary emergency stops).

## Recall
**Recall = TP / (TP + FN)**  
Of all ground-truth cones, how many were detected.  
Operationally: prioritize recall when missing an obstacle is costly.

## AP and mAP
AP (Average Precision) is the area under the Precision–Recall curve for one class.

- **mAP@0.5**: AP at IoU threshold 0.5 (more permissive).
- **mAP@0.5:0.95**: mean AP over IoU thresholds 0.50..0.95 (stricter localization quality).

## Baseline numbers (seed=42)
From `block_a_cv/runs/baseline_yolov9s_seed42_img512_b4_e60/results.csv` (epoch ~9):
- Precision ≈ 0.93
- Recall ≈ 0.88
- mAP@0.5 ≈ 0.95
- mAP@0.5:0.95 ≈ 0.65