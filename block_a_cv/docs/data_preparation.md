# Data preparation (Roboflow -> YOLO)

## Source
- Dataset: `cone_wdg6r` (Roboflow Universe export)
- Export format: YOLOv7 (compatible with YOLOv9 training scripts)

## Initial issues found
- The export contained only a `train/` split (no `val/` or `test/`).
- Labels contained multiple class ids (0..4), while the challenge requires a single class of interest.

## Actions
1. **Reproducible split**
   - We generate `train/val/test = 0.8/0.1/0.1` using a fixed `seed=42`.
2. **Single-class remap**
   - All class ids in YOLO labels are mapped to `0` to create a mono-class detector (`cone`).
3. **Leakage checks**
   - We compute SHA1 hashes for all images to detect **exact duplicates**.
   - Result: `1` duplicate pair detected (see `prepare_report.json`).

## Outputs
- Processed dataset (not committed): `block_a_cv/data/processed/cone_wdg6r_yolo/`
- Reproducibility artifacts:
  - `block_a_cv/scripts/prepare_dataset.py`
  - `prepare_report.json` (generated locally)
- Visual sanity check:
  - `block_a_cv/docs/samples_overlay_train.jpg`
  - `block_a_cv/docs/samples_overlay_val.jpg`