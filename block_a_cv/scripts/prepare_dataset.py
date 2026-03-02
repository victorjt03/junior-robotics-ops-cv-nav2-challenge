#!/usr/bin/env python3
# block_a_cv/scripts/prepare_dataset.py
#
# Prepara un dataset YOLO a partir de un export de Roboflow (YOLOv7 format).
# - Soporta: train/valid/test/images|labels (o solo train/)
# - Puede respetar splits existentes o forzar un resplit reproducible
# - Sanity checks de labels
# - Detección de duplicados exactos (sha1)
# - Modo mono-clase: remapea cualquier class_id a 0 (clase "cone")

import argparse
import json
import random
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha1_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_yolo_label(label_path: Path):
    """YOLO label format: class cx cy w h (normalized)."""
    if not label_path.exists():
        return []
    lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
    out = []
    for l in lines:
        parts = l.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        out.append((cls, cx, cy, w, h))
    return out


def rewrite_label_to_single_class(src: Path, dst: Path):
    """Reescribe un label YOLO forzando class_id=0 para todas las anotaciones."""
    if not src.exists():
        dst.write_text("")
        return

    out_lines = []
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        # Force class 0, keep bbox
        out_lines.append("0 " + " ".join(parts[1:5]))

    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))


def sanity_check(img_path: Path, label_path: Path, num_classes: int, remap_to_zero: bool):
    img = cv2.imread(str(img_path))
    if img is None:
        return False, f"Cannot read image: {img_path}"

    anns = parse_yolo_label(label_path)
    for (cls, cx, cy, w, h) in anns:
        # class id checks
        if not remap_to_zero:
            if cls < 0 or cls >= num_classes:
                return False, f"Class out of range in {label_path}: {cls}"
        else:
            if cls < 0:
                return False, f"Negative class id in {label_path}: {cls}"

        # normalized bbox checks
        for v in (cx, cy, w, h):
            if not (0.0 <= v <= 1.0):
                return False, f"Non-normalized value in {label_path}: {v}"
        if w <= 0 or h <= 0:
            return False, f"Non-positive box in {label_path}"

        # box inside [0,1] with epsilon
        if (cx - w / 2) < -1e-3 or (cy - h / 2) < -1e-3 or (cx + w / 2) > 1 + 1e-3 or (cy + h / 2) > 1 + 1e-3:
            return False, f"Box outside [0,1] bounds in {label_path}"

    return True, ""


def list_roboflow_items(raw_root: Path):
    """
    Soporta export Roboflow típico:
      raw_root/{train,valid,test}/{images,labels}
    También soporta que SOLO exista train/.
    """
    items = []
    for split in ("train", "valid", "val", "test"):
        img_dir = raw_root / split / "images"
        lbl_dir = raw_root / split / "labels"
        if not img_dir.exists():
            continue
        for imgp in img_dir.iterdir():
            if imgp.suffix.lower() not in IMG_EXTS:
                continue
            lp = lbl_dir / (imgp.stem + ".txt")
            items.append((split, imgp, lp))
    return items


def copy_split(items, out_root: Path, split_name: str, num_classes: int, remap_to_zero: bool):
    (out_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    bad = []
    for _, imgp, lp in items:
        ok, msg = sanity_check(imgp, lp, num_classes, remap_to_zero)
        if not ok:
            bad.append({"img": str(imgp), "label": str(lp), "error": msg})
            continue

        dst_img = out_root / "images" / split_name / imgp.name
        dst_lbl = out_root / "labels" / split_name / (imgp.stem + ".txt")

        shutil.copy2(imgp, dst_img)
        if remap_to_zero:
            rewrite_label_to_single_class(lp, dst_lbl)
        else:
            if lp.exists():
                shutil.copy2(lp, dst_lbl)
            else:
                dst_lbl.write_text("")

    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Roboflow export root (contains train/valid/test)")
    ap.add_argument("--out", required=True, help="Output dataset root (YOLO format)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-classes", type=int, default=1)
    ap.add_argument("--class-names", default="cone")
    ap.add_argument("--force_resplit", action="store_true",
                    help="Ignore existing splits and create new random split from all images.")
    ap.add_argument("--splits", default="0.8,0.1,0.1", help="train,val,test fractions if --force_resplit")
    ap.add_argument("--remap_all_classes_to_zero", action="store_true",
                    help="Force single-class dataset by mapping any class id to 0.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    items = list_roboflow_items(raw_root)
    if not items:
        raise SystemExit(f"No images found under {raw_root} (expected train/valid/test/images)")

    # Exact duplicate detection across all images (sha1)
    hash_map = defaultdict(list)
    for _, imgp, _ in tqdm(items, desc="Hashing images (sha1)"):
        h = sha1_file(imgp)
        hash_map[h].append(str(imgp))
    duplicates_exact = {h: ps for h, ps in hash_map.items() if len(ps) > 1}

    bad = []
    if args.force_resplit:
        pool = [(imgp, lp) for _, imgp, lp in items]
        random.shuffle(pool)

        s_train, s_val, s_test = map(float, args.splits.split(","))
        if abs((s_train + s_val + s_test) - 1.0) > 1e-6:
            raise SystemExit(f"--splits must sum to 1.0, got {args.splits}")

        n = len(pool)
        n_train = int(n * s_train)
        n_val = int(n * s_val)

        train = pool[:n_train]
        val = pool[n_train:n_train + n_val]
        test = pool[n_train + n_val:]

        def wrap(split_items):
            return [("pool", imgp, lp) for (imgp, lp) in split_items]

        bad += copy_split(wrap(train), out_root, "train", args.num_classes, args.remap_all_classes_to_zero)
        bad += copy_split(wrap(val), out_root, "val", args.num_classes, args.remap_all_classes_to_zero)
        bad += copy_split(wrap(test), out_root, "test", args.num_classes, args.remap_all_classes_to_zero)

        split_counts = {"train": len(train), "val": len(val), "test": len(test)}
        split_mode = "resplit_random"
    else:
        # Respect Roboflow splits if present
        by_split = defaultdict(list)
        for split, imgp, lp in items:
            s = "val" if split in ("valid", "val") else split
            by_split[s].append((split, imgp, lp))

        for s in ("train", "val", "test"):
            if s in by_split:
                bad += copy_split(by_split[s], out_root, s, args.num_classes, args.remap_all_classes_to_zero)

        split_counts = {k: len(v) for k, v in by_split.items()}
        split_mode = "roboflow_original"

    # data.yaml
    names = [n.strip() for n in args.class_names.split(",")]
    data_yaml = f"""path: {out_root.resolve()}
train: images/train
val: images/val
test: images/test
names: {names}
"""
    (out_root / "data.yaml").write_text(data_yaml)

    report = {
        "seed": args.seed,
        "split_mode": split_mode,
        "splits": split_counts,
        "num_images_total": len(items),
        "remap_all_classes_to_zero": bool(args.remap_all_classes_to_zero),
        "num_duplicates_exact_sha1": len(duplicates_exact),
        "duplicates_exact_sha1_sample": dict(list(duplicates_exact.items())[:50]),
        "num_bad_samples": len(bad),
        "bad_samples_sample": bad[:50],
    }
    (out_root / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()