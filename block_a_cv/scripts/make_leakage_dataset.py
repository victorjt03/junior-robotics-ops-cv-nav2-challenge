#!/usr/bin/env python3
import argparse
import json
import random
import shutil
import hashlib
from pathlib import Path

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

def copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

def list_images(img_dir: Path):
    return [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Processed dataset root (has images/train,val,test and labels/...)")
    ap.add_argument("--dst", required=True, help="Output dataset root with induced leakage")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=30, help="How many train images to copy into val")
    args = ap.parse_args()

    random.seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)

    # 1) Copy full dataset
    copy_tree(src, dst)

    train_img = dst / "images" / "train"
    train_lbl = dst / "labels" / "train"
    val_img = dst / "images" / "val"
    val_lbl = dst / "labels" / "val"

    train_imgs = list_images(train_img)
    if len(train_imgs) == 0:
        raise SystemExit("No train images found in dst dataset")

    k = min(args.k, len(train_imgs))
    chosen = random.sample(train_imgs, k=k)

    leaked = []
    for imgp in chosen:
        lblp = train_lbl / (imgp.stem + ".txt")

        dst_img = val_img / imgp.name
        dst_lbl = val_lbl / (imgp.stem + ".txt")

        # If name collision exists, skip
        if dst_img.exists():
            continue

        shutil.copy2(imgp, dst_img)
        if lblp.exists():
            shutil.copy2(lblp, dst_lbl)
        else:
            dst_lbl.write_text("")

        leaked.append(imgp.name)

    # 2) Compute exact duplicates between train and val using sha1
    train_hash = {}
    for p in list_images(train_img):
        train_hash[sha1_file(p)] = p.name

    dup_pairs = []
    for p in list_images(val_img):
        h = sha1_file(p)
        if h in train_hash:
            dup_pairs.append({"sha1": h, "train": train_hash[h], "val": p.name})

    report = {
        "seed": args.seed,
        "k_requested": args.k,
        "k_leaked_effective": len(leaked),
        "leaked_filenames_sample": leaked[:20],
        "num_exact_duplicates_train_val": len(dup_pairs),
        "duplicate_pairs_sample": dup_pairs[:20],
        "note": "This dataset intentionally introduces leakage by duplicating train samples into val.",
    }
    (dst / "leakage_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()