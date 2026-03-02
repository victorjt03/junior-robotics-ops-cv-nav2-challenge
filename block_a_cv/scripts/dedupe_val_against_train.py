#!/usr/bin/env python3
import argparse
import hashlib
import json
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

def list_images(d: Path):
    return [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Processed dataset root")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset)
    train_img = root / "images" / "train"
    val_img = root / "images" / "val"
    train_hashes = {sha1_file(p): p.name for p in list_images(train_img)}

    val_lbl = root / "labels" / "val"

    removed = []
    for p in list_images(val_img):
        h = sha1_file(p)
        if h in train_hashes:
            removed.append({"sha1": h, "val": p.name, "train": train_hashes[h]})
            if not args.dry_run:
                p.unlink(missing_ok=True)
                (val_lbl / (p.stem + ".txt")).unlink(missing_ok=True)

    report = {
        "dataset": str(root),
        "num_removed": len(removed),
        "removed_sample": removed[:30],
        "dry_run": bool(args.dry_run),
    }
    (root / "dedupe_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()