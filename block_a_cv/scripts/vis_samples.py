#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def read_labels(lbl_path: Path):
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((cls, cx, cy, w, h))
    return boxes

def draw_boxes(img, boxes):
    H, W = img.shape[:2]
    for cls, cx, cy, w, h in boxes:
        x1 = int((cx - w/2) * W)
        y1 = int((cy - h/2) * H)
        x2 = int((cx + w/2) * W)
        y2 = int((cy + h/2) * H)
        x1 = max(0, min(W-1, x1))
        y1 = max(0, min(H-1, y1))
        x2 = max(0, min(W-1, x2))
        y2 = max(0, min(H-1, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{cls}", (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    return img

def make_grid(images, cols=5, pad=6):
    assert images
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    rows = (len(images) + cols - 1) // cols
    grid = np.zeros((rows*h + (rows+1)*pad, cols*w + (cols+1)*pad, 3), dtype=np.uint8)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(images):
                break
            im = images[idx]
            y = pad + r*(h+pad)
            x = pad + c*(w+pad)
            grid[y:y+im.shape[0], x:x+im.shape[1]] = im
            idx += 1
    return grid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Processed dataset root (contains images/ labels/)")
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="block_a_cv/docs/samples_overlay.jpg")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    root = Path(args.data)
    img_dir = root / "images" / args.split
    lbl_dir = root / "labels" / args.split
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    imgs.sort()
    chosen = random.sample(imgs, k=min(args.n, len(imgs)))

    rendered = []
    for imgp in chosen:
        img = cv2.imread(str(imgp))
        if img is None:
            continue
        lblp = lbl_dir / (imgp.stem + ".txt")
        boxes = read_labels(lblp)
        img = draw_boxes(img, boxes)
        img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
        rendered.append(img)

    grid = make_grid(rendered, cols=5)
    cv2.imwrite(str(out_path), grid)
    print(f"Wrote: {out_path} ({grid.shape[1]}x{grid.shape[0]})")

if __name__ == "__main__":
    main()