#!/usr/bin/env python3
"""Verify image/label pairs, box validity; write overlay previews to outputs/debug_samples/."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, outputs_dir, processed_dirs


def parse_yolo_lines(text: str) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        if cls != 0:
            raise ValueError(f"Expected class 0, got {cls}")
        xc, yc, w, h = map(float, parts[1:5])
        for v in (xc, yc, w, h):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"Coord out of range: {line}")
        boxes.append((xc, yc, w, h))
    return boxes


def draw_boxes(img: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images_root, labels_root = processed_dirs(REPO_ROOT)
    train_img = images_root / "train"
    if not train_img.is_dir():
        raise SystemExit(f"Missing {train_img}. Run convert_to_yolo.py first.")

    all_images = sorted(train_img.rglob("*.jpg")) + sorted(train_img.rglob("*.png"))
    if not all_images:
        raise SystemExit("No images under train/")

    rng = random.Random(args.seed)
    pick = all_images if len(all_images) <= args.num_samples else rng.sample(all_images, args.num_samples)

    orphan_labels = 0
    bad = 0
    for img_path in all_images:
        rel = img_path.relative_to(images_root / "train")
        lbl = labels_root / "train" / rel.with_suffix(".txt")
        if not lbl.is_file():
            orphan_labels += 1
            bad += 1
            continue
        txt = lbl.read_text(encoding="utf-8").strip()
        try:
            parse_yolo_lines(txt) if txt else []
        except ValueError:
            bad += 1

    out_dir = outputs_dir(REPO_ROOT) / "debug_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in pick:
        rel = img_path.relative_to(images_root / "train")
        lbl = labels_root / "train" / rel.with_suffix(".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes: list[tuple[float, float, float, float]] = []
        if lbl.is_file():
            boxes = parse_yolo_lines(lbl.read_text(encoding="utf-8").strip())
        vis = draw_boxes(img, boxes)
        out_path = out_dir / f"preview_{rel.name}"
        cv2.imwrite(str(out_path), vis)

    print(f"Train images scanned: {len(all_images)}")
    print(f"Missing label files: {orphan_labels}")
    print(f"Invalid label files (counted): {bad}")
    print(f"Wrote {len(pick)} previews to {out_dir}")


if __name__ == "__main__":
    main()
