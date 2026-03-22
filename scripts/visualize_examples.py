#!/usr/bin/env python3
"""Save qualitative figures: predictions overlaid; optional TP/FP/FN/TN if labels CSV given."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, outputs_dir


def draw_xyxy(img: np.ndarray, boxes: list[list[float]], color: tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions",
        type=Path,
        default=outputs_dir(REPO_ROOT) / "predictions" / "test_predictions.jsonl",
    )
    ap.add_argument(
        "--frame-labels",
        type=Path,
        default=None,
        help="Optional CSV with frame_path,label for quadrant sampling",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--per-bin", type=int, default=5)
    ap.add_argument(
        "--out",
        type=Path,
        default=outputs_dir(REPO_ROOT) / "figures",
    )
    args = ap.parse_args()

    if not args.predictions.is_file():
        raise SystemExit(f"Missing {args.predictions}. Run infer_yolo.py first.")

    rows: list[dict] = []
    with open(args.predictions, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    args.out.mkdir(parents=True, exist_ok=True)

    if args.frame_labels is None or not args.frame_labels.is_file():
        # Save a handful of random high/low confidence examples
        rows_sorted = sorted(rows, key=lambda r: r["max_confidence"], reverse=True)
        pick = rows_sorted[: args.per_bin] + rows_sorted[-args.per_bin :]
        for i, r in enumerate(pick):
            img = cv2.imread(r["image_path"])
            if img is None:
                continue
            vis = draw_xyxy(img, r.get("all_boxes", []), (0, 255, 0))
            cv2.putText(
                vis,
                f"conf={r['max_confidence']:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imwrite(str(args.out / f"sample_{i:02d}.jpg"), vis)
        print(f"Wrote examples to {args.out} (no labels CSV; not TP/FP/FN/TN).")
        return

    labels = pd.read_csv(args.frame_labels)
    if "frame_path" not in labels.columns or "label" not in labels.columns:
        raise SystemExit("Labels CSV must include frame_path and label for visualization.")

    lab_map = dict(zip(labels["frame_path"].astype(str), labels["label"].astype(int)))

    bins: dict[str, list[dict]] = {"tp": [], "fp": [], "fn": [], "tn": []}
    for r in rows:
        pth = r["image_path"]
        pred = 1 if r["max_confidence"] >= args.threshold else 0
        gt = lab_map.get(pth)
        if gt is None:
            continue
        gt = int(gt)
        key = {(1, 1): "tp", (1, 0): "fp", (0, 1): "fn", (0, 0): "tn"}[(pred, gt)]
        bins[key].append(r)

    for name, items in bins.items():
        for i, r in enumerate(items[: args.per_bin]):
            img = cv2.imread(r["image_path"])
            if img is None:
                continue
            vis = draw_xyxy(img, r.get("all_boxes", []), (0, 255, 0))
            cv2.putText(vis, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imwrite(str(args.out / f"{name}_{i:02d}.jpg"), vis)

    print(f"Wrote quadrant samples (up to {args.per_bin} each) to {args.out}")


if __name__ == "__main__":
    main()
