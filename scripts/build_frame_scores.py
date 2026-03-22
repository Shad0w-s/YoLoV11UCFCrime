#!/usr/bin/env python3
"""Build frame_scores_yolo.csv from JSONL predictions (raw + optional smoothed scores)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, outputs_dir


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or not values:
        return values
    out: list[float] = []
    half = window // 2
    n = len(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions",
        type=Path,
        default=outputs_dir(REPO_ROOT) / "predictions" / "test_predictions.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=outputs_dir(REPO_ROOT) / "predictions" / "frame_scores_yolo.csv",
    )
    ap.add_argument("--smooth-window", type=int, default=5)
    args = ap.parse_args()

    if not args.predictions.is_file():
        raise SystemExit(f"Missing {args.predictions}. Run infer_yolo.py first.")

    by_video: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    with open(args.predictions, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            vid = row["video_id"]
            fid = row["frame_id"]
            path = row["image_path"]
            raw = float(row["max_confidence"])
            by_video[vid].append((fid, path, raw))

    out_rows: list[dict[str, str | float]] = []
    for vid, items in by_video.items():
        items.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
        raws = [t[2] for t in items]
        smoothed = moving_average(raws, args.smooth_window)
        for i, (fid, path, raw) in enumerate(items):
            out_rows.append(
                {
                    "video_id": vid,
                    "frame_id": fid,
                    "frame_path": path,
                    "raw_score": raw,
                    "smoothed_score": smoothed[i],
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["video_id", "frame_id", "frame_path", "raw_score", "smoothed_score"],
        )
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
