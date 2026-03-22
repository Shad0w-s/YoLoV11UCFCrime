#!/usr/bin/env python3
"""
Frame-level ROC-AUC and AP from frame_scores_yolo.csv + optional binary labels CSV.
Without labels, prints a message and exits 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, outputs_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scores",
        type=Path,
        default=outputs_dir(REPO_ROOT) / "predictions" / "frame_scores_yolo.csv",
    )
    ap.add_argument(
        "--frame-labels",
        type=Path,
        default=None,
        help="CSV with columns frame_path,label (0/1) OR video_id,frame_id,label",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=outputs_dir(REPO_ROOT) / "metrics" / "metrics_anomaly.json",
    )
    args = ap.parse_args()

    if not args.scores.is_file():
        raise SystemExit(f"Missing {args.scores}. Run build_frame_scores.py first.")
    if args.frame_labels is None or not args.frame_labels.is_file():
        print(
            "No --frame-labels provided (or file missing). "
            "Skipping ROC-AUC/AP; frame scores are still available for your VadCLIP project."
        )
        return

    scores = pd.read_csv(args.scores)
    labels = pd.read_csv(args.frame_labels)

    if "frame_path" in labels.columns and "label" in labels.columns:
        merged = scores.merge(labels[["frame_path", "label"]], on="frame_path", how="inner")
    elif {"video_id", "frame_id", "label"}.issubset(labels.columns):
        merged = scores.merge(labels[["video_id", "frame_id", "label"]], on=["video_id", "frame_id"], how="inner")
    else:
        raise SystemExit("frame-labels CSV must have (frame_path,label) or (video_id,frame_id,label).")

    if merged.empty:
        raise SystemExit("No overlapping rows between scores and labels after merge.")

    y_true = merged["label"].astype(int).values
    y_score = merged["raw_score"].astype(float).values

    if len(set(y_true)) < 2:
        raise SystemExit(
            "Labels must contain both classes 0 and 1 for ROC-AUC/AP (got only one class)."
        )

    payload = {
        "n_frames": int(len(merged)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "scores_csv": str(args.scores.resolve()),
        "labels_csv": str(args.frame_labels.resolve()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
