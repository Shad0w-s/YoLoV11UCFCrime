#!/usr/bin/env python3
"""Print a Markdown comparison table from YOLO metrics_anomaly.json and optional VadCLIP JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, outputs_dir


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--yolo",
        type=Path,
        default=outputs_dir(REPO_ROOT) / "metrics" / "metrics_anomaly.json",
        help="JSON metrics from evaluate_anomaly.py (with frame labels).",
    )
    ap.add_argument("--vadclip", type=Path, default=None, help="Optional JSON from your VadCLIP project.")
    args = ap.parse_args()

    if not args.yolo.is_file():
        raise SystemExit(f"Missing YOLO metrics: {args.yolo} (run evaluate_anomaly.py with labels first).")

    y = load_json(args.yolo)
    rows = [
        "| Model | Type | ROC-AUC | AP |",
        "| --- | --- | ---: | ---: |",
    ]
    if args.vadclip and args.vadclip.is_file():
        v = load_json(args.vadclip)
        rows.append(
            f"| VadCLIP | anomaly | {v.get('roc_auc', 'n/a')} | {v.get('average_precision', 'n/a')} |"
        )
    rows.append(
        f"| YOLO11l | detection-derived | {y.get('roc_auc', 'n/a')} | {y.get('average_precision', 'n/a')} |"
    )

    md = "\n".join(rows) + "\n"
    print(md)


if __name__ == "__main__":
    main()
