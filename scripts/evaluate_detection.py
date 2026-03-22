#!/usr/bin/env python3
"""Run Ultralytics validation; write precision/recall/mAP to metrics_detection.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, outputs_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    _default = REPO_ROOT / "runs_ucfcrime" / "yolo11l_anomaly_region" / "weights" / "best.pt"
    _ultra = REPO_ROOT / "runs" / "detect" / "runs_ucfcrime" / "yolo11l_anomaly_region" / "weights" / "best.pt"
    ap.add_argument(
        "--weights",
        type=Path,
        default=_ultra if _ultra.is_file() else _default,
    )
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "processed" / "data.yaml")
    ap.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=outputs_dir(REPO_ROOT) / "metrics" / "metrics_detection.json",
    )
    args = ap.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    if not args.data.is_file():
        raise SystemExit(f"Missing {args.data}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    val_kw: dict = {
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
    }
    if args.device is not None:
        val_kw["device"] = args.device

    metrics = model.val(**val_kw)

    box = metrics.box
    # Ultralytics Box: mp/mr (means) vs p/r — prefer mp/mr when present
    prec = float(getattr(box, "mp", getattr(box, "p", float("nan"))))
    rec = float(getattr(box, "mr", getattr(box, "r", float("nan"))))
    payload = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "precision": prec,
        "recall": rec,
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
