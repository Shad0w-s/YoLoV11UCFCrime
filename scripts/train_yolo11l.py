#!/usr/bin/env python3
"""
Train YOLO11l on data/processed/data.yaml.

Ultralytics API (see Context7 /ultralytics/ultralytics): YOLO('*.pt').train(
    data=..., epochs=..., imgsz=..., batch=..., device=..., project=..., name=..., ...
)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "processed" / "data.yaml")
    ap.add_argument("--model", type=str, default="yolo11l.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default=None, help="e.g. 0, cpu, mps")
    ap.add_argument("--project", type=str, default="runs_ucfcrime")
    ap.add_argument("--name", type=str, default="yolo11l_anomaly_region")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--cache",
        type=str,
        default="ram",
        choices=("false", "ram", "disk"),
        help="Cache images in RAM or on disk for faster training (recommended on MPS).",
    )
    ap.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Optional Ultralytics fraction of training images (0–1). Use for smoke tests on limited RAM/GPU.",
    )
    args = ap.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Missing dataset yaml: {args.data}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kw: dict = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "exist_ok": True,
    }
    if args.cache != "false":
        train_kw["cache"] = args.cache  # "ram" or "disk"
    if args.fraction is not None:
        train_kw["fraction"] = args.fraction
    if args.device is not None:
        train_kw["device"] = args.device

    model.train(**train_kw)


if __name__ == "__main__":
    main()
