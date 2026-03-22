#!/usr/bin/env python3
"""Run trained YOLO on processed test images; save JSONL predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, outputs_dir, processed_dirs


def video_frame_from_path(processed_images: Path, split: str, image_path: Path) -> tuple[str, str]:
    rel = image_path.relative_to(processed_images / split)
    video_id = str(rel.parent)
    frame_id = rel.stem
    return video_id, frame_id


def main() -> None:
    ap = argparse.ArgumentParser()
    _default = REPO_ROOT / "runs_ucfcrime" / "yolo11l_anomaly_region" / "weights" / "best.pt"
    _ultra = REPO_ROOT / "runs" / "detect" / "runs_ucfcrime" / "yolo11l_anomaly_region" / "weights" / "best.pt"
    ap.add_argument(
        "--weights",
        type=Path,
        default=_ultra if _ultra.is_file() else _default,
    )
    ap.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSONL output path (default: outputs/predictions/<split>_predictions.jsonl)",
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for smoke tests (processes only the first N images after sort).",
    )
    args = ap.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")

    images_root, _ = processed_dirs(REPO_ROOT)
    test_dir = images_root / args.split
    if not test_dir.is_dir():
        raise SystemExit(f"Missing image dir {test_dir}. Run convert_to_yolo.py first.")

    images = sorted(test_dir.rglob("*.jpg")) + sorted(test_dir.rglob("*.png"))
    if not images:
        raise SystemExit(f"No images under {test_dir}")
    if args.max_images is not None:
        images = images[: args.max_images]

    out_path = args.out or (outputs_dir(REPO_ROOT) / "predictions" / f"{args.split}_predictions.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    pred_kw: dict = {"imgsz": args.imgsz, "conf": args.conf, "verbose": False}
    if args.device is not None:
        pred_kw["device"] = args.device

    with open(out_path, "w", encoding="utf-8") as f:
        for img_path in images:
            results = model.predict(source=str(img_path), **pred_kw)
            r = results[0]
            boxes = r.boxes
            confs: list[float] = []
            xyxy: list[list[float]] = []
            if boxes is not None and len(boxes):
                confs = [float(x) for x in boxes.conf.cpu().numpy().tolist()]
                xyxy = [list(map(float, row)) for row in boxes.xyxy.cpu().numpy().tolist()]
            max_conf = max(confs) if confs else 0.0
            vid, fid = video_frame_from_path(images_root, args.split, img_path)
            rec = {
                "video_id": vid,
                "frame_id": fid,
                "image_path": str(img_path.resolve()),
                "num_boxes": len(confs),
                "max_confidence": max_conf,
                "all_confidences": confs,
                "all_boxes": xyxy,
            }
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(images)} rows to {out_path}")


if __name__ == "__main__":
    main()
