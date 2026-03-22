#!/usr/bin/env python3
"""Copy rgb-images + remap all box classes to 0; build data/processed YOLO tree and data.yaml."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    REPO_ROOT,
    get_dataset_root,
    image_path_from_label_rel,
    label_path_from_label_rel,
    processed_dirs,
    read_list_file,
    split_dir,
    video_key_from_label_rel,
)


def remap_label_to_single_class(src_label: Path, dst_label: Path) -> None:
    lines_out: list[str] = []
    if src_label.is_file():
        raw = src_label.read_text(encoding="utf-8").strip()
        if raw:
            for line in raw.splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                xc, yc, w, h = map(float, parts[1:5])
                for v in (xc, yc, w, h):
                    if not (0.0 <= v <= 1.0):
                        raise ValueError(f"Invalid normalized coord in {src_label}: {line}")
                lines_out.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copy (faster, may confuse some tools on Windows).",
    )
    args = ap.parse_args()

    ds = get_dataset_root()
    sd = split_dir(REPO_ROOT)
    for name in ("train_videos.txt", "val_videos.txt", "test_videos.txt"):
        if not (sd / name).is_file():
            raise SystemExit(f"Missing {sd / name}. Run scripts/split_videos.py first.")

    train_v = set(read_list_file(sd / "train_videos.txt"))
    val_v = set(read_list_file(sd / "val_videos.txt"))
    test_v = set(read_list_file(sd / "test_videos.txt"))

    train_frames = read_list_file(ds / "train.txt")
    test_frames = read_list_file(ds / "test.txt")

    images_root, labels_root = processed_dirs(REPO_ROOT)
    if images_root.exists():
        shutil.rmtree(images_root)
    if labels_root.exists():
        shutil.rmtree(labels_root)

    def split_for(rel: str) -> str:
        vk = video_key_from_label_rel(rel)
        if vk in train_v:
            return "train"
        if vk in val_v:
            return "val"
        if vk in test_v:
            return "test"
        raise KeyError(vk)

    n = {"train": 0, "val": 0, "test": 0}
    missing = 0

    def process_list(frame_list: list[str]) -> None:
        nonlocal missing
        for rel in frame_list:
            try:
                sp = split_for(rel)
            except KeyError:
                missing += 1
                continue
            ip = image_path_from_label_rel(ds, rel)
            lp = label_path_from_label_rel(ds, rel)
            if not ip.is_file():
                missing += 1
                continue
            rel_img = Path(rel).with_suffix(".jpg")
            dst_img = images_root / sp / rel_img
            dst_lbl = labels_root / sp / Path(rel)
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if args.symlink:
                if dst_img.exists() or dst_img.is_symlink():
                    dst_img.unlink()
                dst_img.symlink_to(ip.resolve())
            else:
                shutil.copy2(ip, dst_img)
            if not lp.is_file():
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                dst_lbl.write_text("", encoding="utf-8")
            else:
                remap_label_to_single_class(lp, dst_lbl)
            n[sp] += 1

    process_list(train_frames)
    process_list(test_frames)

    data_yaml = {
        "path": str((REPO_ROOT / "data" / "processed").resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "anomaly_region"},
    }
    yaml_path = REPO_ROOT / "data" / "processed" / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print("Processed frames:", n)
    if missing:
        print(f"Skipped/missing entries: {missing}")
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
