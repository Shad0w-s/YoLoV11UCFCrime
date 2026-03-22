#!/usr/bin/env python3
"""
Official test.txt -> test videos. Official train.txt -> train/val by video (no frame leakage).
Default: ~85% train / ~15% val of train videos (seeded).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, get_dataset_root, read_list_file, split_dir, video_key_from_label_rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of official train videos assigned to validation (default 0.15).",
    )
    args = ap.parse_args()

    ds = get_dataset_root()
    train_lines = read_list_file(ds / "train.txt")
    test_lines = read_list_file(ds / "test.txt")

    train_videos = sorted({video_key_from_label_rel(x) for x in train_lines})
    test_videos = sorted({video_key_from_label_rel(x) for x in test_lines})

    inter = set(train_videos) & set(test_videos)
    if inter:
        raise SystemExit(f"Train and test share videos (unexpected): {sorted(inter)[:5]} ...")

    rng = random.Random(args.seed)
    rng.shuffle(train_videos)
    n_val = max(1, int(round(len(train_videos) * args.val_fraction)))
    if n_val >= len(train_videos):
        n_val = len(train_videos) - 1
    val_set = set(train_videos[:n_val])
    tr_set = set(train_videos[n_val:])

    out = split_dir(REPO_ROOT)
    out.mkdir(parents=True, exist_ok=True)

    (out / "train_videos.txt").write_text("\n".join(sorted(tr_set)) + "\n", encoding="utf-8")
    (out / "val_videos.txt").write_text("\n".join(sorted(val_set)) + "\n", encoding="utf-8")
    (out / "test_videos.txt").write_text("\n".join(test_videos) + "\n", encoding="utf-8")

    print(f"Train videos: {len(tr_set)}")
    print(f"Val videos:   {len(val_set)}")
    print(f"Test videos:  {len(test_videos)}")
    print(f"Wrote lists under {out}")


if __name__ == "__main__":
    main()
