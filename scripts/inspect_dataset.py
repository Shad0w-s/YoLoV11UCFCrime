#!/usr/bin/env python3
"""Summarize UCF-Crime2Local layout: paths, classes, splits, counts."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    get_dataset_root,
    image_path_from_label_rel,
    label_path_from_label_rel,
    read_list_file,
    video_key_from_label_rel,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional markdown summary path (e.g. outputs/dataset_summary.md)",
    )
    args = ap.parse_args()

    ds = get_dataset_root()
    labels_txt = ds / "labels.txt"
    train_list = ds / "train.txt"
    test_list = ds / "test.txt"

    classes: list[str] = []
    if labels_txt.is_file():
        classes = [ln.strip() for ln in labels_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]

    train_lines = read_list_file(train_list) if train_list.is_file() else []
    test_lines = read_list_file(test_list) if test_list.is_file() else []

    train_videos = {video_key_from_label_rel(x) for x in train_lines}
    test_videos = {video_key_from_label_rel(x) for x in test_lines}
    overlap = train_videos & test_videos

    # Class histogram (sample up to 20k train lines for speed)
    cls_counts: Counter[int] = Counter()
    missing_img = 0
    missing_lbl = 0
    empty_lbl = 0
    for rel in train_lines[:20000]:
        lp = label_path_from_label_rel(ds, rel)
        ip = image_path_from_label_rel(ds, rel)
        if not lp.is_file():
            missing_lbl += 1
            continue
        if not ip.is_file():
            missing_img += 1
        txt = lp.read_text(encoding="utf-8").strip()
        if not txt:
            empty_lbl += 1
            continue
        for line in txt.splitlines():
            parts = line.split()
            if parts:
                cls_counts[int(parts[0])] += 1

    lines_out: list[str] = []
    lines_out.append("# UCF-Crime2Local inspection\n")
    lines_out.append(f"- **Dataset root:** `{ds}`\n")
    lines_out.append("- **Inputs:** extracted frames under `rgb-images/<class>/<video>/<frame>.jpg`.\n")
    lines_out.append("- **Labels:** YOLO txt per frame under `labels/...` (same relative path as list files).\n")
    lines_out.append(f"- **Classes ({len(classes)}):** " + (", ".join(f"`{i}:{c}`" for i, c in enumerate(classes)) if classes else "n/a") + "\n")
    lines_out.append(f"- **train.txt lines:** {len(train_lines)} (unique videos: {len(train_videos)})\n")
    lines_out.append(f"- **test.txt lines:** {len(test_lines)} (unique videos: {len(test_videos)})\n")
    lines_out.append(f"- **Train/test video overlap:** {len(overlap)} (should be 0)\n")
    lines_out.append("\n## Sample checks (first 20k train lines)\n\n")
    lines_out.append(f"- Missing label files: {missing_lbl}\n")
    lines_out.append(f"- Missing image files: {missing_img}\n")
    lines_out.append(f"- Empty label files: {empty_lbl}\n")
    lines_out.append(f"- Class id histogram (raw box rows): {dict(sorted(cls_counts.items()))}\n")

    text = "".join(lines_out)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
