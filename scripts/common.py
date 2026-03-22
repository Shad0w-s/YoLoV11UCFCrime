"""Shared helpers for UCF-Crime2Local YOLO pipeline."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_dataset_root() -> Path:
    root = os.environ.get("UCFCRIME2LOCAL_ROOT")
    if not root:
        raise SystemExit(
            "Set UCFCRIME2LOCAL_ROOT to the UCF-Crime2Local directory "
            "(contains rgb-images/, labels/, train.txt, test.txt)."
        )
    p = Path(root).expanduser().resolve()
    if not p.is_dir():
        raise SystemExit(f"UCFCRIME2LOCAL_ROOT is not a directory: {p}")
    return p


def video_key_from_label_rel(label_rel: str) -> str:
    """label_rel: 'Arrest/Arrest002/00288.txt' -> 'Arrest/Arrest002'."""
    p = Path(label_rel.strip())
    return str(p.parent)


def frame_stem_from_label_rel(label_rel: str) -> str:
    """Return '00288' from 'Arrest/Arrest002/00288.txt'."""
    return Path(label_rel.strip()).stem


def image_path_from_label_rel(ds: Path, label_rel: str) -> Path:
    """rgb-images/<Category>/<Video>/<frame>.jpg"""
    p = Path(label_rel.strip())
    return ds / "rgb-images" / p.with_suffix(".jpg")


def label_path_from_label_rel(ds: Path, label_rel: str) -> Path:
    return ds / "labels" / label_rel.strip()


def processed_dirs(repo: Path) -> tuple[Path, Path]:
    base = repo / "data" / "processed"
    return base / "images", base / "labels"


def split_dir(repo: Path) -> Path:
    return repo / "data" / "splits"


def outputs_dir(repo: Path) -> Path:
    return repo / "outputs"


def read_list_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]
