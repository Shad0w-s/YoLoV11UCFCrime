#!/usr/bin/env python3
"""Placeholder: UCF-Crime2Local ships pre-extracted frames under rgb-images/."""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(
        description="No-op: dataset already provides frames. Use ffmpeg manually if extracting from raw video."
    )
    p.parse_args()
    print(
        "This project expects `rgb-images/` in UCFCRIME2LOCAL_ROOT.\n"
        "Example ffmpeg (2 fps):\n"
        '  ffmpeg -i input.mp4 -vf fps=2 "out_dir/%06d.jpg"\n'
    )


if __name__ == "__main__":
    main()
