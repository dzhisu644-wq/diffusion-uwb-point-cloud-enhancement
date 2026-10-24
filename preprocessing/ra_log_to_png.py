#!/usr/bin/env python3
"""Convert an SDK beamformed log-amplitude RA matrix to the released PNG grid.

The input is a NumPy array with shape (37, 192). Static-background removal,
near-field suppression, beamforming, I/Q magnitude reduction, and log1p
compression occur upstream in the X7 acquisition pipeline. This script
reproduces the deterministic post-beamforming raster conversion used for the
released dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def rasterize(matrix: np.ndarray, clip_min: float = 0.0, clip_max: float = 3.0) -> np.ndarray:
    if matrix.shape != (37, 192):
        raise ValueError(f"expected a 37x192 matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("input contains non-finite values")
    if clip_max <= clip_min:
        raise ValueError("clip_max must be greater than clip_min")

    normalized = np.clip(matrix, clip_min, clip_max)
    normalized = (normalized - clip_min) / (clip_max - clip_min)
    image = np.rint(normalized * 255.0).astype(np.uint8).T
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    return cv2.flip(image, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="37x192 log-amplitude matrix in .npy format")
    parser.add_argument("output", type=Path, help="destination 128x128 grayscale PNG")
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=3.0)
    args = parser.parse_args()

    output = rasterize(np.load(args.input), args.clip_min, args.clip_max)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise OSError(f"failed to write {args.output}")


if __name__ == "__main__":
    main()

