#!/usr/bin/env python3
"""Reduce X7 SDK beam components to a log-amplitude RA matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def components_to_log_ra(components: np.ndarray, near_bins: int = 19) -> np.ndarray:
    if components.shape != (37, 2, 192):
        raise ValueError(f"expected a 37x2x192 array, got {components.shape}")
    if not np.isfinite(components).all():
        raise ValueError("input contains non-finite values")
    if not 0 <= near_bins <= components.shape[2]:
        raise ValueError("near_bins must be between 0 and 192")

    filtered = components.copy()
    filtered[:, :, :near_bins] = 0
    amplitude = np.max(np.abs(filtered), axis=1)
    return np.log1p(amplitude).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="37x2x192 SDK beam components in .npy format")
    parser.add_argument("output", type=Path, help="destination 37x192 log-amplitude .npy")
    parser.add_argument("--near-bins", type=int, default=19)
    args = parser.parse_args()

    output = components_to_log_ra(np.load(args.input), args.near_bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)


if __name__ == "__main__":
    main()
