#!/usr/bin/env python3
"""Pair processed UWB and LiDAR PNGs by nearest timestamp."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def read_timestamps(path: Path) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    timestamps: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected '<filename> <timestamp_ms>'")
        names.append(fields[0])
        timestamps.append(int(fields[1]))
    if not names:
        raise ValueError(f"empty timestamp file: {path}")
    return names, np.asarray(timestamps, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radar-dir", type=Path, required=True)
    parser.add_argument("--lidar-dir", type=Path, required=True)
    parser.add_argument("--radar-timestamps", type=Path, required=True)
    parser.add_argument("--lidar-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gap-ms", type=int, default=50)
    args = parser.parse_args()

    radar_names, radar_times = read_timestamps(args.radar_timestamps)
    lidar_names, lidar_times = read_timestamps(args.lidar_timestamps)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")

    radar_output = args.output / "radar"
    lidar_output = args.output / "lidar"
    radar_output.mkdir(parents=True, exist_ok=True)
    lidar_output.mkdir(parents=True, exist_ok=True)

    matched = 0
    for radar_name, radar_time in zip(radar_names, radar_times):
        lidar_index = int(np.argmin(np.abs(lidar_times - radar_time)))
        if abs(int(lidar_times[lidar_index] - radar_time)) >= args.max_gap_ms:
            continue
        destination = f"{matched:06d}.png"
        shutil.copy2(args.radar_dir / radar_name, radar_output / destination)
        shutil.copy2(args.lidar_dir / lidar_names[lidar_index], lidar_output / destination)
        matched += 1
    print(f"paired {matched} frames in {args.output}")


if __name__ == "__main__":
    main()

