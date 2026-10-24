#!/usr/bin/env python3
"""Benchmark batch-one EDM or CD latency with a fixed inference protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from evaluate_image_checkpoints import (
    PairedSceneDataset,
    model_and_diffusion,
    polar_to_points,
    resolve_inference_protocol,
    sha256_file,
)


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


def summarize_ms(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty timing sequence")
    mean = statistics.fmean(values)
    return {
        "iterations": len(values),
        "mean_ms": mean,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": statistics.pstdev(values),
        "mean_hz": 1000.0 / mean,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the latency benchmark")
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("--warmup and --iterations must both be positive")
    gpu_name = torch.cuda.get_device_name(0)
    if args.expected_gpu_name and args.expected_gpu_name.lower() not in gpu_name.lower():
        raise RuntimeError(
            f"expected GPU containing {args.expected_gpu_name!r}, found {gpu_name!r}"
        )

    args.training_bundle = args.training_bundle.resolve()
    args.diffusion_root = args.training_bundle / "diffusion_X7"
    args.dataset_root = args.training_bundle / "X7_A1_dataset"
    args.sampler, args.sampling_steps = resolve_inference_protocol(
        args.model_type, args.sampler, args.sampling_steps
    )
    if args.update is None:
        args.update = 120000 if args.model_type == "cd" else 160000
    args.run_dir = (
        args.training_bundle / "runs" / f"{args.model_type}_{args.frames}f_full_v4"
    )
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = args.run_dir / f"ema_0.9999_{args.update:06d}.pt"
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    output_root = args.output_root
    if output_root is None:
        output_root = (
            args.training_bundle
            / "evaluation_results"
            / f"{args.model_type}_{args.frames}f"
            / f"latency_seed{args.seed}"
        )
    output_root = output_root.resolve()
    artifact = (
        f"{args.model_type}_{args.frames}f_latency_ema0.9999_"
        f"step{args.update:06d}_seed{args.seed}"
    )
    summary_path = output_root / f"{artifact}_latency_summary.json"
    flat_summary_path = output_root / f"{artifact}_latency_summary.csv"
    raw_path = output_root / f"{artifact}_latency_raw.csv"
    if summary_path.is_file() and flat_summary_path.is_file() and not args.force:
        print(f"Skipping completed latency benchmark: {summary_path}")
        return

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    dataset = PairedSceneDataset(
        args.dataset_root,
        frames=args.frames,
        common_history=args.common_history,
        scenes=None,
        max_frames_per_scene=1,
        include_boundary_predictions=False,
    )
    condition_cpu, _, _ = dataset[0]
    condition_cpu = condition_cpu.unsqueeze(0).contiguous()

    model, diffusion = model_and_diffusion(args)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    del state
    model.to(device).eval()

    from cm.karras_diffusion import karras_sample

    def infer(condition: torch.Tensor) -> torch.Tensor:
        return karras_sample(
            diffusion=diffusion,
            model=model,
            shape=(1, 1, 128, 128),
            steps=args.sampling_steps,
            model_kwargs={"y": condition},
            device=device,
            clip_denoised=True,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            last_sample_result_list=[],
        )

    with torch.inference_mode():
        condition_device = condition_cpu.to(device)
        for _ in range(args.warmup):
            infer(condition_device)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(device)
        model_times: list[float] = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            infer(condition_device)
            end.record()
            end.synchronize()
            model_times.append(float(start.elapsed_time(end)))
        model_peak_bytes = torch.cuda.max_memory_allocated(device)

        torch.cuda.reset_peak_memory_stats(device)
        pipeline_rows: list[dict[str, object]] = []
        for iteration in range(args.iterations):
            wall_start = time.perf_counter_ns()
            condition = condition_cpu.to(device, non_blocking=False)
            model_start = torch.cuda.Event(enable_timing=True)
            model_end = torch.cuda.Event(enable_timing=True)
            model_start.record()
            sample = infer(condition)
            model_end.record()
            model_end.synchronize()
            model_ms = float(model_start.elapsed_time(model_end))

            prediction = (
                (sample[0, 0] * 255)
                .clamp(0, 255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            points = polar_to_points(prediction, args.pixel_threshold, args.max_range_m)
            end_to_end_ms = (time.perf_counter_ns() - wall_start) / 1e6
            pipeline_rows.append(
                {
                    "iteration": iteration,
                    "model_cuda_ms": model_ms,
                    "end_to_end_wall_ms": end_to_end_ms,
                    "output_points": len(points),
                }
            )
        pipeline_peak_bytes = torch.cuda.max_memory_allocated(device)

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(raw_path, pipeline_rows)
    model_summary = summarize_ms(model_times)
    pipeline_model_summary = summarize_ms(
        [float(row["model_cuda_ms"]) for row in pipeline_rows]
    )
    end_to_end_summary = summarize_ms(
        [float(row["end_to_end_wall_ms"]) for row in pipeline_rows]
    )
    checkpoint_hash = sha256_file(checkpoint)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "protocol": (
            "final_cd_latency_v1"
            if args.model_type == "cd"
            else "in_domain_frame_count_ablation_v1"
        ),
        "artifact_id": artifact,
        "model_type": args.model_type,
        "condition_total_frames": args.frames,
        "history_frames": args.frames - 1,
        "ema_rate": "0.9999",
        "update": args.update,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "batch_size": 1,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "sampler": args.sampler,
        "sampling_steps": args.sampling_steps,
        "precision": "fp32",
        "parameter_count": parameter_count,
        "condition_buffer_bytes": condition_cpu.numel() * condition_cpu.element_size(),
        "model_peak_memory_bytes": model_peak_bytes,
        "pipeline_peak_memory_bytes": pipeline_peak_bytes,
        "model_only": model_summary,
        "pipeline_model": pipeline_model_summary,
        "end_to_end": end_to_end_summary,
        "scope": {
            "model_only": "CUDA sampling with a device-resident batch-one condition",
            "end_to_end": "CPU condition H2D + CUDA sampling + D2H + polar-image-to-point conversion",
            "excluded": "sensor acquisition, disk I/O, ROS transport, output smoothing, and SLAM",
        },
        "raw_csv": str(raw_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    flat_row = {
        "artifact_id": artifact,
        "model_type": args.model_type,
        "condition_total_frames": args.frames,
        "history_frames": args.frames - 1,
        "ema_rate": "0.9999",
        "update": args.update,
        "seed": args.seed,
        "checkpoint_sha256": checkpoint_hash,
        "gpu": summary["gpu"],
        "sampler": args.sampler,
        "sampling_steps": args.sampling_steps,
        "precision": "fp32",
        "parameter_count": parameter_count,
        "condition_buffer_bytes": summary["condition_buffer_bytes"],
        "model_peak_memory_gib": model_peak_bytes / (1024**3),
        "pipeline_peak_memory_gib": pipeline_peak_bytes / (1024**3),
        "model_mean_ms": model_summary["mean_ms"],
        "model_p95_ms": model_summary["p95_ms"],
        "end_to_end_mean_ms": end_to_end_summary["mean_ms"],
        "end_to_end_p95_ms": end_to_end_summary["p95_ms"],
        "end_to_end_mean_hz": end_to_end_summary["mean_hz"],
    }
    write_csv(flat_summary_path, [flat_row])
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--training-bundle", type=Path, default=Path(__file__).resolve().parents[1]
    )
    result.add_argument("--model-type", choices=("edm", "cd"), default="edm")
    result.add_argument("--frames", type=int, choices=(1, 3, 5, 8, 10), required=True)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--output-root", type=Path)
    result.add_argument("--update", type=int)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--warmup", type=int, default=100)
    result.add_argument("--iterations", type=int, default=500)
    result.add_argument("--common-history", type=int, default=9)
    result.add_argument("--sampler")
    result.add_argument("--sampling-steps", type=int)
    result.add_argument("--sigma-min", type=float, default=0.002)
    result.add_argument("--sigma-max", type=float, default=80.0)
    result.add_argument("--pixel-threshold", type=int, default=150)
    result.add_argument("--max-range-m", type=float, default=4.0)
    result.add_argument(
        "--expected-gpu-name",
        default="",
        help="Fail unless CUDA device 0 contains this name (for example RTX 3090)",
    )
    result.add_argument("--force", action="store_true")
    return result


if __name__ == "__main__":
    benchmark(parser().parse_args())
