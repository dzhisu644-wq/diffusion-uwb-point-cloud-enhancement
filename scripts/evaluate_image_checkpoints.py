#!/usr/bin/env python3
"""Evaluate EDM checkpoints directly on paired X7 radar/LiDAR PNG scenes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset


CHECKPOINT_RE = re.compile(r"^ema_(0\.9999|0\.999)_(\d{6})\.pt$")


@dataclass(frozen=True)
class FrameRecord:
    scene: str
    dataset: str
    index: int
    filename: str
    metric_eligible: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def parse_csv_set(value: str, cast) -> set | None:
    if value.strip().lower() == "all":
        return None
    parsed = {cast(item.strip()) for item in value.split(",") if item.strip()}
    if not parsed:
        raise ValueError(f"empty selection: {value!r}")
    return parsed


def checkpoint_identity(path: Path) -> tuple[str, int]:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        raise ValueError(f"unsupported checkpoint name: {path.name}")
    return match.group(1), int(match.group(2))


def artifact_id(args: argparse.Namespace, ema_rate: str, update: int) -> str:
    return (
        f"{args.model_type}_{args.frames}f_{args.evaluation_tag}_ema{ema_rate}_"
        f"step{update:06d}_seed{args.seed}"
    )


def resolve_inference_protocol(
    model_type: str, sampler: str | None, sampling_steps: int | None
) -> tuple[str, int]:
    if model_type not in {"edm", "cd"}:
        raise ValueError(f"unsupported model type: {model_type}")
    resolved_sampler = sampler or ("onestep" if model_type == "cd" else "heun")
    resolved_steps = sampling_steps if sampling_steps is not None else (1 if model_type == "cd" else 7)
    if resolved_steps <= 0:
        raise ValueError("sampling steps must be positive")
    if resolved_sampler == "onestep" and resolved_steps != 1:
        raise ValueError("the onestep sampler requires exactly one sampling step")
    return resolved_sampler, resolved_steps


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    candidates = [path.resolve() for path in args.checkpoint]
    if not candidates:
        candidates = sorted(args.run_dir.resolve().glob("ema_*.pt"))
    ema_rates = parse_csv_set(args.ema_rates, str)
    updates = parse_csv_set(args.updates, int)
    selected = []
    for path in candidates:
        if not path.is_file() or not CHECKPOINT_RE.match(path.name):
            continue
        ema_rate, update = checkpoint_identity(path)
        if ema_rates is not None and ema_rate not in ema_rates:
            continue
        if updates is not None and update not in updates:
            continue
        selected.append(path)
    selected.sort(key=lambda path: (checkpoint_identity(path)[1], checkpoint_identity(path)[0]))
    if not selected:
        raise FileNotFoundError("no checkpoints matched the requested EMA rates and updates")
    return selected


def evenly_spaced(values: Sequence[int], count: int) -> list[int]:
    if count <= 0 or len(values) <= count:
        return list(values)
    positions = np.linspace(0, len(values) - 1, num=count, dtype=np.int64)
    return [values[int(position)] for position in positions]


class PairedSceneDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        frames: int,
        common_history: int,
        scenes: set[str] | None,
        max_frames_per_scene: int,
        include_boundary_predictions: bool,
    ) -> None:
        self.dataset_root = dataset_root
        self.frames = frames
        self.records: list[FrameRecord] = []
        self.scene_files: dict[str, list[str]] = {}
        scene_dirs = sorted(path for path in dataset_root.glob("dataset_aligned_*") if path.is_dir())
        for scene_dir in scene_dirs:
            scene = scene_dir.name.removeprefix("dataset_aligned_")
            if scenes is not None and scene not in scenes and scene_dir.name not in scenes:
                continue
            radar = sorted(path.name for path in (scene_dir / "radar").glob("*.png"))
            lidar = sorted(path.name for path in (scene_dir / "lidar").glob("*.png"))
            if not radar or radar != lidar:
                raise ValueError(f"invalid paired scene: {scene_dir}")
            self.scene_files[scene_dir.name] = radar
            valid_indices = list(range(common_history, len(radar)))
            selected_valid = evenly_spaced(valid_indices, max_frames_per_scene)
            selected_indices = selected_valid
            if include_boundary_predictions and max_frames_per_scene == 0:
                selected_indices = list(range(len(radar)))
            selected_set = set(selected_valid)
            for index in selected_indices:
                self.records.append(
                    FrameRecord(
                        scene=scene,
                        dataset=scene_dir.name,
                        index=index,
                        filename=radar[index],
                        metric_eligible=index in selected_set,
                    )
                )
        if not self.records:
            raise ValueError(f"no evaluation frames found under {dataset_root}")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def load_gray(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (128, 128):
            raise ValueError(f"invalid 128x128 grayscale PNG: {path}")
        return image

    def __getitem__(self, item: int):
        record = self.records[item]
        filenames = self.scene_files[record.dataset]
        scene_dir = self.dataset_root / record.dataset
        history = []
        for offset in range(self.frames - 1, -1, -1):
            history_index = max(0, record.index - offset)
            history.append(self.load_gray(scene_dir / "radar" / filenames[history_index]))
        condition = torch.from_numpy(np.stack(history)).float() / 255.0
        label = torch.from_numpy(self.load_gray(scene_dir / "lidar" / record.filename))
        return condition, label, item


def polar_to_points(image: np.ndarray, threshold: int, max_range_m: float) -> np.ndarray:
    rows, cols = np.where(image > threshold)
    if not len(rows):
        return np.empty((0, 2), dtype=np.float64)
    height, width = image.shape
    distances = (height - 1 - rows) * (max_range_m / float(height))
    normalized = cols / float(width)
    angles_deg = np.where(
        normalized < 0.5,
        -90.0 - normalized * 180.0,
        180.0 - (normalized - 0.5) * 180.0,
    )
    angles = (np.deg2rad(angles_deg) + np.pi) % (2.0 * np.pi) - np.pi
    return np.column_stack([distances * np.cos(angles), distances * np.sin(angles)])


def point_metrics(
    prediction: np.ndarray,
    label: np.ndarray,
    threshold: int,
    max_range_m: float,
    fscore_threshold_m: float,
) -> dict[str, object]:
    estimate = polar_to_points(prediction, threshold, max_range_m)
    reference = polar_to_points(label, threshold, max_range_m)
    base: dict[str, object] = {
        "est_points": len(estimate),
        "gt_points": len(reference),
        "status": "ok",
        "cd_m": math.nan,
        "hd_m": math.nan,
        "hd95_m": math.nan,
        "precision": math.nan,
        "recall": math.nan,
        "fscore": math.nan,
    }
    if not len(reference):
        base["status"] = "empty_gt"
        return base
    if not len(estimate):
        base.update(status="empty_est", precision=0.0, recall=0.0, fscore=0.0)
        return base
    gt_to_est = cKDTree(estimate).query(reference, k=1, workers=1)[0]
    est_to_gt = cKDTree(reference).query(estimate, k=1, workers=1)[0]
    precision = float(np.mean(est_to_gt <= fscore_threshold_m))
    recall = float(np.mean(gt_to_est <= fscore_threshold_m))
    fscore = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    base.update(
        cd_m=float(np.mean(gt_to_est) + np.mean(est_to_gt)),
        hd_m=float(max(np.max(gt_to_est), np.max(est_to_gt))),
        hd95_m=float(max(np.percentile(gt_to_est, 95), np.percentile(est_to_gt, 95))),
        precision=precision,
        recall=recall,
        fscore=fscore,
    )
    return base


def finite_values(rows: Sequence[dict[str, object]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row[key]
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def summarize_rows(rows: Sequence[dict[str, object]], scene: str) -> dict[str, object]:
    cd = finite_values(rows, "cd_m")
    hd95 = finite_values(rows, "hd95_m")
    precision = finite_values(rows, "precision")
    recall = finite_values(rows, "recall")
    fscore = finite_values(rows, "fscore")
    empty_est = sum(row["status"] == "empty_est" for row in rows)

    def mean_or_nan(values: Sequence[float]) -> float:
        return statistics.fmean(values) if values else math.nan

    def percentile_or_nan(values: Sequence[float], percentile: float) -> float:
        return float(np.percentile(values, percentile)) if values else math.nan

    return {
        "scene": scene,
        "evaluated_frames": len(rows),
        "finite_cd_frames": len(cd),
        "finite_fscore_frames": len(fscore),
        "empty_est_frames": empty_est,
        "empty_est_rate": empty_est / len(rows),
        "cd_mean_m": mean_or_nan(cd),
        "cd_median_m": statistics.median(cd) if cd else math.nan,
        "cd_p90_m": percentile_or_nan(cd, 90),
        "hd95_mean_m": mean_or_nan(hd95),
        "hd95_p90_m": percentile_or_nan(hd95, 90),
        "precision_mean": mean_or_nan(precision),
        "recall_mean": mean_or_nan(recall),
        "fscore_mean": mean_or_nan(fscore),
        "fscore_median": statistics.median(fscore) if fscore else math.nan,
        "fscore_p10": percentile_or_nan(fscore, 10),
    }


def bootstrap_scene_ci(values: Sequence[float], seed: int, samples: int = 2000) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(array, size=len(array), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_scene_weighted_ci(
    scene_rows: Sequence[dict[str, object]],
    value_key: str,
    weight_key: str,
    seed: int,
    samples: int = 2000,
) -> tuple[float, float]:
    valid: list[tuple[float, int]] = []
    for row in scene_rows:
        value = float(row[value_key])
        weight = int(row[weight_key])
        if math.isfinite(value) and weight > 0:
            valid.append((value, weight))
    if not valid:
        return math.nan, math.nan

    values = np.asarray([item[0] for item in valid], dtype=np.float64)
    weights = np.asarray([item[1] for item in valid], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(valid), size=len(valid))
        selected_weights = weights[selected]
        means[index] = np.average(values[selected], weights=selected_weights)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def micro_summary(
    frame_rows: Sequence[dict[str, object]],
    scene_rows: Sequence[dict[str, object]],
    seed: int,
) -> dict[str, object]:
    pooled = summarize_rows(frame_rows, "__all__")
    cd_low, cd_high = bootstrap_scene_weighted_ci(
        scene_rows, "cd_mean_m", "finite_cd_frames", seed
    )
    fs_low, fs_high = bootstrap_scene_weighted_ci(
        scene_rows, "fscore_mean", "finite_fscore_frames", seed + 1
    )
    return {
        "finite_cd_frames": pooled["finite_cd_frames"],
        "finite_fscore_frames": pooled["finite_fscore_frames"],
        "cd_frame_micro_mean_m": pooled["cd_mean_m"],
        "cd_frame_micro_median_m": pooled["cd_median_m"],
        "cd_frame_micro_p90_m": pooled["cd_p90_m"],
        "cd_frame_micro_cluster_ci95_low_m": cd_low,
        "cd_frame_micro_cluster_ci95_high_m": cd_high,
        "hd95_frame_micro_mean_m": pooled["hd95_mean_m"],
        "hd95_frame_micro_p90_m": pooled["hd95_p90_m"],
        "precision_frame_micro_mean": pooled["precision_mean"],
        "recall_frame_micro_mean": pooled["recall_mean"],
        "fscore_frame_micro_mean": pooled["fscore_mean"],
        "fscore_frame_micro_median": pooled["fscore_median"],
        "fscore_frame_micro_p10": pooled["fscore_p10"],
        "fscore_frame_micro_cluster_ci95_low": fs_low,
        "fscore_frame_micro_cluster_ci95_high": fs_high,
        "empty_est_frame_micro_rate": pooled["empty_est_rate"],
    }


def macro_summary(scene_rows: Sequence[dict[str, object]], seed: int) -> dict[str, object]:
    cd = finite_values(scene_rows, "cd_mean_m")
    fs = finite_values(scene_rows, "fscore_mean")
    cd_low, cd_high = bootstrap_scene_ci(cd, seed)
    fs_low, fs_high = bootstrap_scene_ci(fs, seed + 1)
    return {
        "scene_count": len(scene_rows),
        "finite_cd_scenes": len(cd),
        "finite_fscore_scenes": len(fs),
        "evaluated_frames": sum(int(row["evaluated_frames"]) for row in scene_rows),
        "cd_scene_macro_mean_m": statistics.fmean(cd) if cd else math.nan,
        "cd_scene_macro_median_m": statistics.median(cd) if cd else math.nan,
        "cd_scene_macro_ci95_low_m": cd_low,
        "cd_scene_macro_ci95_high_m": cd_high,
        "fscore_scene_macro_mean": statistics.fmean(fs) if fs else math.nan,
        "fscore_scene_macro_median": statistics.median(fs) if fs else math.nan,
        "fscore_scene_macro_ci95_low": fs_low,
        "fscore_scene_macro_ci95_high": fs_high,
        "empty_est_scene_macro_rate": statistics.fmean(
            float(row["empty_est_rate"]) for row in scene_rows
        ),
    }


def model_and_diffusion(args: argparse.Namespace):
    sys.path.insert(0, str(args.diffusion_root.resolve()))
    from cm.script_util_cond import create_model_and_diffusion

    return create_model_and_diffusion(
        image_size=128,
        class_cond=False,
        learn_sigma=False,
        num_channels=64,
        num_res_blocks=3,
        channel_mult="",
        in_ch=args.frames + 1,
        out_ch=1,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.0 if args.model_type == "cd" else 0.1,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        weight_schedule="karras",
        loss_norm="l2",
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        distillation=args.model_type == "cd",
    )


def evaluate_one(
    args: argparse.Namespace,
    dataset: PairedSceneDataset,
    loader: DataLoader,
    model,
    diffusion,
    checkpoint: Path,
    checkpoint_position: int,
    checkpoint_count: int,
) -> dict[str, object]:
    from cm.karras_diffusion import karras_sample

    ema_rate, update = checkpoint_identity(checkpoint)
    checkpoint_label = f"ema_{ema_rate}_{update:06d}"
    artifact = artifact_id(args, ema_rate, update)
    output_dir = args.output_root / artifact
    summary_path = output_dir / f"{artifact}_macro_summary.json"
    prediction_manifest = output_dir / f"{artifact}_prediction_manifest.csv"
    complete = summary_path.is_file() and (
        not args.save_predictions or prediction_manifest.is_file()
    )
    if complete and not args.force:
        print(f"Skipping completed checkpoint: {checkpoint_label}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    print(f"Loading {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    del state
    model.to(args.device).eval()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    checkpoint_started = time.monotonic()
    last_progress_report = checkpoint_started

    prediction_root = output_dir / "predictions"
    executor = ThreadPoolExecutor(max_workers=args.metric_workers)
    pending: list[tuple[Future, FrameRecord, str]] = []
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    def drain(limit: int) -> None:
        while len(pending) > limit:
            future, record, relative_prediction = pending.pop(0)
            metrics = future.result()
            if record.metric_eligible:
                metric_rows.append(
                    {
                        "checkpoint": checkpoint_label,
                        "ema_rate": ema_rate,
                        "update": update,
                        "scene": record.scene,
                        "dataset": record.dataset,
                        "frame_index": record.index,
                        "filename": record.filename,
                        **metrics,
                    }
                )
            if args.save_predictions:
                prediction_rows.append(
                    {
                        "checkpoint": checkpoint_label,
                        "scene": record.scene,
                        "dataset": record.dataset,
                        "frame_index": record.index,
                        "filename": record.filename,
                        "source_timestamp_s": "",
                        "prediction_file": relative_prediction,
                    }
                )

    processed = 0
    with torch.inference_mode():
        for condition, label, item_indices in loader:
            condition = condition.to(args.device, non_blocking=args.pin_memory)
            sample = karras_sample(
                diffusion=diffusion,
                model=model,
                shape=(condition.shape[0], 1, 128, 128),
                steps=args.sampling_steps,
                model_kwargs={"y": condition},
                device=args.device,
                clip_denoised=True,
                sampler=args.sampler,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                last_sample_result_list=[],
            )
            predictions = (sample[:, 0] * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            labels = label.numpy().astype(np.uint8, copy=False)
            for prediction, target, item_index in zip(predictions, labels, item_indices.tolist()):
                record = dataset.records[item_index]
                relative_prediction = ""
                save_path = None
                if args.save_predictions:
                    relative_prediction = f"predictions/{record.dataset}/{record.filename}"
                    save_path = output_dir / relative_prediction
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                def work(pred=prediction.copy(), gt=target.copy(), destination=save_path):
                    if destination is not None and not cv2.imwrite(str(destination), pred):
                        raise OSError(f"failed to save prediction: {destination}")
                    return point_metrics(
                        pred,
                        gt,
                        args.pixel_threshold,
                        args.max_range_m,
                        args.fscore_threshold_m,
                    )

                pending.append((executor.submit(work), record, relative_prediction))
            processed += len(item_indices)
            drain(args.batch_size * 4)
            now = time.monotonic()
            report_due = now - last_progress_report >= args.progress_interval_seconds
            if report_due or processed == len(dataset):
                checkpoint_elapsed = max(now - checkpoint_started, 1e-9)
                rate = processed / checkpoint_elapsed
                checkpoint_remaining = (len(dataset) - processed) / rate
                overall_completed = (checkpoint_position - 1) * len(dataset) + processed
                overall_total = checkpoint_count * len(dataset)
                overall_remaining = (overall_total - overall_completed) / rate
                total_elapsed = now - args.evaluation_started
                print(
                    f"ETA checkpoint={checkpoint_position}/{checkpoint_count} "
                    f"{checkpoint_label} frames={processed}/{len(dataset)} "
                    f"({processed / len(dataset):.1%}) rate={rate:.2f} frames/s "
                    f"checkpoint_remaining={format_duration(checkpoint_remaining)} "
                    f"overall_remaining={format_duration(overall_remaining)} "
                    f"elapsed={format_duration(total_elapsed)}",
                    flush=True,
                )
                last_progress_report = now
    drain(0)
    executor.shutdown(wait=True)

    metric_rows.sort(key=lambda row: (str(row["dataset"]), int(row["frame_index"])))
    prediction_rows.sort(key=lambda row: (str(row["dataset"]), int(row["frame_index"])))
    by_scene: dict[str, list[dict[str, object]]] = {}
    for row in metric_rows:
        by_scene.setdefault(str(row["scene"]), []).append(row)
    scene_rows = [summarize_rows(rows, scene) for scene, rows in sorted(by_scene.items())]
    summary = micro_summary(metric_rows, scene_rows, args.seed)
    summary.update(macro_summary(scene_rows, args.seed))
    summary.update(
        artifact_id=artifact,
        model_type=args.model_type,
        evaluation_tag=args.evaluation_tag,
        checkpoint=checkpoint_label,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        ema_rate=ema_rate,
        update=update,
        condition_total_frames=args.frames,
        prediction_frames=len(dataset),
        metric_common_history=args.common_history,
        sampler=args.sampler,
        sampling_steps=args.sampling_steps,
        seed=args.seed,
        pixel_threshold=args.pixel_threshold,
        fscore_threshold_m=args.fscore_threshold_m,
        max_range_m=args.max_range_m,
        distance_scale=1.0,
        predictions_saved=args.save_predictions,
        timestamps_available=False,
        primary_aggregation="frame_micro",
        uncertainty_unit="scene_cluster",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / f"{artifact}_per_frame_metrics.csv", metric_rows)
    write_csv(output_dir / f"{artifact}_per_scene_summary.csv", scene_rows)
    if prediction_rows:
        write_csv(prediction_manifest, prediction_rows)
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"{checkpoint_label}: frame-micro CD={summary['cd_frame_micro_mean_m']:.6f} m, "
        f"F-score={summary['fscore_frame_micro_mean']:.6f}; "
        f"scene-macro CD={summary['cd_scene_macro_mean_m']:.6f} m, "
        f"F-score={summary['fscore_scene_macro_mean']:.6f}"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--training-bundle", type=Path, default=Path(__file__).resolve().parents[1]
    )
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--model-type", choices=("edm", "cd"), default="edm")
    result.add_argument("--frames", type=int, choices=(1, 3, 5, 8, 10), required=True)
    result.add_argument("--checkpoint", type=Path, action="append", default=[])
    result.add_argument("--ema-rates", default="all", help="all or comma-separated 0.999,0.9999")
    result.add_argument("--updates", default="all", help="all or comma-separated integer updates")
    result.add_argument("--scenes", default="all", help="all or comma-separated scene names")
    result.add_argument("--output-root", type=Path)
    result.add_argument(
        "--evaluation-tag",
        choices=("sweep", "full", "eval", "frame_ablation"),
        default="eval",
    )
    result.add_argument("--max-frames-per-scene", type=int, default=0)
    result.add_argument("--common-history", type=int, default=9)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--metric-workers", type=int, default=2)
    result.add_argument("--progress-interval-seconds", type=float, default=60.0)
    result.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--save-predictions", action="store_true")
    result.add_argument("--sampler")
    result.add_argument("--sampling-steps", type=int)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--sigma-min", type=float, default=0.002)
    result.add_argument("--sigma-max", type=float, default=80.0)
    result.add_argument("--pixel-threshold", type=int, default=150)
    result.add_argument("--fscore-threshold-m", type=float, default=0.10)
    result.add_argument("--max-range-m", type=float, default=4.0)
    result.add_argument("--force", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    args.training_bundle = args.training_bundle.resolve()
    args.diffusion_root = args.training_bundle / "diffusion_X7"
    args.dataset_root = args.training_bundle / "X7_A1_dataset"
    args.run_dir = args.run_dir.resolve()
    args.sampler, args.sampling_steps = resolve_inference_protocol(
        args.model_type, args.sampler, args.sampling_steps
    )
    if args.output_root is None:
        args.output_root = (
            args.training_bundle / "evaluation_results" / f"{args.model_type}_{args.frames}f"
        )
    args.output_root = args.output_root.resolve()
    scenes = parse_csv_set(args.scenes, str)
    checkpoints = discover_checkpoints(args)
    include_boundary = args.save_predictions and args.max_frames_per_scene == 0
    dataset = PairedSceneDataset(
        args.dataset_root,
        args.frames,
        args.common_history,
        scenes,
        args.max_frames_per_scene,
        include_boundary,
    )
    metric_frames = sum(record.metric_eligible for record in dataset.records)
    print(
        f"frames={args.frames}, checkpoints={len(checkpoints)}, "
        f"prediction_frames={len(dataset)}, metric_frames={metric_frames}"
    )
    for checkpoint in checkpoints:
        print(checkpoint)
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if (
        args.batch_size <= 0
        or args.num_workers < 0
        or args.metric_workers <= 0
        or args.progress_interval_seconds <= 0
    ):
        raise ValueError("invalid worker or batch configuration")
    args.device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    model, diffusion = model_and_diffusion(args)
    args.evaluation_started = time.monotonic()
    summaries = []
    for position, checkpoint in enumerate(checkpoints, start=1):
        summaries.append(
            evaluate_one(
                args,
                dataset,
                loader,
                model,
                diffusion,
                checkpoint,
                position,
                len(checkpoints),
            )
        )
    summaries.sort(key=lambda row: (int(row["update"]), str(row["ema_rate"])))
    summary_name = (
        f"{args.model_type}_{args.frames}f_{args.evaluation_tag}_"
        f"seed{args.seed}_checkpoint_summary.csv"
    )
    write_csv(args.output_root / summary_name, summaries)


if __name__ == "__main__":
    main()
