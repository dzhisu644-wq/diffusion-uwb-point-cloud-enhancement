import argparse
import csv
import os
import random
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset

from cm import dist_util, logger
from cm.resample import create_named_schedule_sampler
from cm.script_util_cond import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from cm.train_util_cond import TrainLoop

from cm.dataset_radar2lidar import Radar2LidarDataset

def main():
    args = create_argparser().parse_args()
    args.in_ch = args.num_past_frames + 2

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.benchmark = args.cudnn_benchmark

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dist_util.setup_dist()
    logger.configure(dir=args.out_dir)
    logger.log(f"random seed: {args.seed}")

    logger.log(
        f"using {args.num_past_frames + 1} radar frames as condition, model in_ch={args.in_ch}"
    )
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion
    )

    logger.log("creating dataset...")
    transform = None
    if not args.fast_io:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),   # -> [0,1], shape [1,H,W]
        ])
    dataset_ranges = []
    if args.split_manifest:
        with open(args.split_manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == args.split_name:
                    dataset_ranges.append(
                        (
                            row["dataset"],
                            int(row["start_index"]),
                            int(row["end_index_exclusive"]),
                        )
                    )
        if not dataset_ranges:
            raise ValueError(
                f"no {args.split_name!r} ranges found in {args.split_manifest}"
            )
    else:
        dataset_ranges = [
            (path.name, 0, None)
            for path in sorted(Path(args.dataset_dir).glob("dataset_aligned_*"))
            if path.is_dir()
        ]

    all_datasets = []
    for name, start_index, end_index in dataset_ranges:
        sub_dir = os.path.join(args.dataset_dir, name)
        if os.path.exists(sub_dir):
            logger.log(
                f"Loading sub-dataset: {sub_dir} range=[{start_index}, {end_index})"
            )
            all_datasets.append(
                Radar2LidarDataset(
                    root_dir=sub_dir,
                    transform=transform,
                    num_past_frames=args.num_past_frames,
                    start_index=start_index,
                    end_index=end_index,
                    cache_images=args.cache_images,
                )
            )
        else:
            logger.log(f"Warning: {sub_dir} does not exist, skipping.")

    if not all_datasets:
        raise ValueError("no training datasets were loaded")
    dataset = ConcatDataset(all_datasets)
    logger.log(f"Total images in combined dataset: {len(dataset)}")

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(**loader_kwargs)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        max_train_steps=args.max_train_steps,
    ).run_loop()


def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        # ===== training =====
        schedule_sampler="uniform",
        lr=5e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        max_train_steps=0,
        batch_size=16,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=200,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        seed=0,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        cache_images=False,
        fast_io=False,
        allow_tf32=True,
        cudnn_benchmark=True,

        # ===== model =====
        image_size=128,
        in_ch=2,     # 只对 LiDAR 做 diffusion
        out_ch=1,
        num_past_frames=0,
        loss_norm="l2+lpips",

        # ===== path =====
        dataset_dir="./X7_A1_dataset",
        out_dir="./runs/edm",
        split_manifest="",
        split_name="train",
    ))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
