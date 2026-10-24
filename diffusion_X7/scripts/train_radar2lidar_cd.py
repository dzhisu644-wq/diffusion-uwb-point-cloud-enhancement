import argparse
import csv
import os
import copy
import random
from pathlib import Path
import numpy as np
import torch as th
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from cm import dist_util, logger
from cm.resample import create_named_schedule_sampler
from cm.script_util_cond import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util_cond import CMTrainLoop
from cm.dataset_radar2lidar import Radar2LidarDataset

def main():
    args = create_argparser().parse_args()
    args.in_ch = args.num_past_frames + 2

    if th.cuda.is_available():
        th.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        th.backends.cudnn.allow_tf32 = args.allow_tf32
        th.backends.cudnn.benchmark = args.cudnn_benchmark

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)

    dist_util.setup_dist()
    logger.configure(dir=args.out_dir)
    logger.log(f"random seed: {args.seed}")
    logger.log(
        f"using {args.num_past_frames + 1} radar frames as condition, model in_ch={args.in_ch}"
    )

    # 1. 创建一致性训练的 EMA 和 Scales 调度函数
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )

    # CD 模式下 distillation 必须为 True
    distillation = "consistency" in args.training_mode

    logger.log("creating model and diffusion...")
    model_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    model_kwargs["distillation"] = distillation
    
    model, diffusion = create_model_and_diffusion(**model_kwargs)
    model.to(dist_util.dev())
    model.train()

    # 2. 加载教师模型 (Teacher Model)
    if len(args.teacher_model_path) > 0:
        logger.log(f"loading the teacher model from {args.teacher_model_path}")
        teacher_kwargs = copy.deepcopy(model_kwargs)
        teacher_kwargs["distillation"] = False # 教师是普通的 EDM
        teacher_kwargs["dropout"] = args.teacher_dropout
        # Teacher diffusion is only used for denoising; it does not evaluate a loss.
        teacher_kwargs["loss_norm"] = "l2"
        
        teacher_model, teacher_diffusion = create_model_and_diffusion(**teacher_kwargs)
        teacher_model.load_state_dict(
            dist_util.load_state_dict(args.teacher_model_path, map_location="cpu")
        )
        teacher_model.to(dist_util.dev())
        if args.use_fp16:
            teacher_model.convert_to_fp16()
        teacher_model.eval()

        # 初始时将教师的权重拷贝给学生
        for dst, src in zip(model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)
    else:
        raise ValueError("CD training requires a pre-trained teacher_model_path")

    # 3. 创建目标模型 (Target Model)
    target_kwargs = copy.deepcopy(model_kwargs)
    # The EMA target contributes predictions only. LPIPS belongs to the student loss.
    target_kwargs["loss_norm"] = "l2"
    target_model, _ = create_model_and_diffusion(**target_kwargs)
    target_model.to(dist_util.dev())
    target_model.train()
    dist_util.sync_params(target_model.parameters())
    # 同步权重
    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    # 4. 准备数据集 (使用你之前的 ConcatDataset 逻辑)
    logger.log("creating dataset...")
    transform = None
    if not args.fast_io:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
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

    if not all_datasets:
        raise ValueError("no training datasets were loaded")
    dataset = ConcatDataset(all_datasets)
    logger.log(f"Total images: {len(dataset)}")

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(**loader_kwargs)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # 5. 启动 CD 训练循环
    logger.log("starting consistency distillation...")
    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
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
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
    ).run_loop()

def create_argparser():
    defaults = model_and_diffusion_defaults()
    defaults.update(dict(
        training_mode="consistency_distillation",
        teacher_model_path="",
        teacher_dropout=0.1,
        total_training_steps=600000,
        batch_size=16,
        microbatch=-1,
        lr=5e-5,
        dataset_dir="./X7_A1_dataset",
        out_dir="./runs/cd",
        num_past_frames=0,
        save_interval=10000,
        log_interval=100,      
        seed=0,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        cache_images=False,
        fast_io=False,
        allow_tf32=True,
        cudnn_benchmark=True,
        resume_checkpoint="",
        split_manifest="",
        split_name="train",
        ema_rate="0.999,0.9999", 
        schedule_sampler="uniform",
    ))
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    temp_args, _ = parser.parse_known_args()
    if not hasattr(temp_args, 'weight_decay'):
        parser.add_argument("--weight_decay", type=float, default=0.0)
    if not hasattr(temp_args, 'lr_anneal_steps'):
        parser.add_argument("--lr_anneal_steps", type=int, default=0)
    if not hasattr(temp_args, 'fp16_scale_growth'):
        parser.add_argument("--fp16_scale_growth", type=float, default=1e-3)
    
    return parser

if __name__ == "__main__":
    main()
