#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BUNDLE_ROOT
CONFIG_FILE="${1:-$BUNDLE_ROOT/config/paper_8f.env}"
source "$CONFIG_FILE"

if [[ -z "${RUN_TAG:-}" ]]; then
  RUN_TAG="$((NUM_PAST_FRAMES + 1))f"
fi
OUT_DIR="$RUNS_DIR/edm_${RUN_TAG}_full_v4"
if [[ ! "$EDM_STEPS" =~ ^[0-9]+$ ]]; then
  echo "EDM_STEPS must be a non-negative integer, got: $EDM_STEPS" >&2
  exit 2
fi
printf -v FINAL_STEP "%06d" "$((10#$EDM_STEPS))"
FINAL_CHECKPOINT="$OUT_DIR/ema_0.9999_${FINAL_STEP}.pt"
mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONPATH="$BUNDLE_ROOT/diffusion_X7:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_HOME="$BUNDLE_ROOT/weights/torch"

resume_args=()
shopt -s nullglob
checkpoints=("$OUT_DIR"/model*.pt)
if ((${#checkpoints[@]})); then
  mapfile -t checkpoints < <(printf '%s\n' "${checkpoints[@]}" | sort)
  resume_args=(--resume_checkpoint "${checkpoints[-1]}")
  echo "Resuming EDM from ${checkpoints[-1]}"
fi

command=(
  "$PYTHON_BIN" -u "$BUNDLE_ROOT/diffusion_X7/scripts/train_radar2lidar_edm.py"
  --attention_resolutions 32,16,8
  --class_cond False
  --use_scale_shift_norm True
  --dropout 0.1
  --ema_rate 0.999,0.9999
  --batch_size "$GLOBAL_BATCH"
  --microbatch "$EDM_MICROBATCH"
  --image_size 128
  --lr "$EDM_LR"
  --num_channels 64
  --num_head_channels 64
  --num_res_blocks 3
  --resblock_updown True
  --schedule_sampler lognormal
  --use_fp16 "$USE_FP16"
  --weight_decay 0.0
  --weight_schedule karras
  --loss_norm l2+lpips
  --out_ch 1
  --num_past_frames "$NUM_PAST_FRAMES"
  --max_train_steps "$EDM_STEPS"
  --save_interval "$SAVE_INTERVAL"
  --seed "$SEED"
  --num_workers "$NUM_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
  --persistent_workers "$PERSISTENT_WORKERS"
  --pin_memory "$PIN_MEMORY"
  --cache_images "$CACHE_IMAGES"
  --fast_io "$FAST_IO"
  --allow_tf32 "$ALLOW_TF32"
  --cudnn_benchmark "$CUDNN_BENCHMARK"
  --dataset_dir "$DATASET_DIR"
  --split_manifest "$SPLIT_MANIFEST"
  --split_name train
  --out_dir "$OUT_DIR"
  "${resume_args[@]}"
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY RUN:"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

source_files=(
  "$BUNDLE_ROOT"/diffusion_X7/cm/*.py
  "$BUNDLE_ROOT/diffusion_X7/scripts/train_radar2lidar_edm.py"
)
signature_command=(
  "$PYTHON_BIN" "$BUNDLE_ROOT/scripts/run_signature.py"
  --output "$OUT_DIR/run_signature.json"
)
if ((${#resume_args[@]})); then
  signature_command+=(--require-existing)
fi
for source_file in "${source_files[@]}"; do
  signature_command+=(--source "$source_file")
done
signature_command+=(
  --artifact "split_csv=$SPLIT_MANIFEST"
  --artifact "split_json=${SPLIT_MANIFEST%.csv}.json"
  -- "${command[@]}"
)
"${signature_command[@]}"

if [[ -f "$FINAL_CHECKPOINT" ]]; then
  echo "EDM already complete and signature-compatible: $FINAL_CHECKPOINT"
  exit 0
fi

cp "$CONFIG_FILE" "$OUT_DIR/launch_config.env"
cp "$SPLIT_MANIFEST" "$OUT_DIR/split.csv"
cp "${SPLIT_MANIFEST%.csv}.json" "$OUT_DIR/split.json"
sha256sum "${source_files[@]}" > "$OUT_DIR/source_sha256.txt"
printf '%q ' "${command[@]}" > "$OUT_DIR/launch_command.txt"
printf '\n' >> "$OUT_DIR/launch_command.txt"
"${command[@]}"
