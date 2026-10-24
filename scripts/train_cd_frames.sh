#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMES="${1:-}"
CONFIG_FILE="${2:-$BUNDLE_ROOT/config/paper_8f.env}"

case "$FRAMES" in
  1|3|5|8|10) ;;
  *)
    echo "Usage: bash scripts/train_cd_frames.sh {1|3|5|8|10} [config.env]" >&2
    exit 2
    ;;
esac

export RUN_TAG="${FRAMES}f"
export NUM_PAST_FRAMES="$((FRAMES - 1))"
export EDM_TEACHER_EMA="${EDM_TEACHER_EMA:-0.9999}"
exec bash "$BUNDLE_ROOT/scripts/train_cd.sh" "$CONFIG_FILE"
