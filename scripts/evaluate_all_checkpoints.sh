#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FONT="${FONT:-/usr/share/fonts/truetype/arphic/uming.ttc}"
if [[ ! -f "$FONT" && -f "/System/Library/Fonts/Supplemental/Songti.ttc" ]]; then
  FONT="/System/Library/Fonts/Supplemental/Songti.ttc"
fi

for EXP in unet_l1_128_full pix2pix_128_full unet_l1_256_full; do
  CKPT="outputs/$EXP/checkpoints/best.pt"
  if [[ -f "$CKPT" ]]; then
    python3 scripts/evaluate_quality.py \
      --checkpoint "$CKPT" \
      --content-font "$FONT" \
      --text-file configs/eval_chars_stage1.txt \
      --out-dir "outputs/$EXP/fixed_eval_best"
  fi
done

