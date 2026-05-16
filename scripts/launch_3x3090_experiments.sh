#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FONT="${FONT:-/usr/share/fonts/truetype/arphic/uming.ttc}"
if [[ ! -f "$FONT" && -f "/System/Library/Fonts/Supplemental/Songti.ttc" ]]; then
  FONT="/System/Library/Fonts/Supplemental/Songti.ttc"
fi

mkdir -p outputs/cloud_logs

CUDA_VISIBLE_DEVICES=0 nohup python3 scripts/train_unet_baseline.py \
  --data-dir data/processed/zhaomengfu_full_128 \
  --out-dir outputs/unet_l1_128_full \
  --epochs 100 \
  --batch-size 32 \
  --image-size 128 \
  > outputs/cloud_logs/unet_l1_128_full.log 2>&1 &
echo $! > outputs/cloud_logs/unet_l1_128_full.pid

CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/train_pix2pix.py \
  --data-dir data/processed/zhaomengfu_full_128 \
  --out-dir outputs/pix2pix_128_full \
  --epochs 100 \
  --batch-size 24 \
  --image-size 128 \
  > outputs/cloud_logs/pix2pix_128_full.log 2>&1 &
echo $! > outputs/cloud_logs/pix2pix_128_full.pid

CUDA_VISIBLE_DEVICES=2 nohup python3 scripts/train_unet_baseline.py \
  --data-dir data/processed/zhaomengfu_full_256 \
  --out-dir outputs/unet_l1_256_full \
  --epochs 100 \
  --batch-size 16 \
  --image-size 256 \
  > outputs/cloud_logs/unet_l1_256_full.log 2>&1 &
echo $! > outputs/cloud_logs/unet_l1_256_full.pid

echo "launched 3 experiments"
cat outputs/cloud_logs/*.pid

