#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RAW_ROOT="${RAW_ROOT:-data/raw/extracted_calligrapher_dataset/chinese-calligraphy-dataset-with-calligrapher}"
WRITER="${WRITER:-楷-赵孟俯三门记}"
FONT="${FONT:-/usr/share/fonts/truetype/arphic/uming.ttc}"

if [[ ! -f "$FONT" && -f "/System/Library/Fonts/Supplemental/Songti.ttc" ]]; then
  FONT="/System/Library/Fonts/Supplemental/Songti.ttc"
fi

python3 scripts/audit_dataset.py \
  --raw-root "$RAW_ROOT" \
  --out-dir outputs/audit_real_dataset

python3 scripts/prepare_pix2pix_pairs.py \
  --raw-root "$RAW_ROOT" \
  --writer-name "$WRITER" \
  --content-font "$FONT" \
  --out-dir data/processed/zhaomengfu_full_128 \
  --image-size 128 \
  --max-items 0

python3 scripts/prepare_pix2pix_pairs.py \
  --raw-root "$RAW_ROOT" \
  --writer-name "$WRITER" \
  --content-font "$FONT" \
  --out-dir data/processed/zhaomengfu_full_256 \
  --image-size 256 \
  --max-items 0

