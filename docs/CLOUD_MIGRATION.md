# Cloud Migration Notes

## Current Rule

Do not launch new full training runs until the structure-aware preprocessing and
evaluation protocol are implemented.

Cloud should currently be used for:

```text
dataset audit
preprocessing verification
fixed evaluation generation
small smoke runs only
artifact storage
```

Not for:

```text
plain Pix2Pix
U-Net resume
blind epoch extension
3x3090 experiment sweeps
FontDiffuser retry
```

## Preferred Server

```text
Ubuntu 20.04 or 22.04
Python 3.9 or 3.10
CUDA 11.8 or 12.1
GPU >= 12GB VRAM, preferably 24GB
disk >= 100GB
SSH / rsync / scp available
```

## Cloud Directory

```text
/opt/calligraphy_generation_algo/
  configs/
  data/
    raw/
    processed/
  docs/
  outputs/
  runtime_models/
  scripts/
```

## Upload Items

Upload:

```text
README.md
requirements.txt
configs/
docs/
scripts/
```

Do not put these in git:

```text
raw dataset zip
extracted dataset
processed training pairs
checkpoints
runtime models
cloud outputs
```

Use GitHub Release or object storage for model weights.

## Basic Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 scripts/audit_dataset.py \
  --raw-root data/raw/chinese-calligraphy-dataset \
  --out-dir outputs/audit
```

## Current Smoke-Only Preparation

Prepare a small paired glyph set:

```bash
FONT=/usr/share/fonts/truetype/arphic/uming.ttc

python3 scripts/prepare_glyph_pairs.py \
  --raw-root data/raw/chinese-calligraphy-dataset \
  --writer-name "楷-赵孟俯三门记" \
  --content-font "$FONT" \
  --out-dir data/processed/zhaomengfu_smoke_256 \
  --image-size 256 \
  --max-items 1000
```

Generate evaluation board only after fixed evaluation characters are defined.

Full training requires approval from `docs/ALGORITHM_AUDIT_AND_NEXT_STAGE.md`.
