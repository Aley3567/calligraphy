# Calligraphy Generation Algorithm

当前目标已经收口：

```text
中文单字书法生成
弱配对监督
结构保真优先
先重建 preprocessing / evaluation / structure-aware loss
```

## Handoff

This repository is now prepared as a clean handoff package for continuing the
calligraphy generation work on a stronger GPU machine.

Start here:

- `docs/HANDOFF_TO_COLLABORATOR.md`
- `docs/NEXT_AI_PROMPT.md`

Large prepared datasets and model artifacts should be downloaded from GitHub
Releases instead of committed to git.

当前保留的基线模型版本是：

```text
U-Net L1 baseline best.pt
Release: baseline-unet-l1-zhaomengfu-256-100ep
```

Release:

```text
https://github.com/Aley3567/calligraphy/releases/tag/baseline-unet-l1-zhaomengfu-256-100ep
```

## Current Algorithm Position

这个任务不是普通 paired image-to-image translation，而是：

```text
weakly paired, single-style, structure-preserving Chinese calligraphy glyph generation
```

当前输入输出定义：

```text
x = 标准字体渲染出的 content glyph
y = 目标书法字 target glyph
y_hat = f_theta(x)
```

下一阶段的模型输入、监督和评价需要显式加入：

```text
mask
skeleton
distance transform
hole / white-space map
ink density
fixed high-risk evaluation board
```

## Data

使用数据集：

```text
zhuojg/chinese-calligraphy-dataset
https://github.com/zhuojg/chinese-calligraphy-dataset
```

本阶段使用的单风格：

```text
楷-赵孟俯三门记
约 7200 images
约 6388 unique characters
256x256 grayscale pairs
```

数据目录结构：

```text
书法家或风格目录 / 汉字目录 / 数字.gif
```

数字文件名不是标签，汉字目录名才是字符标签。

## Local First Run

审计数据：

```bash
python3 scripts/audit_dataset.py \
  --raw-root data/raw/chinese-calligraphy-dataset \
  --out-dir outputs/audit
```

准备单风格 glyph pairs：

```bash
python3 scripts/prepare_glyph_pairs.py \
  --raw-root data/raw/chinese-calligraphy-dataset \
  --writer-name WRITER_FOLDER_NAME \
  --content-font /System/Library/Fonts/Supplemental/Songti.ttc \
  --out-dir data/processed/single_writer_glyph_pairs \
  --image-size 256 \
  --max-items 1000
```

## Next Valid Work

下一步是实现结构化底座：

```text
1. preprocessing:
   target gray / clean mask / skeleton / distance transform / hole map / metadata

2. evaluation:
   seen / unseen / high-risk / structure groups
   ink ratio / local density / hole preservation / skeleton metrics

3. model:
   structure-aware U-Net
   mask head + ink head
   gray + mask + edge + density + hole + bbox loss
```

## Structure-Aware Preprocessing

从已有 `manifest.csv` paired dataset 生成结构辅助图和 metadata：

```bash
python3 scripts/build_structure_dataset.py \
  --data-dir data/processed/zhaomengfu_full_256 \
  --out-dir data/processed/zhaomengfu_structure_256 \
  --threshold 220 \
  --workers 8
```

输出包括：

```text
content_mask / content_skeleton / content_distance / content_hole
target_mask / target_skeleton / target_distance / target_edge / target_hole
manifest.csv
metadata.jsonl
structure_summary.json
```

小样本 smoke：

```bash
python3 scripts/build_structure_dataset.py \
  --data-dir data/processed/smoke_zhaomengfu_128 \
  --out-dir outputs/structure_smoke_128 \
  --max-items 8
```

## Fixed Structural Evaluation

使用固定结构分组评估 baseline：

```bash
python3 scripts/evaluate_quality.py \
  --checkpoint artifacts/baseline_unet_l1_zhaomengfu_256_100ep/checkpoints/best.pt \
  --content-font /System/Library/Fonts/Supplemental/Songti.ttc \
  --groups-file configs/eval_groups_stage1.json \
  --out-dir outputs/structure_eval_baseline_best \
  --image-size 256
```

主要输出：

```text
eval_board.png
quality_metrics.csv
quality_summary.json
generated/
```

## Important Docs

```text
docs/ALGORITHM_PLAN.md
docs/baseline_unet_l1_zhaomengfu_256_100ep.md
docs/CLOUD_MIGRATION.md
docs/COLLABORATION.md
```
