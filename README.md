# Calligraphy Generation Algorithm

当前目标已经收口：

```text
中文单字书法生成
弱配对监督
结构保真优先
先不继续训练
先重建 preprocessing / evaluation / structure-aware loss
```

本仓库不再把 Pix2Pix、继续加 epoch、FontDiffuser retry 当作当前主线。当前唯一保留的可用模型版本是：

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

但下一阶段不能只用裸 U-Net + L1。需要显式加入：

```text
mask
skeleton
distance transform
hole / white-space map
ink density
fixed high-risk evaluation board
```

## What Is Kept

保留：

```text
scripts/audit_dataset.py
scripts/prepare_glyph_pairs.py
scripts/train_unet_baseline.py
scripts/evaluate_quality.py
scripts/generate_eval_board.py
scripts/modal_train_unet.py
docs/ALGORITHM_AUDIT_AND_NEXT_STAGE.md
docs/GPT_PRO_ALGORITHM_RESEARCH_PROMPT.md
docs/baseline_unet_l1_zhaomengfu_256_100ep.md
```

## What Is Not Current Mainline

不再作为当前训练入口：

```text
plain Pix2Pix
U-Net resume / continue training
FontDiffuser retry
diffusion large training
refiner stacking
3x3090 parallel experiment sweep
```

原因见：

```text
docs/ALGORITHM_AUDIT_AND_NEXT_STAGE.md
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

训练 U-Net baseline 只用于复现，不作为继续调参入口：

```bash
python3 scripts/train_unet_baseline.py \
  --data-dir data/processed/single_writer_glyph_pairs \
  --out-dir outputs/unet_baseline \
  --epochs 20 \
  --batch-size 16 \
  --image-size 256
```

生成固定评估板：

```bash
python3 scripts/generate_eval_board.py \
  --checkpoint outputs/unet_baseline/checkpoints/best.pt \
  --content-font /System/Library/Fonts/Supplemental/Songti.ttc \
  --text 一二三人日田回国民夜耀翔龟鬱齋 \
  --out outputs/unet_baseline/eval_board.png \
  --image-size 256
```

## Next Valid Work

下一步不是继续训练，而是实现结构化底座：

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

没有上述三件事，不启动新的完整训练。

## Important Docs

```text
docs/ALGORITHM_AUDIT_AND_NEXT_STAGE.md
docs/GPT_PRO_ALGORITHM_RESEARCH_PROMPT.md
docs/baseline_unet_l1_zhaomengfu_256_100ep.md
docs/CLOUD_MIGRATION.md
docs/COLLABORATION.md
```
