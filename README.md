# Calligraphy Generation Algorithm Prep

当前唯一目标：先准备中文手写体/书法体仿真生成算法代码，后续迁移到云端训练。

本目录不改现有前后端项目。它只负责算法准备：

- 审计 `zhuojg/chinese-calligraphy-dataset` 这类按书法家组织的数据。
- 先选一个书法家训练单风格 baseline。
- 第一阶段用 U-Net / Pix2Pix 思路跑通 `标准字形 content -> 目标书法字形 target`。
- 输出固定测试字评估板，重点看结构清晰、缺笔多笔、笔画粘连、是否便于骨架提取。
- 后续云端再接 VQ-Font / MX-Font 做正式 few-shot 字体生成实验。

前端植入边界：

- 可放到前端：用户上传样本后的去噪、二值化、倾斜校正、字符区域定位、字符分割预览、特征提取。
- 不放普通前端：训练数据集解压、书法家统计、content-target pair 生成、train/val split、epoch、batch size、loss、checkpoint、CUDA 日志。
- 训练与云端任务放后台/实验端，包装为模型训练任务、风格库、模型版本和固定评估图。

## Directory

```text
calligraphy_generation_algo/
  configs/                 # 训练配置
  data/
    raw/                   # 原始数据集放这里，或软链接到这里
    processed/             # 处理后的单书法家配对数据
  docs/                    # 算法说明和迁移说明
  outputs/                 # 训练日志、checkpoint、评估板
  repos/                   # 后续可放 VQ-Font/MX-Font 源码
  runtime_models/          # 训练完成后导出的模型
  scripts/                 # 数据审计、预处理、训练、评估脚本
```

## First Run

1. 放入或软链接原始数据集：

```bash
ln -s /path/to/chinese-calligraphy-dataset /Users/admin/Desktop/calligraphy_generation_algo/data/raw/chinese-calligraphy-dataset
```

2. 审计数据：

```bash
python3 scripts/audit_dataset.py \
  --raw-root data/raw/chinese-calligraphy-dataset \
  --out-dir outputs/audit
```

3. 选一个书法家后准备 U-Net baseline 数据：

```bash
python3 scripts/prepare_pix2pix_pairs.py \
  --raw-root data/raw/chinese-calligraphy-dataset \
  --writer-name WRITER_FOLDER_NAME \
  --content-font /System/Library/Fonts/Supplemental/Songti.ttc \
  --out-dir data/processed/single_writer_pix2pix \
  --image-size 128 \
  --max-items 1000
```

4. 训练 baseline：

```bash
python3 scripts/train_unet_baseline.py \
  --data-dir data/processed/single_writer_pix2pix \
  --out-dir outputs/unet_baseline \
  --epochs 20 \
  --batch-size 16 \
  --image-size 128
```

5. 生成固定评估板：

```bash
python3 scripts/generate_eval_board.py \
  --checkpoint outputs/unet_baseline/checkpoints/best.pt \
  --content-font /System/Library/Fonts/Supplemental/Songti.ttc \
  --text 一中国永道德山水风月 \
  --out outputs/unet_baseline/eval_board.png
```

## Algorithm Links

zhuojg/chinese-calligraphy-dataset: https://github.com/zhuojg/chinese-calligraphy-dataset

Pix2Pix: https://github.com/phillipi/pix2pix

U-Net: https://arxiv.org/abs/1505.04597

VQ-Font: https://github.com/awei669/VQ-Font

MX-Font: https://github.com/clovaai/mxfont

fewshot-font-generation: https://github.com/clovaai/fewshot-font-generation

FsFont: https://github.com/tlc121/FsFont

DG-Font: https://github.com/ecnuycxie/DG-Font

## Cloud 3x3090 Quick Start

```bash
chmod +x scripts/*.sh
FONT=/usr/share/fonts/truetype/arphic/uming.ttc ./scripts/launch_prepare_full.sh
./scripts/launch_3x3090_experiments.sh
./scripts/evaluate_all_checkpoints.sh
```

三组并行实验：

- `unet_l1_128_full`: 快速结构 baseline。
- `pix2pix_128_full`: 对比 GAN 是否增强风格、是否破坏结构。
- `unet_l1_256_full`: 检查更高分辨率下笔画和骨架是否更清楚。
