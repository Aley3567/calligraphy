# Cloud Migration Notes

## Preferred Server

优先要 Linux + NVIDIA GPU。

建议配置：

- Ubuntu 20.04 或 22.04
- Python 3.9 或 3.10
- CUDA 11.8 或 12.1
- GPU 显存至少 12GB，最好 24GB
- 磁盘至少 100GB
- 支持 SSH 登录
- 支持 `scp` 或 `rsync`

## Cloud Directory

```text
/opt/calligraphy_generation_algo/
  configs/
  data/
    raw/
    processed/
  docs/
  outputs/
  repos/
  runtime_models/
  scripts/
```

## Upload Items

迁移时至少上传：

- `scripts/`
- `configs/`
- `docs/`
- `requirements.txt`
- 原始书法数据集或处理后的 `data/processed/`

不要只上传 checkpoint。云端需要能从数据审计、预处理、训练到评估完整复现。

## Basic Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/audit_dataset.py --raw-root data/raw/chinese-calligraphy-dataset --out-dir outputs/audit
```

## 3x3090 Stage 1 Commands

先准备全量 paired 数据：

```bash
chmod +x scripts/*.sh
FONT=/usr/share/fonts/truetype/arphic/uming.ttc ./scripts/launch_prepare_full.sh
```

再并行启动三组实验：

```bash
./scripts/launch_3x3090_experiments.sh
```

查看日志：

```bash
tail -f outputs/cloud_logs/unet_l1_128_full.log
tail -f outputs/cloud_logs/pix2pix_128_full.log
tail -f outputs/cloud_logs/unet_l1_256_full.log
```

训练中或训练后统一生成固定评估板和质量指标：

```bash
./scripts/evaluate_all_checkpoints.sh
```

第一阶段只看中文结构，不接英文、数字、标点。
