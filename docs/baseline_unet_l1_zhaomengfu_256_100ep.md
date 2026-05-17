# U-Net L1 Baseline - Zhaomengfu Sanmenji

## Run Summary

- Task: Chinese single-character calligraphy generation
- Dataset slice: chinese-calligraphy-dataset, `楷-赵孟俯三门记`
- Training pairs: 7200
- Unique characters in source slice: 6388
- Model: U-Net generator
- Loss: L1 reconstruction loss
- Resolution: 256 x 256 grayscale
- Epochs: 100
- GPU/backend: Modal, CUDA
- Modal app id: `ap-TbElvO8VPskqNTOLvF8B4v`
- Modal volume dir: `outputs/modal_prepared_unet_256_1778941389`

## Final Metrics

- epoch: 100
- train_l1: 0.06861742266902217
- val_l1: 0.1305038692222701

## Files

- `checkpoints/best.pt`: best validation checkpoint
- `checkpoints/last.pt`: final epoch checkpoint
- `previews/epoch_100.png`: final training preview
- `eval/generalization_board.png`: fixed generalization board
- `eval/generalization_chars.txt`: characters used for generalization test
- `eval/quality_metrics.csv`: simple generated-image metrics
- `train_log.csv`: full epoch metrics
- `SHA256SUMS.txt`: artifact checksums

## Current Judgment

This baseline proves the data preprocessing, Modal training, checkpoint persistence,
and content-to-calligraphy generation path are working. The generated characters are
mostly recognizable and show clear brush-style transfer. It is not the final model:
complex characters still lose fine structure, and some outputs are too dark or blob-like.

The next fair experiment is Pix2Pix with a U-Net generator and PatchGAN discriminator,
using the same dataset, resolution, epoch budget, and evaluation character board.
