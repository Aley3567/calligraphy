# Handoff to Collaborator

## Project Requirement

Build a Chinese single-character calligraphy generation system.

Input:

- a standard-font Chinese glyph image for the target character;
- several reference glyphs from the target writer/style.

Output:

- the same character rendered in the target calligraphy style.

Hard quality requirements:

- the output must remain readable as the requested character;
- radicals, stroke topology, and inner white space must be preserved;
- complex characters must not collapse into dense ink blobs;
- validation must include zoomed visual review, not only scalar metrics.

Current single-style dataset focus:

- writer/style: `楷-赵孟俯三门记`;
- image size: `256 x 256`;
- prepared data includes content glyphs, target glyphs, masks, skeleton maps, distance maps, holes, and edge maps.

## What This Repository Provides

- data preparation scripts for paired glyph data;
- structure map generation and validation utilities;
- structure-aware U-Net baseline code;
- Modal training entrypoints;
- documentation describing the target task and algorithm constraints;
- a GitHub Release dataset package for the prepared Zhao Mengfu split.

This repository does not include a final production checkpoint. The available local/cloud budget was not enough to establish a definitive production-quality model. Treat this repo as a clean code/data handoff and research starting point.

## Recommended Next Direction

Use a diffusion-font or two-stage structure-rendering route instead of direct residual ink add/remove.

Recommended order:

1. Reproduce a strong font-generation baseline, preferably FontDiffuser or a similar diffusion-font architecture.
2. Train on the provided Zhao Mengfu data split.
3. Add explicit structure conditioning if the baseline is not structurally stable:
   - `target_skeleton`;
   - `target_edge`;
   - distance/SDF-like map;
   - hole/white-space constraints.
4. Separate structure generation from ink rendering when possible:
   - stage A: `content + style_refs -> target structure`;
   - stage B: `predicted structure + style_refs -> final ink`.
5. Add a PatchGAN/discriminator only after the generated structure is readable. Do not use GAN loss to fix wrong glyph geometry.

## Directions Not Recommended

Do not spend the next compute budget on:

- direct `content_mask * (1 - remove) + add` residual-mask prediction as the main architecture;
- tuning outside-target, edge, or gate losses around blob-like outputs;
- judging progress only by Dice or L1 metrics;
- post-processing dense ink blobs into glyphs;
- training without comparing output-to-target against output-to-content shortcut baselines.

## Suggested 3090 x 3 Training Plan

Hardware assumption: three RTX 3090 GPUs.

First pass:

- use the full provided train/val split;
- run a known diffusion-font baseline;
- keep previews at fixed validation characters every epoch or every N steps;
- review both full-size and zoomed previews.

Minimum validation table:

| Check | Required Evidence |
|---|---|
| content correctness | generated character is still the requested character |
| structure | skeleton/edge/white-space are not collapsed |
| style | output resembles the target writer rather than the standard content font |
| shortcut | output is closer to target than to content |
| visual | zoomed preview is readable, not only acceptable from far away |

Stop rule:

- If the structure map is unreadable, stop ink/rendering work and fix structure first.
- If the output is content-like, stop and inspect conditioning/style injection.
- If the output is visually blob-like while metrics improve, do not continue by epochs alone.

## Dataset Release

Expected release asset:

```text
zhaomengfu_structure_256_scale052_full_no_valchar_handoff.tar.gz
```

Expected extracted layout:

```text
data/processed/zhaomengfu_structure_256_scale052_full_no_valchar/
data/processed/zhaomengfu_structure_256_scale052_full_no_valchar_splits/
```

Use the split manifests from:

```text
data/processed/zhaomengfu_structure_256_scale052_full_no_valchar_splits/train_manifest.csv
data/processed/zhaomengfu_structure_256_scale052_full_no_valchar_splits/val_manifest.csv
```

## Minimal Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After extracting the dataset release asset into the repository root, verify:

```bash
python3 -m py_compile scripts/*.py
ls data/processed/zhaomengfu_structure_256_scale052_full_no_valchar
ls data/processed/zhaomengfu_structure_256_scale052_full_no_valchar_splits
```

## Collaboration Notes

- Keep training logs and checkpoints out of git.
- Store large datasets and model artifacts as GitHub Release assets, cloud storage objects, or experiment-tracking artifacts.
- Add a short run card for every serious run: objective, architecture, dataset, budget, checkpoint, validation preview, metrics, and decision.
