# Prompt for the Next Local AI

You are taking over a Chinese calligraphy generation repository.

## Goal

Build a model that generates a target-style Chinese calligraphy glyph from:

1. a standard-font content glyph for the requested Chinese character;
2. several reference glyphs from the target calligrapher/style.

The output must remain the same character and must be readable at zoomed resolution. Do not optimize for "looks like calligraphy" if the character structure collapses.

## Current Data

Use the prepared Zhao Mengfu dataset release asset:

```text
zhaomengfu_structure_256_scale052_full_no_valchar_handoff.tar.gz
```

After extraction, expected directories are:

```text
data/processed/zhaomengfu_structure_256_scale052_full_no_valchar/
data/processed/zhaomengfu_structure_256_scale052_full_no_valchar_splits/
```

The data includes:

- content image;
- content mask/skeleton/distance/hole;
- target image;
- target mask/skeleton/distance/hole/edge;
- train/val manifests.

## Important Constraint

Do not continue by blindly training more epochs on a direct residual-mask model. The useful next direction is to model glyph structure explicitly.

Recommended mainline:

```text
content glyph + style references
  -> target structure condition
  -> ink/rendering model
```

Good candidates:

- FontDiffuser or another diffusion-based font-generation baseline;
- diffusion model with skeleton/edge/distance conditioning;
- two-stage model: structure predictor first, renderer second.

## Suggested Work Plan

1. Inspect the dataset and manifests.
2. Establish a baseline using a diffusion-font architecture.
3. Generate a fixed validation board after every serious run.
4. Judge zoomed visual quality before trusting metrics.
5. If output is structurally unstable, train/evaluate the structure stage separately:
   - target skeleton;
   - target edge;
   - distance/SDF-like map;
   - hole/white-space preservation.
6. Only add GAN/PatchGAN after the structure stage is readable.

## Validation Rules

Every serious run should report:

| Metric / Evidence | Why |
|---|---|
| generated vs target similarity | primary target quality |
| generated vs content similarity | shortcut check |
| skeleton/edge/white-space quality | structure preservation |
| fixed validation preview | human-readable review |
| zoomed preview | catches ink blobs and collapsed strokes |

Do not mark a model as useful if it only looks acceptable from far away.

## Avoid

- judging only by Dice/L1;
- treating a content-copy shortcut as progress;
- using GAN loss to fix wrong glyph geometry;
- post-processing dense blobs as a replacement for structure modeling;
- committing checkpoints, outputs, logs, or generated experiment folders to git.

## Expected Deliverable

A useful next result should include:

- code/config for the chosen architecture;
- fixed validation board;
- compact metrics;
- a short run card explaining the objective, data, budget, and decision.
