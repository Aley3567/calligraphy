# Collaboration Workflow

## Current Shared Goal

The repository should now align around one target:

```text
structure-preserving Chinese calligraphy glyph generation
```

Current promoted model artifact:

```text
U-Net L1 baseline best.pt
```

Current next work:

```text
preprocessing + evaluation + structure-aware loss infrastructure
```

## Branch Rules

```text
main: stable code and docs
feature/preprocessing: mask/skeleton/distance/hole map pipeline
feature/evaluation: fixed boards and structural metrics
feature/structure-aware-unet: mask-head/ink-head and loss work
```

Do not push large artifacts to git.

## GitHub Issues

Recommended issues now:

```text
[Preprocessing] target mask/skeleton/distance/hole maps
[Evaluation] fixed high-risk character board and metrics
[Model] structure-aware U-Net loss design
[Docs] structure preprocessing and evaluation usage
```

## What Goes Into Git

Commit:

```text
README.md
requirements.txt
configs/
docs/
scripts/
small documentation images
```

Do not commit:

```text
raw datasets
processed datasets
checkpoints
cloud outputs
runtime models
zip/tar archives
```

Large model files go to:

```text
GitHub Release
cloud disk
object storage
shared drive
```

## Review Gate

Before any new training PR is accepted, it must show:

```text
fixed evaluation characters
expected failure mode addressed
loss definition
smoke validation result
stop rule
comparison target: U-Net baseline best.pt
```
