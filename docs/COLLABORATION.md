# Collaboration Workflow

## Roles

- Friend/main owner: owns the GitHub repository, cloud server, main branch, and final merge decisions.
- yufeng: watches the full process, reviews pull requests, opens issues, and submits PRs for fixes or improvements.

## Branch Rules

- `main`: stable code only.
- `dev`: integration branch if needed.
- `feature/*`: implementation branches.
- `fix/*`: bug-fix branches.
- `reports/*`: optional branch for lightweight reports only.

Do not push directly to `main`.

## GitHub Monitoring

Use GitHub Issues for cloud training monitoring.

Recommended issues:

- `[Training Monitor] unet_l1_128_full`
- `[Training Monitor] pix2pix_128_full`
- `[Training Monitor] unet_l1_256_full`

Cloud server should comment progress every 5 or 10 epochs:

```text
epoch: 25/100
train loss:
val loss:
latest eval board:
risk:
next:
```

## What Goes Into Git

Commit:

- `README.md`
- `requirements.txt`
- `configs/`
- `docs/`
- `scripts/`
- `.gitignore`

Do not commit:

- raw dataset zip
- extracted dataset
- processed training pairs
- checkpoints
- cloud outputs
- runtime models

Large files stay on cloud disk, object storage, or a separate shared drive.

## Cloud Handoff

First cloud commands:

```bash
pip install -r requirements.txt
chmod +x scripts/*.sh
FONT=/usr/share/fonts/truetype/arphic/uming.ttc ./scripts/launch_prepare_full.sh
./scripts/launch_3x3090_experiments.sh
```

Generate fixed evaluation reports:

```bash
./scripts/evaluate_all_checkpoints.sh
```

