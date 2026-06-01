# Lesson 14 — MLOps Assignment 1: Distributed Training

Nebius ML course, Lesson 14.

Covers distributed model training on a Nebius Managed Kubernetes cluster using SkyPilot, Docker, and a custom training container.

## Contents

- `train.py` — PyTorch training script (distributed with `torchrun`)
- `train_job.yaml` — SkyPilot job definition for the Kubernetes cluster
- `Dockerfile` — container image for the training job
- `mk8s-ng-config.json` — Managed Kubernetes node group configuration
- `asgmnet_review.md` — assignment submission review and grading notes
- `fixes_to_share.md` — issues encountered and solutions
- `full_asgmnt.md` — full assignment description
- `training_log.txt` — training run output log
