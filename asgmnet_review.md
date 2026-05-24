# Lecturer Review — Nebius DDP MLOps HW1

**Student:** Ilay Nisim & Boris Kogan
**Submission date:** 2026-05-24
**Reviewed against:** `full_asgmnt.md` grading criteria

---

## Score breakdown

| Criterion | Max | Score | Notes |
|---|---:|---:|---|
| mk8s cluster running with correct setup of GPU nodes | 15 | **15** | 2-node `gpu-h100-sxm` group, preset `1gpu-16vcpu-200gb`, SA attached. Config exported cleanly. |
| Dockerfile builds successfully | 15 | **15** | Matches Task 1 spec verbatim. Image confirmed pulled by both nodes during the job. |
| `train.py` runs correctly | 20 | **20** | DDP + NCCL backend, DistributedSampler, env-var configurable. Completed 500 steps with **real, decreasing loss (2.41 → 0.66)** — model is actually learning. Good choice of `bfloat16` over `float16` to avoid numerical instability on H100. |
| `train_job.yaml` correct (torchrun + SkyPilot vars) | 20 | **20** | All required fields present: `accelerators: H100:1`, `memory: 60+`, `num_nodes: 2`, `image_id`, `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET`. `torchrun` uses all four `SKYPILOT_*` env vars correctly. |
| Training log contains NCCL init section | 20 | **20** | 103 `NCCL INFO` lines. NCCL version (2.28.9+cuda13.0), RDMA plugin load, NET/Plugin lines, and successful `ncclCommInitRankConfig` for both ranks all logged. The explicit grading example pattern (`NCCL INFO NET/Plugin`) is present. |
| **Total** | **90** | **90/90** | |

> Note: the assignment header states "Total: 100" but the criteria list sums to 90. Scored against the listed criteria.

---

## What was done well

1. **End-to-end pipeline is sound.** Build VM → registry → mk8s + SA → SkyPilot → DDP job all wired correctly with proper auth handling.
2. **Multi-node DDP genuinely works.** Both ranks emit NCCL traces, world size = 2, and the loss converges — not just a passing init.
3. **Sensible model precision choice.** `bfloat16` on H100 is the right call for stable LLM training; many submissions ship `float16` and silently produce NaN losses.
4. **Reproducibility.** The script is deterministic given the env vars; matches `train_job.yaml` env configuration cleanly.

## Minor observations (no points deducted)

- `training_log.txt` contains ANSI color escape codes (`\033[36m`, etc.) from the `sky logs` terminal stream. Functionally fine, but a `sed` to strip them would make the log more readable.
- No gradient clipping in `train.py`. Not required by the assignment; bf16 made it unnecessary here, but as a defensive practice for LLM training it's worth adding (`clip_grad_norm_(model.parameters(), 1.0)`).
- Initial loss spike (2.41 at step 10) then a fast drop to ~0.43 by step 30 is unusual — suggests either very rapid memorisation on the small wikitext-2 corpus or a learning-rate-warmup-related artefact. Doesn't affect grading.

## Deliverables checklist

- [x] `mk8s-ng-config.json` (1.2 KB)
- [x] `Dockerfile` (302 B)
- [x] `train.py` (3.3 KB)
- [x] `train_job.yaml` (1.3 KB)
- [x] `training_log.txt` (33 KB, 266 lines, 103 NCCL lines)

---

## Final remark

Clean, well-instrumented submission. The combination of bf16 + working multi-node NCCL collective produces a log that is straightforward to grade. **Full marks.**
