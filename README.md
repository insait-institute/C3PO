# Parameter Exploration for RLVR via Variational Learning

Companion code for the paper **"Parameter Exploration for RLVR via Variational Learning"**, which
introduces **Perturbed Parameter Policy Optimization (3PO)**: a family of parameter-space
exploration strategies for Reinforcement Learning with Verifiable Rewards (RLVR).

Instead of relying only on action-space heuristics (temperature, clipping, entropy bonuses), 3PO
samples model weights from an approximate posterior learned with a variational optimizer, [IVON](https://arxiv.org/abs/2402.17641). The amount of weight noise becomes an extra control lever for exploration. We study three variants:

| Variant | Idea |
|---------|------|
| **B3PO** | One weight perturbation per gradient step, synced to the rollout engine. |
| **M3PO** | `M` Monte-Carlo perturbations per step; rollouts and advantages computed per sample, gradients averaged. |
| **C3PO** | Each GRPO group of `G` rollouts is split across `N` independent perturbations (`G/N` rollouts each); advantages are computed over the full, more-diverse group, with a Seq-MIS importance-sampling correction. |

## Repository layout

| Path | Description |
|------|-------------|
| `verl/` | Fork of [verl](https://github.com/volcengine/verl) (`v0.8.0.dev`) with the IVON optimizer and the 3PO training loop integrated. |
| `ivon/` | Fork of [IVON](https://github.com/team-approx-bayes/ivon) (`ivon-opt 0.1.3`) modified to support FSDP training. |
| `verl/recipes/` | Launch scripts: `run_sft.sh` (warm-start SFT), `run_rl.sh` (RLVR). |
| `verl/scripts/data/` | Dataset preparation scripts (DapoMath, Nemotron SFT, eval benchmarks). 

The bulk of the 3PO logic lives in:
- `verl/verl/trainer/ppo/ray_trainer.py`: B3PO / M3PO / C3PO rollout-and-update loop.
- `verl/verl/workers/config/optimizer.py` and `verl/verl/trainer/config/optim/fsdp.yaml`: IVON optimizer wiring and `ivon_config` fields.
- `verl/verl/workers/fsdp_workers.py` and `verl/verl/workers/actor/dp_actor.py`: model update and noising logic.
- `ivon/ivon/_ivon.py`: the IVON optimizer itself.

## Installation

```bash
bash setup_env.sh
```

This installs `verl` (editable), the vLLM/SGLang/mcore stack, pinned dependencies
(`transformers 4.57.6`, `vllm 0.11.0`, `torch 2.8.0`, `ray 2.45.0`, `math-verify`), and the vendored
`ivon` package. Edit `CUDA_HOME` in the script to match your environment. Experiments were run on
8× NVIDIA H200 (144 GB) GPUs.

## Data preparation

Parquet files are written to `verl/data/` (the default `DATA_ROOT`).

```bash
cd verl
# RLVR training data (DapoMath-17k), per base model
python scripts/data/dapomath.py --model qwm        # or olmo3
# Evaluation benchmarks (AIME 24-26, MATH-500, AMC, Minerva)
python scripts/data/math_evals.py --output data/math_evals.parquet
# SFT warm-start data (Llama-Nemotron Post-Training subset)
python scripts/data/nemotron_ptds.py
```

## Training

The pipeline is two stages: a warm-start SFT followed by RLVR. Both launch scripts are configured
through environment variables (see the top of each script for the full list); `CUDA_VISIBLE_DEVICES`
determines the number of GPUs.

### 1. Warm-start SFT

```bash
MODEL_NAME=olmo3 OPTIMIZER=ivon DATA_NAME=nmt \
  bash verl/recipes/run_sft.sh
```

### 2. RLVR

`run_rl.sh` is the single entry point for all methods. The method is selected by the optimizer plus a
few IVON knobs:

| Paper method | Invocation |
|--------------|------------|
| **GRPO** (baseline) | `OPTIMIZER=adamw` |
| **B3PO** | `OPTIMIZER=ivon` with `ivon_config.use_ivon_rollout_only=true` (single perturbation/step) |
| **M3PO** | `OPTIMIZER=ivon M3PO_M=M` (e.g. `M=4`. To keep compute roughly same as GRPO, set `GROUP_SIZE=G/M`) |
| **C3PO** | `OPTIMIZER=ivon C3PO_N=N` (`rollout.c3po_n=N` chunks + Seq-MIS correction) |

Example - C3PO on Olmo3 with `N=4`:

```bash
MODEL_NAME=olmo3 MODEL_TYPE=base OPTIMIZER=ivon \
  C3PO_N=4 GROUP_SIZE=16 ESS=1e9 \
  bash verl/recipes/run_rl.sh
```

`METHOD` can be used to select action-space methods that 3PO is composable with:
`grpo_cliphigh`, `grpo_klcov`, `grpo_entloss`, `grpo_clipcov`.

> **Note on naming.** The `DECOUPLED_MC_SAMPLES` / `decoupled_mc_samples` knob corresponds to the
> *decoupled* variance-reduction variant discussed in Appendix C, not to any of the three main-paper
> methods.

### Key hyperparameters

`ESS` is the effective sample size **λ** from the paper — the central noise lever. Larger λ ⇒ less
noise; smaller λ ⇒ more exploration. The main results use:

`M3PO_M` controls the number of MC samples used in M3PO

`C3PO_N` controls the number of chunks used in C3PO

`DECOUPLED_MC_SAMPLES` corresponds to the *decoupled* variance-reduction approach from Appendix C

By default the IVON Hessian `h` is initialized from a constant `h0` rather than the
SFT optimizer state, so the method is drop-in for any off-the-shelf checkpoint (set
`IVON_INIT_METHOD=trained` to load a learned prior instead).

## Models & benchmarks

- **Base models:** `allenai/Olmo-3-1025-7B` and `Qwen/Qwen2.5-Math-7B`.
- **Warm-started models:** ``
- **Training data:** DapoMath-17k.
- **Benchmarks:** AIME 2024–2026, MATH-500, AMC 2023, Minerva.

## Citation
If you find this work useful, please consider citing:

```bibtex
@misc{venkatkrishna2026parameter,
      title={Parameter Exploration for RLVR via Variational Learning}, 
      author={Vatsal Venkatkrishna and Nico Daheim and Iryna Gurevych},
      year={2026},
}
```