# BayesRL

Research artifacts for **variational / Bayesian approaches to reinforcement learning**, centered on
**parameter-space exploration for RLVR**.

Our current release accompanies the paper **"Parameter Exploration for RLVR via Variational Learning"**,
which introduces **Perturbed Parameter Policy Optimization (3PO)**: a family of exploration strategies for
Reinforcement Learning with Verifiable Rewards (RLVR). Rather than relying only on action-space heuristics
(temperature, clipping, entropy bonuses), 3PO samples model weights from an approximate posterior learned
with the variational optimizer [IVON](https://arxiv.org/abs/2402.17641), turning the amount of weight
noise into an explicit control lever for exploration.

📦 **Code:** [insait-institute/c3po](https://github.com/insait-institute/c3po)

## The 3PO family

| Variant | Idea |
|---------|------|
| **B3PO** | One weight perturbation from the IVON posterior per gradient step, synced to the rollout engine. |
| **M3PO** | `M` Monte-Carlo perturbations per step; rollouts and advantages computed per sample, gradients averaged. |
| **C3PO** | Each GRPO group of `G` rollouts is split across `N` independent perturbations (`G/N` each); advantages are computed over the full, more-diverse group with a Seq-MIS importance-sampling correction. |

## Collections

- **[3PO Models](https://huggingface.co/collections/BayesRL/3po-models)** — Olmo-3 and Qwen2.5-Math
  7B/8B checkpoints trained on DAPO-Math-17k with B3PO, M3PO, and C3PO (plus the `M3PO+` and decoupled-MC
  ablations).
- **[Warm-started Checkpoints](https://huggingface.co/collections/BayesRL/warm-started-checkpoints)** —
  Olmo-3, Qwen2.5-Math, and Llama-3.1 base models SFT'd with IVON on the Nemotron Post-Training Dataset.
  IVON learns a posterior (mean + diagonal Hessian) that seeds the 3PO RL runs.

## Models & data

- **Foundation models:** `allenai/Olmo-3-1025-7B`, `Qwen/Qwen2.5-Math-7B`, `meta-llama/Llama-3.1-8B`.
- **RL data:** [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k).
- **SFT data:** [Llama-Nemotron Post-Training Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset).
- **Benchmarks:** AIME 2024–2026, MATH-500, AMC 2023, Minerva.

## Citation

```bibtex
@misc{venkatkrishna2026parameter,
      title={Parameter Exploration for RLVR via Variational Learning},
      author={Vatsal Venkatkrishna and Nico Daheim and Iryna Gurevych},
      year={2026},
}
```
