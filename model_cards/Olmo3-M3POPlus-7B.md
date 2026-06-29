---
license: apache-2.0
base_model: BayesRL/Olmo3-IVON-SFT-7B
datasets:
  - BytedTsinghua-SIA/DAPO-Math-17k
language:
  - en
pipeline_tag: text-generation
library_name: transformers
tags:
  - rlvr
  - grpo
  - ivon
  - variational-learning
  - 3po
  - m3po
  - math
---

# Olmo3-M3POPlus-7B

📦 **Code:** [insait-institute/c3po](https://github.com/insait-institute/c3po)

Olmo-3 7B fine-tuned with **M3PO+** (M3PO-Plus), from the paper **"Parameter Exploration for RLVR via
Variational Learning"**.

3PO is a family of *parameter-space* exploration strategies for Reinforcement Learning with Verifiable
Rewards (RLVR). Instead of relying only on action-space heuristics (temperature, clipping, entropy
bonuses), 3PO samples model weights from an approximate posterior learned with the variational optimizer
[IVON](https://arxiv.org/abs/2402.17641); the amount of weight noise becomes an extra control lever for
exploration.

**M3PO+** is the **non-equal-compute** M3PO run: `M = 4` Monte-Carlo weight perturbations per step with
the group size kept at `G = 16` (rather than reduced to `G/M`). It therefore uses ~4× the rollouts of the
compute-matched M3PO configuration, isolating the effect of more parameter-space samples when compute is
not held fixed.

## Training

| | |
|---|---|
| Base / warm-start | [`BayesRL/Olmo3-IVON-SFT-7B`](https://huggingface.co/BayesRL/Olmo3-IVON-SFT-7B) |
| Foundation model | `allenai/Olmo-3-1025-7B` |
| Algorithm | M3PO, `M = 4`, `G = 16` (non-equal-compute) |
| RL data | [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) |
| Optimizer | IVON, lr `1.0`, ESS (λ) `1e9` |
| Hardware | 8× NVIDIA H200 (144 GB) |

## Evaluation

Evaluated on AIME 2024–2026, MATH-500, AMC 2023, and Minerva. See the paper for full results.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("BayesRL/Olmo3-M3POPlus-7B")
tok = AutoTokenizer.from_pretrained("BayesRL/Olmo3-M3POPlus-7B")
```

## Citation

```bibtex
@misc{venkatkrishna2026parameter,
      title={Parameter Exploration for RLVR via Variational Learning},
      author={Vatsal Venkatkrishna and Nico Daheim and Iryna Gurevych},
      year={2026},
}
```
