---
license: apache-2.0
base_model: BayesRL/Qwen2.5Math-IVON-SFT-7B
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
  - c3po
  - math
---

# Qwen-C3PO-7B

📦 **Code:** [insait-institute/c3po](https://github.com/insait-institute/c3po)

Qwen2.5-Math 7B fine-tuned with **C3PO**, from the paper **"Parameter Exploration for RLVR via
Variational Learning"**.

3PO is a family of *parameter-space* exploration strategies for Reinforcement Learning with Verifiable
Rewards (RLVR). Instead of relying only on action-space heuristics (temperature, clipping, entropy
bonuses), 3PO samples model weights from an approximate posterior learned with the variational optimizer
[IVON](https://arxiv.org/abs/2402.17641); the amount of weight noise becomes an extra control lever for
exploration.

**C3PO** splits each GRPO group of `G` rollouts across `N` independent weight perturbations (`G/N`
rollouts each). Advantages are computed over the full, more-diverse group, with a Seq-MIS
(sequence-level multiple-importance-sampling) correction to account for the differing perturbations. This
gives extra parameter-space diversity at the same rollout budget as GRPO.

## Training

| | |
|---|---|
| Base / warm-start | [`BayesRL/Qwen2.5Math-IVON-SFT-7B`](https://huggingface.co/BayesRL/Qwen2.5Math-IVON-SFT-7B) |
| Foundation model | `Qwen/Qwen2.5-Math-7B` |
| Algorithm | C3PO (GRPO + IVON, `N` chunked perturbations + Seq-MIS) |
| RL data | [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) |
| Optimizer | IVON, lr `1.0`, ESS (λ) `1e10` |
| Hardware | 8× NVIDIA H200 (144 GB) |

## Evaluation

Evaluated on AIME 2024–2026, MATH-500, AMC 2023, and Minerva. See the paper for full results.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("BayesRL/Qwen-C3PO-7B")
tok = AutoTokenizer.from_pretrained("BayesRL/Qwen-C3PO-7B")
```

## Citation

```bibtex
@misc{venkatkrishna2026parameter,
      title={Parameter Exploration for RLVR via Variational Learning},
      author={Vatsal Venkatkrishna and Nico Daheim and Iryna Gurevych},
      year={2026},
}
```
