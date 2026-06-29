---
license: apache-2.0
base_model: allenai/Olmo-3-1025-7B
datasets:
  - nvidia/Llama-Nemotron-Post-Training-Dataset
language:
  - en
pipeline_tag: text-generation
library_name: transformers
tags:
  - ivon
  - variational-learning
  - sft
  - 3po
  - math
  - reasoning
---

# Olmo3-IVON-SFT-7B

📦 **Code:** [insait-institute/c3po](https://github.com/insait-institute/c3po)

Olmo-3 7B supervised-fine-tuned with the variational optimizer
[IVON](https://arxiv.org/abs/2402.17641), from the paper **"Parameter Exploration for RLVR via
Variational Learning"**.

This is a **warm-start checkpoint**: SFT'ing with IVON yields not just point weights but an approximate
Gaussian posterior over them (a mean and a diagonal Hessian/precision estimate). That posterior is the
learned prior used to seed the **3PO** RLVR runs (B3PO / M3PO / C3PO), where weight perturbations sampled
from it drive parameter-space exploration.

## Training

| | |
|---|---|
| Foundation model | `allenai/Olmo-3-1025-7B` |
| Stage | Warm-start SFT |
| Data | [Llama-Nemotron Post-Training Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) (SFT subset) |
| Optimizer | IVON, lr `50.0`, ESS (λ) `1e10` |
| Hardware | 8× NVIDIA H200 (144 GB) |

## Usage

Loads as a standard causal LM:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("BayesRL/Olmo3-IVON-SFT-7B")
tok = AutoTokenizer.from_pretrained("BayesRL/Olmo3-IVON-SFT-7B")
```

To use it as the warm-start prior for 3PO RLVR, load the IVON optimizer state via
`IVON_INIT_METHOD=trained` in the companion code's `run_rl.sh`.

## Citation

```bibtex
@misc{venkatkrishna2026parameter,
      title={Parameter Exploration for RLVR via Variational Learning},
      author={Vatsal Venkatkrishna and Nico Daheim and Iryna Gurevych},
      year={2026},
}
```
