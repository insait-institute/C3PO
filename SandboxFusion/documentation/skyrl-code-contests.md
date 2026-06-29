# SkyRL Environment for Code-Contests-O

This tutorial walks through building a complete [SkyRL](https://docs.skyrl.ai/) reinforcement learning environment that trains an LLM to solve competitive programming problems from the [Code-Contests-O](https://huggingface.co/datasets/OctoReasoner/Code-Contests-O) dataset, using SandboxFusion as the code execution backend.

By the end you will have:

1. A dataset conversion script that transforms Code-Contests-O into SkyRL's Parquet format.
2. A custom SkyRL environment (`CodeContestsEnv`) that extracts Python code from model completions, runs it against hidden test cases via SandboxFusion, and returns a reward.
3. A training entrypoint and config to launch GRPO training.

## Prerequisites

- **SkyRL** installed ([installation guide](https://docs.skyrl.ai/docs/getting-started/installation))
- **SandboxFusion** server running (see [Getting Started](getting-started.md))
- **Code-Contests-O** dataset downloaded locally (290 Arrow shard files)
- Python 3.11+, `pyarrow`, `pandas`
- GPUs for training (the config below assumes 4x GPUs)

---

## Step 1: Understand the Dataset

Code-Contests-O is a collection of 8,215 competitive programming problems with hidden test cases. Each row has:

| Field | Type | Content |
|-------|------|---------|
| `data_source` | `string` | Always `"code_contests_o"` |
| `prompt` | `list[{role, content}]` | A single user message containing a system preamble and the problem statement enclosed in `[QUESTION]...[/QUESTION]` tags |
| `ability` | `string` | Always `"code"` |
| `reward_model` | `struct{method, ground_truth}` | `method` is `"rule"`. `ground_truth` is a JSON string with `{"inputs": [...], "outputs": [...]}` — the hidden judge test cases |
| `extra_info` | `struct{index}` | Unique problem ID (e.g., `"cco_train_2722"`) |

The `prompt` field already contains a well-structured instruction asking the model to read from stdin, solve the problem, and reply with a markdown code snippet. The `ground_truth` field contains 13–64+ stdin/stdout test case pairs per problem.

### Inspect a sample row

```python
import pyarrow.ipc as ipc

path = "/path/to/Code-Contests-0/data-00000-of-00290.arrow"
with open(path, "rb") as f:
    reader = ipc.open_stream(f)
    table = reader.read_all()

row = table.to_pydict()
print(row["prompt"][0])          # list of message dicts
print(row["reward_model"][0])    # {"method": "rule", "ground_truth": "{...}"}
print(row["extra_info"][0])      # {"index": "cco_train_0"}
```

---

## Step 2: Convert to SkyRL Parquet Format

SkyRL expects Parquet files with these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `data_source` | Yes | Identifier for the data source |
| `prompt` | Yes | List of `{role, content}` message dicts |
| `env_class` | Yes | Registered environment name (must match `register(id=...)`) |
| `reward_spec` | Yes | Dict passed to the environment (contains reward method and ground truth) |
| `extra_info` | No | Arbitrary metadata dict |

The Code-Contests-O dataset is close but needs three changes:
1. **Rename** `reward_model` → `reward_spec` (SkyRL's expected field name).
2. **Add** an `env_class` column set to `"code_contests"`.
3. **Drop** the `ability` column (not used by SkyRL).

Create `convert_dataset.py`:

```python
#!/usr/bin/env python3
"""Convert Code-Contests-O Arrow shards to SkyRL Parquet format."""

import argparse
import json
import os

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


def load_arrow_shards(data_dir: str) -> pa.Table:
    """Load all Arrow IPC stream files from a directory into a single table."""
    tables = []
    shard_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".arrow"))
    for filename in shard_files:
        path = os.path.join(data_dir, filename)
        with open(path, "rb") as f:
            reader = ipc.open_stream(f)
            tables.append(reader.read_all())
    return pa.concat_tables(tables)


def convert_row(row: dict) -> dict:
    """Transform a single Code-Contests-O row into SkyRL format."""
    return {
        "data_source": row["data_source"],
        "prompt": row["prompt"],
        "env_class": "code_contests",
        "reward_spec": {
            "method": row["reward_model"]["method"],
            "ground_truth": row["reward_model"]["ground_truth"],
        },
        "extra_info": {
            **row["extra_info"],
            "ability": row.get("ability", "code"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Convert Code-Contests-O to SkyRL Parquet")
    parser.add_argument("--input_dir", required=True, help="Path to Code-Contests-0/ directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for Parquet files")
    parser.add_argument("--val_size", type=int, default=200, help="Number of validation examples")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Arrow shards from {args.input_dir}...")
    table = load_arrow_shards(args.input_dir)
    rows = table.to_pylist()
    print(f"Loaded {len(rows)} examples")

    converted = [convert_row(r) for r in rows]

    # Split into train / validation
    val_rows = converted[: args.val_size]
    train_rows = converted[args.val_size :]

    for split_name, split_rows in [("train", train_rows), ("validation", val_rows)]:
        # Build Arrow arrays from Python dicts
        records = {
            "data_source": [r["data_source"] for r in split_rows],
            "prompt": [r["prompt"] for r in split_rows],
            "env_class": [r["env_class"] for r in split_rows],
            "reward_spec": [r["reward_spec"] for r in split_rows],
            "extra_info": [r["extra_info"] for r in split_rows],
        }
        out_table = pa.Table.from_pydict(records)
        out_path = os.path.join(args.output_dir, f"{split_name}.parquet")
        pq.write_table(out_table, out_path)
        print(f"Wrote {len(split_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python convert_dataset.py \
    --input_dir ~/Downloads/Code-Contests-0 \
    --output_dir ~/data/code_contests \
    --val_size 200
```

This produces `~/data/code_contests/train.parquet` (8,015 rows) and `~/data/code_contests/validation.parquet` (200 rows).

---

## Step 3: Build the Environment

The environment receives the model's completion, extracts Python code from it, runs the code against hidden test cases via SandboxFusion, and returns a reward based on the fraction of test cases that pass.

Create `code_contests_env.py`:

```python
"""SkyRL environment for Code-Contests-O competitive programming problems.

Extracts Python code from the model's markdown-fenced completion, runs it
against hidden stdin/stdout test cases via SandboxFusion, and returns a
reward equal to 1.0 if ALL test cases pass (0.0 otherwise).
"""

import json
import re
from typing import Optional

import requests
from skyrl_gym.core import BaseTextEnv, BaseTextEnvStepOutput


SANDBOX_ENDPOINT = "http://localhost:8080"
RUN_TIMEOUT = 30
COMPILE_TIMEOUT = 10


def extract_python_code(completion: str) -> Optional[str]:
    """Extract Python code from a markdown-fenced code block.

    Tries these patterns in order:
    1. ```python ... ```
    2. ``` ... ```  (unfenced but triple-backtick)
    3. The entire completion as raw code (fallback)
    """
    # Try fenced python block
    match = re.search(r"```python\s*\n(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic fenced block
    match = re.search(r"```\s*\n(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: treat entire response as code
    return completion.strip() if completion.strip() else None


def run_against_test_cases(
    code: str,
    inputs: list[str],
    outputs: list[str],
    endpoint: str = SANDBOX_ENDPOINT,
) -> tuple[int, int]:
    """Run code against stdin/stdout test cases via SandboxFusion.

    Uses the /submit endpoint which handles test case evaluation natively.

    Args:
        code: Python source code to execute.
        inputs: List of stdin strings (one per test case).
        outputs: List of expected stdout strings (one per test case).
        endpoint: SandboxFusion server URL.

    Returns:
        (passed, total) tuple.
    """
    test_cases = []
    for stdin, stdout in zip(inputs, outputs):
        test_cases.append({
            "input": {"stdin": stdin},
            "output": {"stdout": stdout},
        })

    payload = {
        "id": "eval",
        "completion": f"```python\n{code}\n```",
        "config": {
            "language": "python",
            "run_timeout": RUN_TIMEOUT,
            "compile_timeout": COMPILE_TIMEOUT,
        },
        "test_cases": test_cases,
    }

    try:
        resp = requests.post(f"{endpoint}/submit", json=payload, timeout=300)
        resp.raise_for_status()
        result = resp.json()

        total = len(test_cases)
        passed = sum(1 for t in result.get("tests", []) if t.get("passed"))
        return passed, total
    except Exception:
        return 0, len(test_cases)


class CodeContestsEnv(BaseTextEnv):
    """Single-turn environment for competitive programming.

    The model generates a Python solution in one shot. The environment
    extracts code, runs it against hidden test cases, and returns a binary
    reward (1.0 if all tests pass, 0.0 otherwise).
    """

    def __init__(self, env_config: dict, extras: dict):
        super().__init__()
        reward_spec = extras.get("reward_spec", {})
        gt_raw = reward_spec.get("ground_truth", "{}")
        if isinstance(gt_raw, str):
            gt = json.loads(gt_raw)
        else:
            gt = gt_raw
        self.inputs = gt.get("inputs", [])
        self.outputs = gt.get("outputs", [])
        self.endpoint = env_config.get("sandbox_endpoint", SANDBOX_ENDPOINT)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        code = extract_python_code(action)
        if code is None:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={"error": "no_code_extracted", "passed": 0, "total": len(self.inputs)},
            )

        passed, total = run_against_test_cases(
            code, self.inputs, self.outputs, self.endpoint
        )
        reward = 1.0 if passed == total and total > 0 else 0.0

        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata={"passed": passed, "total": total, "code": code[:500]},
        )
```

### Key design decisions

- **Binary reward**: 1.0 only when *all* test cases pass. This matches the standard competitive programming acceptance criterion. An alternative is fractional reward (`passed / total`), which provides denser signal but may train the model to produce partially-correct solutions.
- **Single-turn**: The model gets one attempt. For a multi-turn variant, see [Extending to multi-turn](#extending-to-multi-turn) below.
- **SandboxFusion `/submit` endpoint**: Handles code extraction, test case execution, and stdout comparison in a single call. The environment wraps the extracted code back in fenced markers so SandboxFusion's extractor picks it up cleanly.

---

## Step 4: Register the Environment and Create the Entrypoint

Create `train_code_contests.py`:

```python
"""SkyRL training entrypoint for Code-Contests-O."""

import ray
from skyrl.config import SkyRLTrainConfig
from skyrl.core import BasePPOExp
from skyrl_gym import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    # Register the custom environment
    register(
        id="code_contests",
        entry_point="code_contests_env:CodeContestsEnv",
    )

    exp = BasePPOExp(cfg)
    exp.run()


def main():
    ray.init()
    cfg = SkyRLTrainConfig.from_yaml("config.yaml")
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
```

The `id="code_contests"` must match the `env_class` column in the Parquet dataset.

The `entry_point` is a dotted Python import path. Adjust it to match your project layout (e.g., `"my_project.envs.code_contests_env:CodeContestsEnv"`).

---

## Step 5: Write the Training Config

Create `config.yaml`:

```yaml
# SkyRL training config for Code-Contests-O
# Assumes 4 GPUs and SandboxFusion running on localhost:8080

data:
  train_data:
    - "~/data/code_contests/train.parquet"
  val_data:
    - "~/data/code_contests/validation.parquet"

environment:
  env_class: "code_contests"    # default env class (matches register id)
  env_config:
    sandbox_endpoint: "http://localhost:8080"

generator:
  n_samples_per_prompt: 4       # 4 rollouts per problem for variance reduction
  max_new_tokens: 2048          # competitive programming solutions can be long
  max_turns: 1                  # single-turn (one-shot code generation)
  batched: true                 # batch generation (efficient for single-turn)
  temperature: 0.7
  top_p: 0.95
  inference_engine:
    backend: "vllm"
    tensor_parallel_size: 1

model:
  model_path: "Qwen/Qwen2.5-Coder-7B-Instruct"   # or any code-capable model
  tokenizer_path: "Qwen/Qwen2.5-Coder-7B-Instruct"

trainer:
  total_epochs: 3
  save_freq: 50
  micro_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-6
  warmup_ratio: 0.05
  kl_coef: 0.01
  algorithm:
    advantage_estimator: "grpo"   # Group Relative Policy Optimization
  strategy: "fsdp2"
  placement:
    colocate_all: true            # colocate policy + generation on same GPUs

logging:
  project: "code-contests-rl"
  run_name: "grpo-7b-coder"
```

### Config notes

| Setting | Value | Rationale |
|---------|-------|-----------|
| `n_samples_per_prompt` | 4 | GRPO needs multiple rollouts per prompt to compute group-relative advantages. 4 is a good starting point; increase to 8 for more stable training. |
| `max_new_tokens` | 2048 | Competitive programming solutions are typically 20–100 lines, but some require more. 2048 tokens gives ample room. |
| `max_turns` | 1 | Single-turn: the model generates a solution in one shot. |
| `temperature` | 0.7 | Encourages exploration during training. Lower (0.3–0.5) for evaluation. |
| `kl_coef` | 0.01 | Mild KL penalty to prevent the policy from diverging too far from the reference model. |
| `advantage_estimator` | `"grpo"` | GRPO computes advantages relative to other samples for the same prompt, which works well for binary rewards. |

---

## Step 6: Launch Training

Start SandboxFusion:

```bash
# Terminal 1: start the sandbox server
docker run -d --rm --privileged -p 8080:8080 ineil77/sandbox-fusion-server:25042026-2
```

Verify it's healthy:

```bash
curl http://localhost:8080/v1/ping
# "pong"
```

Launch SkyRL training:

```bash
# Terminal 2: start training
export WANDB_API_KEY=your_wandb_api_key
python train_code_contests.py
```

---

## Step 7: Monitor and Evaluate

### Training metrics to watch

| Metric | Healthy range | Meaning |
|--------|--------------|---------|
| `reward/mean` | 0.05–0.30 (epoch 1), rising | Average reward across rollouts. Competitive programming is hard; even 10% solve rate is strong. |
| `reward/std` | > 0.1 | Indicates variance in rollout outcomes — needed for GRPO to compute meaningful advantages. If near 0, the model is stuck. |
| `kl_divergence` | < 10.0 | Policy divergence from reference. If this spikes, lower `learning_rate` or increase `kl_coef`. |
| `response_length/mean` | 200–800 tokens | Typical code solution lengths. If this maxes out at `max_new_tokens`, increase the limit. |

### Offline evaluation

After training, evaluate the checkpoint on the validation set:

```python
"""Evaluate a trained checkpoint on Code-Contests-O validation set."""

import json
import pandas as pd
from vllm import LLM, SamplingParams

from code_contests_env import extract_python_code, run_against_test_cases

# Load model
llm = LLM(model="path/to/checkpoint", tensor_parallel_size=1)
params = SamplingParams(temperature=0.0, max_tokens=2048)  # greedy for eval

# Load validation set
df = pd.read_parquet("~/data/code_contests/validation.parquet")

solved = 0
for _, row in df.iterrows():
    messages = row["prompt"]
    prompt_text = messages[0]["content"] if isinstance(messages[0], dict) else str(messages[0])

    outputs = llm.generate([prompt_text], params)
    completion = outputs[0].outputs[0].text

    code = extract_python_code(completion)
    if code is None:
        continue

    gt = json.loads(row["reward_spec"]["ground_truth"])
    passed, total = run_against_test_cases(code, gt["inputs"], gt["outputs"])
    if passed == total and total > 0:
        solved += 1

print(f"Solved: {solved}/{len(df)} ({solved/len(df):.1%})")
```

---

## Extending to Multi-Turn

The single-turn environment above can be extended to give the model feedback and a second attempt. This is useful when the model produces almost-correct solutions.

```python
class CodeContestsMultiTurnEnv(BaseTextEnv):
    """Multi-turn variant: model gets feedback and can retry."""

    def __init__(self, env_config: dict, extras: dict):
        super().__init__()
        reward_spec = extras.get("reward_spec", {})
        gt_raw = reward_spec.get("ground_truth", "{}")
        gt = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
        self.inputs = gt.get("inputs", [])
        self.outputs = gt.get("outputs", [])
        self.endpoint = env_config.get("sandbox_endpoint", SANDBOX_ENDPOINT)
        self.max_turns = env_config.get("max_turns", 3)
        self.turn = 0

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turn += 1
        code = extract_python_code(action)

        if code is None:
            if self.turn >= self.max_turns:
                return BaseTextEnvStepOutput(
                    observations=[], reward=0.0, done=True,
                    metadata={"error": "no_code_extracted"},
                )
            return BaseTextEnvStepOutput(
                observations=[{"role": "user", "content":
                    "Your response did not contain a Python code block. "
                    "Please reply with a ```python ... ``` code block."}],
                reward=0.0, done=False, metadata={},
            )

        passed, total = run_against_test_cases(
            code, self.inputs, self.outputs, self.endpoint
        )

        if passed == total and total > 0:
            return BaseTextEnvStepOutput(
                observations=[], reward=1.0, done=True,
                metadata={"passed": passed, "total": total},
            )

        if self.turn >= self.max_turns:
            return BaseTextEnvStepOutput(
                observations=[], reward=0.0, done=True,
                metadata={"passed": passed, "total": total},
            )

        # Provide feedback for the next attempt
        feedback = (
            f"Your solution passed {passed}/{total} test cases. "
            f"Please fix the bugs and try again."
        )
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": feedback}],
            reward=0.0, done=False,
            metadata={"passed": passed, "total": total},
        )
```

For multi-turn training, update the config:

```yaml
generator:
  max_turns: 3
  batched: false                          # required for multi-turn
  inference_engine:
    async_engine: true                    # enables async rollouts
```

---

## Alternative Reward Strategies

The binary reward (1.0 if all tests pass) provides a clean signal but is sparse. Here are alternatives:

### Fractional reward

```python
reward = passed / total if total > 0 else 0.0
```

Denser signal, but may train the model to produce solutions that pass easy test cases while ignoring edge cases.

### Graded with format bonus

```python
if passed == total and total > 0:
    reward = 1.0
elif code is not None and passed > 0:
    reward = 0.3 * (passed / total)   # partial credit
elif code is not None:
    reward = 0.05                      # at least produced valid code
else:
    reward = 0.0                       # no code extracted
```

### Execution-based shaping

```python
# Run just the first test case for fast feedback
first_passed, _ = run_against_test_cases(code, inputs[:1], outputs[:1])
if first_passed == 0:
    reward = 0.0   # doesn't even pass the example
else:
    # Run all test cases
    passed, total = run_against_test_cases(code, inputs, outputs)
    reward = 1.0 if passed == total else 0.2 * (passed / total)
```

---

## Scaling Tips

### SandboxFusion concurrency

Each rollout calls SandboxFusion to execute code. With `n_samples_per_prompt=4` and batch parallelism, you may have dozens of concurrent execution requests. Tune SandboxFusion's concurrency to match:

```yaml
# SandboxFusion local.yaml
sandbox:
  isolation: lite
  max_concurrency: 64     # handle parallel rollouts
  default_memory_limit_mb: 2048   # competitive programming rarely needs >2 GB
  default_cpu_limit: 1            # 1 core per execution is sufficient
```

### Timeout tuning

Competitive programming problems have varying time limits. The default `run_timeout=30` is generous. For faster training, you can reduce it to 10–15 seconds (most correct solutions finish in under 5 seconds). Be careful not to set it too low — some problems require O(n log n) solutions that take a few seconds on large inputs.

### Dataset filtering

Some Code-Contests-O problems are extremely hard (rated 2400+ on Codeforces). You may want to filter by difficulty or by the number of test cases to focus training on problems the model has a reasonable chance of solving:

```python
# In convert_dataset.py, optionally filter
import json

def is_tractable(row: dict, max_test_cases: int = 50) -> bool:
    gt = json.loads(row["reward_model"]["ground_truth"])
    return len(gt["inputs"]) <= max_test_cases
```

---

## File Layout

```
my_project/
├── convert_dataset.py          # Step 2: Arrow → Parquet conversion
├── code_contests_env.py        # Step 3: SkyRL environment
├── train_code_contests.py      # Step 4: Training entrypoint
├── config.yaml                 # Step 5: Training config
└── eval_checkpoint.py          # Step 7: Offline evaluation
```

## Next Steps

- [SkyRL New Environment Tutorial](https://docs.skyrl.ai/docs/tutorials/new_env) — the upstream guide this tutorial is based on
- [SkyRL Tools Guide](https://docs.skyrl.ai/docs/tutorials/tools_guide) — for environments that use tool calling
- [Isolation Modes](isolation-modes.md) — understanding lite vs full sandbox isolation
- [Configuration](configuration.md) — tuning SandboxFusion for your workload
- [API Reference](api-reference.md) — the `/submit` endpoint used by the environment
