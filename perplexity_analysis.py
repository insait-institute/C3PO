import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from ivon.ivon import IVON
from scipy.stats import norm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()
CACHE_FILE = f"ppl_results_{MODEL_ID.split('/')[-1]}.pkl"


def calculate_perplexity(outputs, prompts, full_texts, tokenizer):
    batch_ppl = []
    for output, prompt, full_text in zip(outputs, prompts, full_texts):
        logprobs_list = output.prompt_logprobs

        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(full_text)

        target_logprobs = []
        for i in range(len(prompt_ids), len(full_ids)):
            if i < len(logprobs_list) and logprobs_list[i] is not None:
                token_id = full_ids[i]
                if token_id in logprobs_list[i]:
                    target_logprobs.append(logprobs_list[i][token_id].logprob)

        if not target_logprobs:
            batch_ppl.append(float("inf"))
        else:
            batch_ppl.append(np.exp(-np.mean(target_logprobs)))
    return batch_ppl


def _quality(example):
    return len(example["problem"]) > 200 and len(example["solution"]) > 200


if os.path.exists(CACHE_FILE):
    print(f"[*] Found cached results in {CACHE_FILE}. Skipping inference.")
    with open(CACHE_FILE, "rb") as f:
        results = pickle.load(f)
else:
    print("--- No cache found. Running model inference... ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu")

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=world_size,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=0.95,
        max_num_seqs=16,
        trust_remote_code=True,
    )

    dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    dataset = dataset.filter(_quality).shuffle(seed=42).select(range(20))

    all_prompts = []
    all_full_texts = []

    for example in dataset:
        full_seq = [
            {"role": "user", "content": f"Answer the following question:\n{example['problem']}"},
            {"role": "assistant", "content": example["solution"]},
        ]
        all_prompts.append(tokenizer.apply_chat_template([full_seq[0]], tokenize=False, add_generation_prompt=True, thinking=False))
        all_full_texts.append(tokenizer.apply_chat_template(full_seq, tokenize=False, add_generation_prompt=False, thinking=False))

    # --- 5. Execution ---
    opt_scratch = IVON(model.parameters(), lr=5, weight_decay=1e-8, ess=1e8, clip_radius=1e-3, hess_init=1e-3)
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1)
    results = {"std": [], "scratch": []}

    # PHASE A: Standard Inference
    if rank == 0:
        print("Running Standard Batch...")
    std_outputs = llm.generate(all_full_texts, sampling_params)
    results["std"] = calculate_perplexity(std_outputs, all_prompts, all_full_texts, tokenizer)

    # PHASE B: IVON Sampling
    if rank == 0:
        print(f"\nAverage Standard PPL: {np.mean(results['std']):.4f}")
        print("Running IVON Batch...")
    torch.manual_seed(42)

    with opt_scratch.sampled_params():
        llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model

        for n, p in model.named_parameters():
            llm_model.load_weights([(n, p.data)])

        ivon_outputs = llm.generate(all_full_texts, sampling_params)
        results["scratch"] = calculate_perplexity(ivon_outputs, all_prompts, all_full_texts, tokenizer)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(results, f)
    print(f"--- Results saved to {CACHE_FILE} ---")

print("[*] Generating distribution plots...")

# Convert to Log-Perplexity (Loss) for better Normal Distribution fit
log_std = np.log(results["std"])
log_scratch = np.log(results["scratch"])

# Fit Normals
mu_std, std_std = norm.fit(log_std)
mu_ivon, std_ivon = norm.fit(log_scratch)

# Create Plot
plt.figure(figsize=(10, 6), dpi=120)
plt.style.use("seaborn-v0_8-muted")  # Cleaner style

x_min = min(log_std.min(), log_scratch.min()) - 1.0
x_max = max(log_std.max(), log_scratch.max()) + 1.0
x = np.linspace(x_min, x_max, 500)

# Plot Standard
y_std = norm.pdf(x, mu_std, std_std)
plt.plot(x, y_std, "b-", lw=2.5, label=f"Standard (μ={mu_std:.2f}, σ={std_std:.2f})")
plt.fill_between(x, y_std, color="blue", alpha=0.15)

# Plot IVON
y_ivon = norm.pdf(x, mu_ivon, std_ivon)
plt.plot(x, y_ivon, "r-", lw=2.5, label=f"IVON Scratch (μ={mu_ivon:.2f}, σ={std_ivon:.2f})")
plt.fill_between(x, y_ivon, color="red", alpha=0.15)

plt.title("Distribution of Log-Perplexity (Model Loss)", fontsize=14, fontweight="bold")
plt.xlabel("Log-Perplexity (Lower is better)", fontsize=12)
plt.ylabel("Probability Density", fontsize=12)
plt.legend(frameon=True, facecolor="white", framealpha=0.9)
plt.grid(True, linestyle="--", alpha=0.4)

# Summary Textbox
stats_text = f"Standard PPL Mean: {np.mean(results['std']):.2f}\nIVON PPL Mean: {np.mean(results['scratch']):.2f}"
plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

plt.tight_layout()
plt.savefig("ppl_log_normal_fit.png")
print("[*] Plot saved as 'ppl_log_normal_fit.png'")
plt.show()
