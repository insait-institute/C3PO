import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from scipy.stats import norm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from ivon.ivon import IVON

NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-Math-7B"
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
    return example["num_tokens"] < 1024


def _domain_filter(example):
    return example["domain"] == "math"


def _preprocess(example):
    example["messages"] = [
        {"role": "user", "content": f"Answer the following question:\n{example['problem']}"},
        {"role": "assistant", "content": f"<think>\n{example['deepseek_reasoning']}\n\n{example['deepseek_solution']}"},
    ]
    example["prompt"] = tokenizer.apply_chat_template(example["messages"][:1], tokenize=False, add_generation_prompt=True, thinking=True)
    example["full_text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False, thinking=True)
    example["num_tokens"] = tokenizer.apply_chat_template(example["messages"], tokenize=True, return_dict=True, return_tensors="pt")["input_ids"].shape[1]
    return example


def _load_optim_state_dict(path_or_repo, filename="optimizer.pt"):
    if os.path.exists(path_or_repo):
        path = os.path.join(path_or_repo, filename) if os.path.isdir(path_or_repo) else path_or_repo
    else:
        path = hf_hub_download(repo_id=path_or_repo, filename=filename)
    return torch.load(path, map_location="cpu", mmap=True)


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

    dataset = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train")
    dataset = dataset.filter(_domain_filter, num_proc=NUM_WORKERS).remove_columns(["test_cases", "starter_code"])
    dataset = dataset.map(_preprocess, num_proc=NUM_WORKERS)
    dataset = dataset.filter(_quality, num_proc=NUM_WORKERS)
    print(f"[*] Filtered dataset to {len(dataset)} examples.")
    dataset = dataset.shuffle(seed=42).select(range(20))

    all_prompts = dataset["prompt"]
    all_full_texts = dataset["full_text"]

    # --- 5. Execution ---
    opt_scratch = IVON(model.parameters(), lr=5, weight_decay=1e-8, ess=1e8, clip_radius=1e-3, hess_init=1e-3)
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1)
    results = {"std": [], "scratch": [], "sft": []}

    # PHASE A: Standard Inference
    if rank == 0:
        print("Running Standard Batch...")
    std_outputs = llm.generate(all_full_texts, sampling_params)
    results["std"] = calculate_perplexity(std_outputs, all_prompts, all_full_texts, tokenizer)

    # PHASE B: IVON Sampling (Scratch)
    if rank == 0:
        print("Running IVON Scratch Batch...")
    opt_scratch = IVON(model.parameters(), lr=5, weight_decay=1e-8, ess=1e8, clip_radius=1e-3, hess_init=1e-3)
    torch.manual_seed(42)

    with opt_scratch.sampled_params():
        llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        for n, p in model.named_parameters():
            llm_model.load_weights([(n, p.data)])

        ivon_outputs = llm.generate(all_full_texts, sampling_params)
        results["scratch"] = calculate_perplexity(ivon_outputs, all_prompts, all_full_texts, tokenizer)

    # Cleanup Scratch Optimizer to free memory
    del opt_scratch
    torch.cuda.empty_cache()

    # PHASE C: IVON Sampling (SFT State)
    opt_sft = None
    try:
        if rank == 0:
            print("Loading SFT Optimizer States...")

        # Initialize fresh optimizer for SFT
        opt_sft = IVON(model.parameters(), lr=5, weight_decay=1e-8, ess=1e8, clip_radius=1e-3, hess_init=1e-3)
        sft_state = _load_optim_state_dict(MODEL_ID)
        opt_sft.load_state_dict(sft_state)

        # Move states to GPU
        for group in opt_sft.param_groups:
            for key in ["momentum", "hess"]:
                if key in group:
                    group[key] = group[key].to("cuda", dtype=torch.bfloat16)

        if rank == 0:
            print("Running IVON SFT Batch...")

        with opt_sft.sampled_params():
            llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
            for n, p in model.named_parameters():
                llm_model.load_weights([(n, p.data)])

            sft_outputs = llm.generate(all_full_texts, sampling_params)
            results["sft"] = calculate_perplexity(sft_outputs, all_prompts, all_full_texts, tokenizer)

    except Exception as e:
        if rank == 0:
            print(f"Skipping SFT loading or execution: {e}")
        opt_sft = None

    # --- Save and Plot ---
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(results, f)
    print(f"\nAverage Standard PPL: {np.mean(results['std']):.4f}")
    print(f"Average IVON PPL: {np.mean(results['scratch']):.4f}")
    print(f"stddev Standard PPL: {np.std(results['std']):.4f}")
    print(f"stddev IVON PPL: {np.std(results['scratch']):.4f}")
    if opt_sft:
        print(f"Average SFT PPL: {np.mean(results['sft']):.4f}")
        print(f"stddev SFT PPL: {np.std(results['sft']):.4f}")
    print(f"--- Results saved to {CACHE_FILE} ---")

print("[*] Generating distribution plots...")

# Convert to Log-Perplexity (Loss) for better Normal Distribution fit
log_std = np.log(results["std"])
log_scratch = np.log(results["scratch"])
if opt_sft:
    log_sft = np.log(results["sft"])

# Fit Normals
mu_std, std_std = norm.fit(log_std)
mu_ivon, std_ivon = norm.fit(log_scratch)
if opt_sft:
    mu_sft, std_sft = norm.fit(log_sft)

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

if opt_sft:
    # Plot SFT
    y_sft = norm.pdf(x, mu_sft, std_sft)
    plt.plot(x, y_sft, "g-", lw=2.5, label=f"IVON SFT (μ={mu_sft:.2f}, σ={std_sft:.2f})")
    plt.fill_between(x, y_sft, color="green", alpha=0.15)

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
