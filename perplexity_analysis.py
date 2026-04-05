import os

import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from ivon.ivon import IVON
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Initialize Accelerator FIRST
accelerator = Accelerator()
device = accelerator.device

MODEL_ID = "Qwen/Qwen3-8B"

# 2. Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# Use the accelerator's device explicitly to ensure GPUs don't fight over GPU 0
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"": device},
    torch_dtype=torch.bfloat16,  # Halves VRAM usage compared to float32
)
model.eval()


def _load_optim_state_dict(path_or_repo, filename="optimizer.pt"):
    if os.path.exists(path_or_repo):
        path = os.path.join(path_or_repo, filename) if os.path.isdir(path_or_repo) else path_or_repo
    else:
        path = hf_hub_download(repo_id=path_or_repo, filename=filename)
    return torch.load(path, map_location="cpu", mmap=True)


def get_perplexity(model, input_ids, labels):
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        return torch.exp(outputs.loss)


# 3. Setup Optimizers
# Initialize IVON with the specific model parameters on the correct device
opt_scratch = IVON(model.parameters(), lr=5, weight_decay=1e-8, ess=1e8, clip_radius=1e-3, hess_init=1e-3)
opt_sft = IVON(model.parameters(), lr=5, weight_decay=1e-8, ess=1e8, clip_radius=1e-3, hess_init=1e-3)

# Load SFT state only if file exists
try:
    sft_state = _load_optim_state_dict(MODEL_ID)
    opt_sft.load_state_dict(sft_state)
    # Ensure state is moved to the specific GPU assigned to this process
    for group in opt_sft.param_groups:
        for key in ["momentum", "hess"]:
            if key in group:
                group[key] = group[key].to(device, dtype=torch.bfloat16)
except Exception as e:
    if accelerator.is_main_process:
        print(f"Skipping SFT loading: {e}")

# 4. Data Preparation
dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
dataset = dataset.shuffle(seed=42).select(range(40))


def preprocess(example):
    full_seq = [
        {"role": "user", "content": f"Answer the following question:\n{example['problem']}"},
        {"role": "assistant", "content": "<think>\n\n</think>\n\n" + example["solution"]},
    ]

    prompt = tokenizer.apply_chat_template([full_seq[0]], tokenize=False, thinking=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full_seq, tokenize=False, thinking=False, add_generation_prompt=False)

    enc = tokenizer(full_text, return_tensors="pt")
    prompt_enc = tokenizer(prompt, return_tensors="pt")

    labels = enc.input_ids.clone()
    labels[:, : prompt_enc.input_ids.shape[1]] = -100
    return {"input_ids": enc.input_ids, "labels": labels}


dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
dataset.set_format(type="torch")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

# 5. Parallelize Everything
model, opt_scratch, opt_sft, dataloader = accelerator.prepare(model, opt_scratch, opt_sft, dataloader)

results_local = {"std": [], "scratch": [], "sft": []}

# 6. Loop with unwrapped optimizers for sampling
for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Std PPL
    results_local["std"].append(get_perplexity(model, input_ids, labels))

    # IVON Sampling (Unwrap to access private sampling methods)
    for opt_key, accel_opt in [("scratch", opt_scratch), ("sft", opt_sft)]:
        raw_opt = accel_opt.optimizer  # Access underlying IVON object

        # Sampling
        noise = raw_opt._sample_params()
        results_local[opt_key].append(get_perplexity(model, input_ids, labels))

        # Cleanup memory immediately after use
        raw_opt._restore_param_average(train=False, noise=noise)
        del noise
        torch.cuda.empty_cache()  # Helps prevent fragmentation during sampling

# 7. Gather and Plot
all_std = accelerator.gather(torch.stack(results_local["std"])).cpu().tolist()
all_scratch = accelerator.gather(torch.stack(results_local["scratch"])).cpu().tolist()
all_sft = accelerator.gather(torch.stack(results_local["sft"])).cpu().tolist()

if accelerator.is_main_process:
    final_results = {"std": all_std, "scratch": all_scratch, "sft": all_sft}

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [final_results["std"], final_results["scratch"], final_results["sft"]],
        labels=["Standard", "IVON (Scratch)", "IVON (SFT)"],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="red"),
    )
    plt.ylabel("Perplexity")
    plt.title("Perplexity Distribution Comparison (Multi-GPU)")
    plt.savefig("perplexity_boxplot.png")

    for key in ["std", "scratch", "sft"]:
        avg = sum(final_results[key]) / len(final_results[key])
        print(f"Average {key.upper()} PPL: {avg:.4f}")
