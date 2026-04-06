import os

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from ivon.ivon import IVON
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B"

# 1. Setup - Load once
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
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
        return torch.exp(outputs.loss).item()


def _quality(example):
    return len(example["problem"]) > 200 and len(example["solution"]) > 200


opt_scratch = IVON(model.parameters(), lr=5, weight_decay=1e-8, ess=1e8, clip_radius=1e-3, hess_init=1e-3)

# opt_sft = IVON(model.parameters(), lr=5, weight_decay=1e-8, ess=1e8, clip_radius=1e-3, hess_init=1e-3)
# sft_state = _load_optim_state_dict(MODEL_ID)
# opt_sft.load_state_dict(sft_state)
# for group in opt_sft.param_groups:
#     for key in ["momentum", "hess"]:
#         group[key] = group[key].to(model.device)

dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
dataset = dataset.filter(_quality)
dataset = dataset.shuffle(seed=42).select(range(20))


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


results = {"std": [], "scratch": [], "sft": []}

for example in tqdm(dataset):
    processed = preprocess(example)
    input_ids = processed["input_ids"].to(model.device)
    labels = processed["labels"].to(model.device)

    results["std"].append(get_perplexity(model, input_ids, labels))

    noise = opt_scratch._sample_params()
    results["scratch"].append(get_perplexity(model, input_ids, labels))
    opt_scratch._restore_param_average(train=False, noise=noise)

    # noise_sft = opt_sft._sample_params()
    # results["sft"].append(get_perplexity(model, input_ids, labels))
    # opt_sft._restore_param_average(train=False, noise=noise_sft)

plt.figure(figsize=(10, 6))
plt.boxplot(
    [results["std"], results["scratch"], results["sft"]],
    labels=["Standard", "IVON (Scratch)", "IVON (SFT)"],
    patch_artist=True,
    boxprops=dict(facecolor="lightblue"),
    medianprops=dict(color="red"),
)
plt.ylabel("Perplexity")
plt.title("Perplexity Distribution Comparison")
plt.savefig("perplexity_boxplot.png")

print(f"Average Standard PPL: {sum(results['std']) / len(results['std']):.4f}")
print(f"Average IVON Scratch PPL: {sum(results['scratch']) / len(results['scratch']):.4f}")
# print(f"Average IVON SFT PPL: {sum(results['sft']) / len(results['sft']):.4f}")
print(f"Min Standard PPL: {min(results['std']):.4f}")
print(f"Min IVON Scratch PPL: {min(results['scratch']):.4f}")
# print(f"Min IVON SFT PPL: {min(results['sft']):.4f}")
print(f"Max Standard PPL: {max(results['std']):.4f}")
print(f"Max IVON Scratch PPL: {max(results['scratch']):.4f}")
# print(f"Max IVON SFT PPL: {max(results['sft']):.4f}")
