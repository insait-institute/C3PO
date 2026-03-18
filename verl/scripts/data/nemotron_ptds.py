import os
import re
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B-SFT")
NUM_WORKERS = len(os.sched_getaffinity(0))
MAX_LEN = 4096 * 5
pattern = re.compile(r"<think>\s*\S+[\s\S]*?</think>[\s\S]*?\S[\s\S]*")


def tokenize_and_filter(example):
    if len(example["messages"]) != 2:
        return False

    prompt = tok.apply_chat_template(example["messages"], tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt")
    return prompt["input_ids"].shape[1] <= 4096


def filter_generator(example):
    return True if pattern.match(example["output"]) else False


def create_messages(example):
    example["messages"] = example["input"] + [{"role": "assistant", "content": example["output"]}]
    return example


if __name__ == "__main__":
    ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", "SFT")
    save_dir = Path(__file__).parents[2] / "data"
    save_dir.mkdir(exist_ok=True, parents=True)
    print([f"{k}: {len(ds[k])}" for k in ds])
    ds = ds.filter(filter_generator, num_proc=NUM_WORKERS)
    print([f"{k}: {len(ds[k])}" for k in ds])
    ds = ds.map(create_messages, num_proc=NUM_WORKERS, remove_columns=["input", "output"])
    ds = ds.filter(tokenize_and_filter, num_proc=NUM_WORKERS)
    print([f"{k}: {len(ds[k])}" for k in ds])

    ds_all = concatenate_datasets([ds[k] for k in ds])
    ds_all = ds_all.add_column("idx", [f"nmt_{i}" for i in range(len(ds_all))])
    ds_all = ds_all.select_columns(["idx", "messages", "category"])
    ds_all.to_parquet(save_dir / "nemotron_ptds.parquet")
