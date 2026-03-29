import os
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
mapping = {
    "olmo3": "allenai/Olmo-3-7B-Instruct-DPO",
    "qwm": "Qwen/Qwen2.5-Math-7B",
}
model = sys.argv[1] if len(sys.argv) > 1 else "olmo3"
tok = AutoTokenizer.from_pretrained(mapping[model])


def tokenize_and_filter(example):
    prompt = tok.apply_chat_template(
        example["prompt"],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return prompt["input_ids"].shape[1] <= 1024


def edit_data_source(example):
    example["data_source"] = "math_dapo_base"
    return example


ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
df = ds.to_pandas()
df["idx"] = df["extra_info"].str["index"]
df = df.drop_duplicates("idx", keep="first").drop("idx", axis=1)
ds = Dataset.from_pandas(df)
ds = ds.filter(tokenize_and_filter, num_proc=NUM_WORKERS)
ds = ds.map(edit_data_source, num_proc=NUM_WORKERS)

save_dir = Path(__file__).parents[2] / "data"
ds.to_parquet(save_dir / f"{model}-base-dapomath-train.parquet")
