import argparse
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("BayesRL/qwm7b_nmtron_ivon")
mapping = {
    "olmo3": "allenai/Olmo-3-7B-Instruct-DPO",
    "qwm": "Qwen/Qwen2.5-Math-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def tokenize_and_filter(example):
    prompt = tok.apply_chat_template(
        example["prompt"],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return prompt["input_ids"].shape[1] <= 1024


def difficulty_filter(example):
    return example["extra_info"]["model_difficulty"]["DeepSeek-R1-Distill-Qwen-7B"] == 16 and example["extra_info"]["model_difficulty"]["DeepSeek-R1-Distill-Qwen-32B"] != 16


def edit_data_source(example):
    example["extra_info"]["legacy_data_source"] = example["data_source"]
    example["data_source"] = "skywork_math_hard"
    return example


def main(args):
    data = load_dataset("Skywork/Skywork-OR1-RL-Data", split="math")
    data = data.filter(difficulty_filter, num_proc=NUM_WORKERS)
    data = data.filter(tokenize_and_filter, num_proc=NUM_WORKERS)
    data = data.map(edit_data_source, num_proc=NUM_WORKERS)
    save_dir = Path(__file__).parents[2] / "data"
    save_name = f"{args.model}-skywork-hard-train.parquet"
    if args.max_rows is not None:
        data = data.shuffle(seed=42).select(range(args.max_rows))
        save_name = f"{args.model}-skywork-hard-dc{args.max_rows}-train.parquet"
    data.to_parquet(save_dir / save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwm")
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(mapping[args.model])
    main(args)
