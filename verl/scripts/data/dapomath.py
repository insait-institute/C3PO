import os
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
model = sys.argv[1] if len(sys) > 1 else "olmo3-instruct"
mapping = {
    "olmo3-instruct": "allenai/Olmo-3-7B-Instruct-DPO",
    "olmo3-base": "allenai/Olmo-3-1025-7B",
    "qwen2-5-instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
    "qwen2-5-base": "Qwen/Qwen2.5-Math-7B",
}
tok = AutoTokenizer.from_pretrained(mapping[model])


def replace_answer_prompt(example):
    math_question = example["prompt"][-1]["content"]
    math_question = math_question.replace(
        "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.",
        "",
    )
    math_question = math_question.replace('Remember to put your answer on its own line after "Answer:".', "").strip()

    example["prompt"] = [
        {
            "role": "user",
            "content": f"Solve the following math problem. You must first think about your reasoning process and enclose it reasoning process within <think> and </think> tags, followed by your final answer within \\boxed{{}}. Any other format will be immediately rejected.\n{math_question}\nRemember to answer as follows:\n\n<think> reasoning process </think> \\boxed{{final_answer}}",
        }
    ]
    return example


def edit_data_source(example):
    example["data_source"] = "math_dapo_instruct"
    return example


def tokenize_and_filter(example):
    prompt = tok.apply_chat_template(
        example["prompt"],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return prompt["input_ids"].shape[1] <= 1024


ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
df = ds.to_pandas()
df["idx"] = df["extra_info"].str["index"]
df = df.drop_duplicates("idx", keep="first").drop("idx", axis=1)
ds = Dataset.from_pandas(df)
ds = ds.map(replace_answer_prompt, num_proc=NUM_WORKERS)
ds = ds.filter(tokenize_and_filter, num_proc=NUM_WORKERS)
ds = ds.map(edit_data_source, num_proc=NUM_WORKERS)
save_dir = Path(__file__).parents[2] / "data"
ds.to_parquet(save_dir / f"{model}-dapomath-train.parquet")
