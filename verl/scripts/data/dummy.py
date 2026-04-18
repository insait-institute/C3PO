# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
from pathlib import Path

import datasets
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    data_source = "openai/gsm8k"

    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")
            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": "dummy_task",
                "prompt": [
                    {
                        "role": "user",
                        "content": f"Solve the following math problem. You must first think about your reasoning process and enclose it reasoning process within <think> and </think> tags, followed by your final answer within \\boxed{{}}. Any other format will be immediately rejected.\n{question}\nRemember to answer as follows:\n\n<think> reasoning process </think> \\boxed{{final_answer}}",
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question,
                },
            }
            return data

        return process_fn

    def tokenize_and_filter(example):
        prompt = tok.apply_chat_template(
            example["prompt"],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        return prompt["input_ids"].shape[1] <= 256

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_dataset = train_dataset.filter(tokenize_and_filter)
    save_dir = Path(__file__).parents[2] / "data"
    save_dir.mkdir(exist_ok=True, parents=True)
    train_dataset.to_parquet(save_dir / "dummy.parquet")
