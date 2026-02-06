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
Convert allenai/tulu-3-sft-olmo-2-mixture-0225 to standard user-assistant messages.
"""

from pathlib import Path

import datasets
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B-SFT")


def tokenize_and_filter(example):
    prompt = tok.apply_chat_template(
        example["messages"],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    return prompt["input_ids"].shape[1] <= 4096 and len(example["messages"]) == 2


if __name__ == "__main__":
    data = datasets.load_dataset("allenai/tulu-3-sft-olmo-2-mixture-0225")["train"]
    save_dir = Path(__file__).parents[2] / "data"
    save_dir.mkdir(exist_ok=True, parents=True)
    data = data.filter(tokenize_and_filter, num_proc=32)
    data.to_parquet(save_dir / "tulu-3-sft-olmo-2-mixture-0225.parquet")
