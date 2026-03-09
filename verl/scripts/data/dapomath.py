from pathlib import Path

from datasets import Dataset, load_dataset


def replace_answer_prompt(example):
    example["prompt"] = [
        {
            "role": "user",
            "content": example["prompt"][-1]["content"].replace(
                '\n\nRemember to put your answer on its own line after "Answer:"',
                "\n\nYour final answer should be in the form \\boxed{{answer}}. Any other format will be immediately rejected.",
            ),
        }
    ]
    return example


ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
df = ds.to_pandas()
df["idx"] = df["extra_info"].str["index"]
df = df.drop_duplicates("idx", keep="first").drop("idx", axis=1)
ds = Dataset.from_pandas(df)
# ds = ds.map(replace_answer_prompt)

save_dir = Path(__file__).parents[2] / "data"
ds.to_parquet(save_dir / "dapomath-train.parquet")
