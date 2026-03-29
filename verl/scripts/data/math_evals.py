import argparse
import logging
from pathlib import Path

# Import our project abstractions
from datasets import Value, concatenate_datasets, load_dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
PROMPT_TEMPLATE = "You are an expert mathematics problem solver. Think carefully step-by-step and reply with your final answer within \\boxed{{}}.\nQuestion:\n\n{question}"


def process_fn(example, idx):
    return {
        "data_source": example["data_source"],
        "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=example["problem"])}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": example["answer"]},
        "extra_info": {"idx": example["id"]},
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Math Datasets for RLHF")
    parser.add_argument("--output", type=str, default="math_evals.parquet", help="Output path")
    args = parser.parse_args()

    # Determine sources
    sources = ["HuggingFaceH4/aime_2024", "math-ai/aime25", "math-ai/aime26", "math-ai/math500"]
    all_processed = []

    for path in sources:
        try:
            raw_ds = load_dataset(path, split="test")
        except:
            raw_ds = load_dataset(path, split="train")
        if "unique_id" in raw_ds.column_names:
            raw_ds = raw_ds.rename_columns({"unique_id": "id"})
        raw_ds = raw_ds.cast_column("id", Value("string"))
        raw_ds = raw_ds.select_columns(["id", "problem", "answer"])
        raw_ds = raw_ds.add_column("data_source", [path.split("/")] * len(raw_ds))
        all_processed.append(raw_ds)

    data = concatenate_datasets(all_processed)
    data = data.map(process_fn, with_indices=True, num_proc=16)
    log.info(f"Saving to {args.output}...")
    save_dir = Path(__file__).parents[2] / "data"
    save_dir.mkdir(exist_ok=True, parents=True)

    data.to_parquet(save_dir / args.output)
    log.info("Done!")
    breakpoint()


if __name__ == "__main__":
    main()
