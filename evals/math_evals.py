import argparse
import ast
import logging
import math
import os
import pickle
import sys
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List
import torch
import constants
import numpy as np
import yaml
from datasets import Dataset, Value, concatenate_datasets, load_dataset
from formatter import BaseFormatter, get_formatter_mapping
from math_verify import parse, verify
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import wandb
from vllm import SamplingParams

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

NUM_WORKERS = len(os.sched_getaffinity(0))
sys.set_int_max_str_digits(0)


class Config(dict):
    """A dictionary that allows attribute-style access"""

    def __init__(self, d: Dict):
        super().__init__(d)
        for k, v in d.items():
            if isinstance(v, dict):
                self[k] = Config(v)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"Config has no attribute '{key}'")

    def get(self, key, default=None):
        return super().get(key, default)


def load_config_with_overrides() -> Config:
    """
    Loads config.yaml and applies CLI overrides.
    """
    parser = argparse.ArgumentParser(description="Math Evaluation Pipeline")
    parser.add_argument("--config", type=str, default="eval_config.yaml", help="Path to config file")
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.config):
        log.warning(f"Config file {args.config} not found. Using empty base config.")
        base_cfg = {}
    else:
        with open(args.config, "r") as f:
            base_cfg = yaml.safe_load(f) or {}

    # Apply overrides from unknown arguments (e.g., model.path=new/path or data=[a,b])
    for arg in unknown:
        if "=" in arg:
            key_path, value = arg.split("=", 1)
            try:
                parsed_value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed_value = value

            # Traverse and set the nested value
            keys = key_path.split(".")
            curr = base_cfg
            for k in keys[:-1]:
                curr = curr.setdefault(k, {})
            curr[keys[-1]] = parsed_value
            log.info(f"Overriding {key_path} with {parsed_value} ({type(parsed_value).__name__})")

    return Config(base_cfg)


def _prompt_map_fn(example, add_open_think):
    example["prompt"] = [{"role": "user", "content": constants.PROMPT_TEMPLATE.format(question=example["problem"])}]
    if add_open_think:
        example["prompt"].append({"role": "assistant", "content": "<think>\n"})
    return example


class MathEvalEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.model.path

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=cfg.model.tp_size or torch.cuda.device_count(),
            trust_remote_code=True,
            max_model_len=cfg.model.max_model_len,
            gpu_memory_utilization=0.95,
        )

    def _get_formatter(self, name: str, ds: Dataset) -> BaseFormatter:
        mapping = get_formatter_mapping()
        formatter_cls = mapping.get(name.lower(), BaseFormatter)
        return formatter_cls(name, ds)

    def load_and_prepare_data(self) -> Dataset:
        processed_list = []
        sources = self.cfg.data.names
        if sources == "all":
            sources = list(constants.DS_ID_MAP.keys())
        elif isinstance(sources, str):
            sources = [sources]

        for name in sources:
            path = constants.DS_ID_MAP.get(name, name)
            log.info(f"Loading dataset: {name} from {path}")
            ds = load_dataset(path, split="test")

            standardized = self._get_formatter(name, ds).format()
            standardized = standardized.cast_column("id", Value("string"))
            if "data_source" not in standardized.column_names:
                standardized = standardized.add_column("data_source", [name] * len(standardized))
            processed_list.append(standardized)

        data = concatenate_datasets(processed_list)
        data = data.map(
            _prompt_map_fn,
            fn_kwargs={"add_open_think": self.cfg.data.add_open_think},
            num_proc=NUM_WORKERS,
            desc="Creating prompts",
        )
        return data

    def _get_sampling_params(self, source_name: str) -> SamplingParams:
        """Merge default constants with hydra overrides and per-dataset settings."""
        base_params = constants.DEFAULT_SAMPLING_PARAMS.copy()
        if "sampling" in self.cfg:
            dataset_overrides = self.cfg.sampling.get(source_name, {})
            base_params.update(dataset_overrides)

        return SamplingParams(**base_params)

    def run_inference(self, dataset: Dataset) -> List[List[str]]:
        log.info(f"Starting batch inference for {len(dataset)} prompts...")

        prompts = list(dataset["prompt"])
        tokenizer_name = self.model_name
        if self.model_name == "allenai/Olmo-3-1025-7B" or "olmo3-base" in self.model_name:
            tokenizer_name = "allenai/Olmo-3-7B-Think-DPO"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        add_generation_prompt = prompts[0][-1]["role"] != "assistant"
        prompts = tokenizer.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            enable_thinking=self.cfg.model.enable_thinking,
        )
        sampling_params_list = [self._get_sampling_params(source) for source in dataset["data_source"]]

        outputs = self.llm.generate(prompts, sampling_params_list)
        return [[res.text for res in output.outputs] for output in outputs]

    def evaluate(self, predictions: List[List[str]], references: List[str], timeout_seconds: float = 5.0) -> Dict[str, Any]:
        """
        Verifies mathematical correctness with a per-sample timeout.
        """
        log.info("Scoring model responses...")
        all_sample_scores = []
        majority_scores = []

        for preds, ref in tqdm(zip(predictions, references), total=len(references), desc="Scoring..."):
            sample_scores = []
            parsed_preds = []

            try:
                if ref is not None:
                    ref_parsed = parse(ref)
                else:
                    ref_parsed = None

                # 2. Parse Predictions and Check Answers
                for p in preds:
                    try:
                        p_parsed = parse(p)[0]
                        is_correct = verify(ref_parsed, p_parsed)

                        parsed_preds.append(p_parsed)
                        sample_scores.append(is_correct)
                    except Exception as e:
                        log.warning(f"Error processing prediction: {e}")
                        parsed_preds.append(None)
                        sample_scores.append(False)

                all_sample_scores.append(sample_scores)

                # 3. Self-Consistency (Majority Vote)
                # Filter out None values from failed parses
                valid_parsed = [p for p in parsed_preds if p is not None]

                if valid_parsed:
                    str_preds = [str(p) for p in valid_parsed]
                    counts = Counter(str_preds)
                    majority_answer_str = counts.most_common(1)[0][0]

                    # Find the representative object
                    representative_idx = str_preds.index(majority_answer_str)
                    # This check is usually fast because it was already computed in sample_scores
                    # but we use the existing score to be safe.
                    orig_idx = [i for i, x in enumerate(parsed_preds) if x is not None][representative_idx]
                    majority_scores.append(sample_scores[orig_idx])
                else:
                    majority_scores.append(False)

            except Exception as e:
                log.error(f"Critical failure in sample evaluation: {e}")
                all_sample_scores.append([False] * len(preds))
                majority_scores.append(False)

        return {"sample_scores": all_sample_scores, "majority_scores": majority_scores}

    def report_metrics(self, dataset: Dataset, eval_results: Dict[str, Any]):
        """
        Calculates metrics for multi-sample evaluations:
        - Avg/Pass@1: Average correctness across all individual samples.
        - Best-of-i (for i=1 to n): Probability that at least one of i random samples is correct.
        - Majority Vote: Correct if the most frequent parsed answer is correct.
        """
        sources = np.array(dataset["data_source"])
        scores = eval_results["sample_scores"]
        maj_scores = eval_results["majority_scores"]

        def get_metrics_for_subset(indices):
            subset_scores = [scores[i] for i in indices]
            subset_maj = [maj_scores[i] for i in indices]

            if not subset_scores:
                return {}

            # Determine the maximum n present in this subset
            max_n = max(len(s) for s in subset_scores)

            # 1. Average (Pass@1)
            avg_acc = np.mean([np.mean(s) for s in subset_scores if len(s) > 0])

            # 2. Majority Vote
            maj_acc = np.mean(subset_maj)

            # 3. Best-of-i for i in [1..max_n] using combinatorial formula:
            # P(at least one correct in i samples) = 1 - ( (n-k)Ci / nCi )
            # where n = total samples, k = correct samples, i = selection size
            bon_metrics = {}
            for i in range(1, max_n + 1):
                bon_i_probs = []
                for s in subset_scores:
                    n = len(s)
                    if n < i:
                        continue  # Skip if prompt has fewer samples than i

                    k = sum(s)
                    # If i samples are picked from n, probability of all being wrong:
                    # (n-k)! / (n-k-i)!  /  (n! / (n-i)!)
                    if k == 0:
                        prob_any_correct = 0.0
                    elif k == n:
                        prob_any_correct = 1.0
                    else:
                        # Probability all i are from the (n-k) wrong ones
                        num = math.comb(n - k, i)
                        den = math.comb(n, i)
                        prob_any_correct = 1.0 - (num / den)
                    bon_i_probs.append(prob_any_correct)

                if bon_i_probs:
                    bon_metrics[f"bon_{i}"] = np.mean(bon_i_probs)

            res = {
                "avg": avg_acc,
                "maj": maj_acc,
                "bon": bon_metrics,
                "stderr": np.std([np.mean(s) for s in subset_scores if len(s) > 0]) / np.sqrt(len(subset_scores)),
            }
            return res

        global_metrics = get_metrics_for_subset(range(len(scores)))
        log.info(f"--- Eval results for {self.model_name} ---")
        global_bon_str = ""
        if "bon" in global_metrics:
            max_n_global = max(int(k.split("_")[1]) for k in global_metrics["bon"].keys())
            powers_of_two = [2**j for j in range(int(math.log2(max_n_global)) + 1)]
            relevant_global_bon = [f"{k}: {v:.2%}" for k, v in global_metrics["bon"].items() if int(k.split("_")[1]) in powers_of_two or int(k.split("_")[1]) == 1]
            global_bon_str = " | ".join(relevant_global_bon)

        log.info(f"average        | Avg: {global_metrics['avg']:.2%} | Maj: {global_metrics['maj']:.2%} | {global_bon_str}")

        logs = {
            "agg/avg_acc": global_metrics["avg"],
            "agg/maj_acc": global_metrics["maj"],
        }

        powers_of_two = [2**j for j in range(16)]
        for k, v in global_metrics.get("bon", {}).items():
            logs[f"agg/{k}"] = v

        # Per-dataset breakdown
        for source in np.unique(sources):
            indices = np.where(sources == source)[0]
            m = get_metrics_for_subset(indices)

            source_logs = {
                f"acc_{source}/avg": m["avg"],
                f"acc_{source}/maj": m["maj"],
                f"stderr_{source}": m["stderr"],
            }
            for k, v in m.get("bon", {}).items():
                source_logs[f"acc_{source}/{k}"] = v

            logs.update(source_logs)

            # Format console log to only show powers of two
            max_n_subset = max(int(k.split("_")[1]) for k in m.get("bon", {}).keys())
            powers_of_two = [2**j for j in range(int(math.log2(max_n_subset)) + 1)]
            relevant_bon = [f"{k}: {v:.2%}" for k, v in m.get("bon", {}).items() if int(k.split("_")[1]) in powers_of_two or int(k.split("_")[1]) == 1]
            bon_str = " | ".join(relevant_bon)

            log.info(f"{source:15} | Avg: {m['avg']:.2%} | Maj: {m['maj']:.2%} | {bon_str}")

        if self.cfg.wandb.enable:
            save_name = self.cfg.model.path
            if save_name.count("/") == 1:
                save_name = save_name.split("/")[-1]
            else:
                save_name = "--".join(self.cfg.model.path.split("/")[-3:-1])
            wandb.init(project=self.cfg.wandb.project, name=save_name, config=dict(self.cfg))
            wandb.log(logs)
            wandb.finish()


def main():
    cfg = load_config_with_overrides()
    engine = MathEvalEngine(cfg)
    dataset = engine.load_and_prepare_data()
    predictions = engine.run_inference(dataset)
    save_dir = Path(cfg.model.path)
    if save_dir.exists():
        with open(save_dir / "eval_predictions.pkl", "wb") as f:
            pickle.dump(predictions, f)
    else:
        save_dir = Path(__file__).parents[0] / "eval_preds" / cfg.model.path
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "eval_predictions.pkl", "wb") as f:
            pickle.dump(predictions, f)
    scores = engine.evaluate(predictions, dataset["answer"])
    engine.report_metrics(dataset, scores)


if __name__ == "__main__":
    main()
