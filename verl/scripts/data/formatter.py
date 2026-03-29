import os

from datasets import Dataset
from parser import extract_boxed_answer

NUM_WORKERS = len(os.sched_getaffinity(0))


class BaseFormatter:
    """
    Standardizes datasets to the schema: [problem, answer, id]
    If no transformation is defined, it assumes the dataset is already standardized.
    """

    REQUIRED_COLS = ["problem", "answer", "id"]

    def __init__(self, name: str, dataset: Dataset):
        self.name = name
        self.ds = dataset

    def format(self) -> Dataset:
        """Executes transformation and prunes extra columns."""
        ds = self.transform()
        return ds.select_columns(self.REQUIRED_COLS)

    def transform(self) -> Dataset:
        """Default behavior: no changes. Override in subclasses if needed."""
        return self.ds


class AIME24Formatter(BaseFormatter):
    def transform(self):
        self.ds = self.ds.rename_columns({"solution": "answer"})
        self.ds = self.ds.map(self._extract_boxed, num_proc=NUM_WORKERS, desc="Extracting gold answer for AIME'24")
        return self.ds

    def _extract_boxed(self, example):
        example["answer"] = extract_boxed_answer(example["answer"])[0]
        return example


class Math500Formatter(BaseFormatter):
    def transform(self):
        return self.ds.rename_columns({"unique_id": "id"})


class AMCFormatter(BaseFormatter):
    def transform(self):
        return self.ds.rename_columns({"question": "problem"})


class MinervaFormatter(BaseFormatter):
    def transform(self):
        self.ds = self.ds.rename_columns({"question": "problem"})
        return self.ds.add_column("id", [f"{self.name}_{i}" for i in range(len(self.ds))])


def get_formatter_mapping():
    """Returns the mapping of dataset keys to specialized formatter classes."""
    return {
        "aime24": AIME24Formatter,
        "math500": Math500Formatter,
        "amc": AMCFormatter,
        "minerva": MinervaFormatter,
    }
