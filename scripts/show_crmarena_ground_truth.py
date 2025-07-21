import argparse
from typing import List, Dict

from datasets import load_dataset, Dataset

HF_DATASET_NAME = "Salesforce/CRMArenaPro"
HF_CONFIG_NAME = "CRMArenaPro"

POLICY_TASKS: List[str] = [
    "invalid_config",
    "policy_violation_identification",
    "lead_qualification",
    "quote_approval",
]


def _load_filtered_dataset(split: str) -> Dataset:  # noqa: D401
    """Load the CRMArena-Pro split and keep only policy tasks."""
    ds = load_dataset(HF_DATASET_NAME, HF_CONFIG_NAME, split=split)
    return ds.filter(lambda x: x["task"] in POLICY_TASKS)


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser("Show CRMArena ground-truth answers")
    parser.add_argument(
        "--split",
        default="b2b",
        help="Dataset split to load (b2b | b2c | original). Default: b2b",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Number of rows to display (-1 = all)",
    )
    args = parser.parse_args()

    ds = _load_filtered_dataset(args.split)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    rows: List[Dict] = ds.to_list()
    for i, row in enumerate(rows):
        print("\n" + "=" * 80)
        print(f"Index      : {i}")
        print(f"Task name  : {row['task']}")
        print(f"Query      : {row['query']}")
        print(f"Answer (GT): {row['answer']}")


if __name__ == "__main__":
    main() 