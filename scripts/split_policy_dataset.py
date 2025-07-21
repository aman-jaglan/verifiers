"""split_policy_dataset.py
Utility to create a reproducible 70 / 10 / 20 split (train / val / test) for
**policy-oriented** tasks in the CRMArena-Pro benchmark.

The four tasks are
    • invalid_configuration_identification
    • solution_violation_identification
    • lead_qualification
    • quote_approval

The script downloads the public dataset from HuggingFace, filters the desired
categories in the **b2b** split, samples *at most* 500 rows (same cap as the
text-skill pipeline), performs a stratified split, and stores the resulting
index lists as JSON.

Run:
    python scripts/split_policy_dataset.py --output_path data/policy_indices.json

Output schema::
    {
        "train": [<int>, …],  # 350 indices
        "val":   [<int>, …],  #  50 indices
        "test":  [<int>, …]   # 100 indices
    }

This mirrors the text-skill splitter so that existing training code can be
re-used with a different JSON file.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, load_dataset

POLICY_TASKS = {
    "invalid_configuration_identification",
    "solution_violation_identification",
    "lead_qualification",
    "quote_approval",
}

HF_DATASET_NAME = "Salesforce/CRMArenaPro"
HF_CONFIG_NAME = "CRMArenaPro"

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.7, 0.1, 0.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_filtered_dataset(max_examples: int | None = 500) -> Dataset:
    """Load CRMArena-Pro B2B split and keep only policy tasks."""

    ds_b2b = load_dataset(HF_DATASET_NAME, HF_CONFIG_NAME, split="b2b")
    filtered = ds_b2b.filter(lambda x: x["task"] in POLICY_TASKS)

    if max_examples is not None and len(filtered) > max_examples:
        filtered = filtered.shuffle(seed=42).select(range(max_examples))
    return filtered


def _stratified_indices(
    dataset: Dataset,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Return stratified indices so each task keeps its proportion."""

    random.seed(seed)
    by_cat: Dict[str, List[int]] = {cat: [] for cat in POLICY_TASKS}
    for idx, task_name in enumerate(dataset["task"]):
        if task_name in by_cat:
            by_cat[task_name].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for indices in by_cat.values():
        random.shuffle(indices)
        n_total = len(indices)
        train_quota = round(train_size * n_total / len(dataset))
        val_quota = round(val_size * n_total / len(dataset))
        test_quota = n_total - train_quota - val_quota

        train_idx.extend(indices[:train_quota])
        val_idx.extend(indices[train_quota : train_quota + val_quota])
        test_idx.extend(indices[train_quota + val_quota :])

    # Final length adjustments (rounding may drift)
    leftover = set(range(len(dataset))) - set(train_idx) - set(val_idx) - set(test_idx)

    def _pad(lst: List[int], target: int):
        need = target - len(lst)
        pool = list(leftover)
        if need > 0:
            lst.extend(random.sample(pool, need))
        elif need < 0:
            random.shuffle(lst)
            del lst[target:]

    _pad(train_idx, train_size)
    _pad(val_idx, val_size)
    _pad(test_idx, test_size)

    assert len(train_idx) == train_size
    assert len(val_idx) == val_size
    assert len(test_idx) == test_size

    return {"train": train_idx, "val": val_idx, "test": test_idx}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_path", type=Path, required=True, help="Where to write the JSON index file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits.")
    parser.add_argument("--max_examples", type=int, default=500, help="Cap examples after filtering.")
    args = parser.parse_args()

    ds = _load_filtered_dataset(max_examples=args.max_examples)
    n = len(ds)
    if n < 10:
        raise ValueError("Not enough examples to create meaningful splits (need ≥10).")

    train_size = max(1, round(n * TRAIN_FRAC))
    val_size   = max(1, round(n * VAL_FRAC))
    test_size  = n - train_size - val_size

    idx_dict = _stratified_indices(ds, train_size, val_size, test_size, seed=args.seed)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fp:
        json.dump(idx_dict, fp, indent=2)
    print(f"Wrote split indices to {args.output_path} (seed={args.seed}).")


if __name__ == "__main__":
    main() 