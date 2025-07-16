"""split_text_skill_dataset.py
Utility to create a reproducible 70 / 10 / 20 split (train / val / test) for the
five *text-skill* categories of the CRMArena-Pro benchmark.

The script downloads the public dataset from HuggingFace, filters the desired
categories across the **b2b** and **b2c** splits, samples *at most* 500 total
examples, performs a stratified split, and stores the resulting index lists as
JSON.

Run:
    python scripts/split_text_skill_dataset.py --output_path data/text_skill_indices.json

The output schema is::
    {
        "train": [<int>, …],  # 350 indices
        "val":   [<int>, …],  #  50 indices
        "test":  [<int>, …]   # 100 indices
    }

Notes
-----
1. We keep the indices relative to the concatenated Dataset so that the same
   indices can be used later via `Dataset.select(indices)`.
2. The exact rows are fixed by `--seed` (default 42) to guarantee that all
   collaborators work on the same split.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, concatenate_datasets, load_dataset

TEXT_SKILL_CATEGORIES = {
    "knowledge_qa",
    "sales_insight_mining",
    "wrong_stage_rectification",
    "activity_priority",
    "named_entity_disambiguation",
}

HF_DATASET_NAME = "Salesforce/CRMArenaPro"
HF_CONFIG_NAME = "CRMArenaPro"

DEFAULT_TRAIN, DEFAULT_VAL, DEFAULT_TEST = 350, 50, 100


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_filtered_dataset(max_examples: int | None = 500) -> Dataset:
    """Load CRMArena-Pro and retain only text-skill rows.

    Args:
        max_examples: Maximum number of rows to keep *after* filtering. If
            ``None`` all rows are returned.

    Returns:
        A concatenated HuggingFace ``Dataset`` containing only the requested
        categories from the *b2b* and *b2c* splits.
    """
    # Only the B2B organisation split is used for training/validation/testing.
    ds_b2b = load_dataset(HF_DATASET_NAME, HF_CONFIG_NAME, split="b2b")

    combined: Dataset = ds_b2b.filter(lambda x: x["task"] in TEXT_SKILL_CATEGORIES)

    if max_examples is not None and len(combined) > max_examples:
        combined = combined.shuffle(seed=42).select(range(max_examples))
    return combined


def _stratified_indices(
    dataset: Dataset,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Produce stratified indices for train / val / test.

    Stratification is performed on the **task** column so each category preserves
    the overall distribution.
    """

    random.seed(seed)
    by_cat: Dict[str, List[int]] = {cat: [] for cat in TEXT_SKILL_CATEGORIES}
    for idx, task_name in enumerate(dataset["task"]):
        if task_name in by_cat:
            by_cat[task_name].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for cat, indices in by_cat.items():
        random.shuffle(indices)
        n_total = len(indices)
        # Proportional allocation based on requested absolute sizes
        train_quota = round(train_size * n_total / len(dataset))
        val_quota = round(val_size * n_total / len(dataset))
        test_quota = n_total - train_quota - val_quota

        train_idx.extend(indices[:train_quota])
        val_idx.extend(indices[train_quota : train_quota + val_quota])
        test_idx.extend(indices[train_quota + val_quota :])

    # Sanity: adjust if rounding deviated
    def _pad(lst: List[int], target: int, pool: List[int]):
        """Add random indices from *pool* until |lst| == target."""
        needed = target - len(lst)
        if needed > 0:
            lst.extend(random.sample(pool, needed))
        elif needed < 0:
            random.shuffle(lst)
            del lst[target:]

    leftover = set(range(len(dataset))) - set(train_idx) - set(val_idx) - set(test_idx)
    _pad(train_idx, train_size, list(leftover))
    _pad(val_idx, val_size, list(leftover))
    _pad(test_idx, test_size, list(leftover))

    assert len(train_idx) == train_size, "train size mismatch"
    assert len(val_idx) == val_size, "val size mismatch"
    assert len(test_idx) == test_size, "test size mismatch"

    return {"train": train_idx, "val": val_idx, "test": test_idx}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    """Create stratified splits and write them to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_path", type=Path, required=True, help="JSON file to save indices to.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatability.")
    parser.add_argument("--max_examples", type=int, default=500, help="Maximum examples after filtering (<=500).")
    args = parser.parse_args()

    ds = _load_filtered_dataset(max_examples=args.max_examples)
    if len(ds) < DEFAULT_TRAIN + DEFAULT_VAL + DEFAULT_TEST:
        raise ValueError("Filtered dataset has fewer rows than required split sizes.")

    idx_dict = _stratified_indices(ds, DEFAULT_TRAIN, DEFAULT_VAL, DEFAULT_TEST, seed=args.seed)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as fp:
        json.dump(idx_dict, fp, indent=2)
    print(f"Wrote split indices to {args.output_path} (seed={args.seed}).")


if __name__ == "__main__":
    main() 