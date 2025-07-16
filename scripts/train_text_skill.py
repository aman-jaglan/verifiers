"""train_text_skill.py
Launches GRPO + LoRA fine-tuning of Llama-3.1-70B on the CRMArena-Pro B2B
text-skill subset.

Usage
-----
    accelerate launch scripts/train_text_skill.py \
        --indices_json data/text_skill_indices.json \
        --config_yaml  configs/llama3_text_skill_grpo.yaml \
        --output_dir   outputs/llama3_text_skill_b2b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from accelerate import Accelerator
from datasets import Dataset, load_dataset, concatenate_datasets
from peft import LoraConfig

from verifiers.trainers import GRPOConfig, GRPOTrainer  # direct import avoids __init__ gate
from verifiers.envs.crmarena_text_env import CRMArenaTextEnv
from verifiers.utils.model_utils import get_model_and_tokenizer

# Categories and dataset loading logic must replicate the splitter exactly so
# that saved indices align with row positions.

TEXT_SKILL_CATEGORIES = {
    "knowledge_qa",
    "sales_insight_mining",
    "wrong_stage_rectification",
    "activity_priority",
    "named_entity_disambiguation",
}

HF_DATASET_NAME = "Salesforce/CRMArenaPro"
HF_CONFIG_NAME = "CRMArenaPro"
MAX_EXAMPLES = 500


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _load_filtered_dataset() -> Dataset:
    """Replicates the filtering + shuffling used by the split script."""
    ds_b2b = load_dataset(HF_DATASET_NAME, HF_CONFIG_NAME, split="b2b")
    filtered = ds_b2b.filter(lambda x: x["task"] in TEXT_SKILL_CATEGORIES)

    if len(filtered) > MAX_EXAMPLES:
        filtered = filtered.shuffle(seed=42).select(range(MAX_EXAMPLES))
    return filtered


def _rows_from_indices(indices: List[int], base_ds: Dataset) -> List[Dict]:
    subset = base_ds.select(indices)
    return [
        {
            "question": q,
            "answer": a,
            "task_name": t,
        }
        for q, a, t in zip(subset["query"], subset["answer"], subset["task"])
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Train Llama-3.1-70B on CRMArena text-skill tasks (B2B)")
    parser.add_argument("--indices_json", type=Path, required=True, help="Path to JSON produced by split_text_skill_dataset.py")
    parser.add_argument("--config_yaml", type=Path, required=True, help="GRPO + LoRA config YAML")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    # Load dataset and filter
    base_ds = _load_filtered_dataset()

    with args.indices_json.open() as fp:
        idx_dict = json.load(fp)

    rows_train = _rows_from_indices(idx_dict["train"], base_ds)
    rows_val = _rows_from_indices(idx_dict["val"], base_ds)

    dataset_train = Dataset.from_list(rows_train)
    dataset_val = Dataset.from_list(rows_val)

    # Build task mapping expected by CRMArenaTextEnv (index -> full task dict)
    tasks_by_idx = {i: base_ds[i] for i in range(len(base_ds))}

    env = CRMArenaTextEnv(tasks=tasks_by_idx, max_turns=10)

    # ----------------- model & tokenizer ------------------------------
    model_name = "meta-llama/Meta-Llama-3.1-70B"
    model, tokenizer = get_model_and_tokenizer(model_name)

    # ----------------- config ----------------------------------------
    with args.config_yaml.open() as fp:
        cfg_dict = yaml.safe_load(fp)
    grpo_cfg = GRPOConfig(output_dir=str(args.output_dir), **cfg_dict)

    # LoRA configuration (matches YAML params)
    peft_cfg = LoraConfig(**cfg_dict["peft"])

    # ----------------- trainer ---------------------------------------
    trainer = GRPOTrainer(
        model=model,
        env=env,
        args=grpo_cfg,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )

    trainer.train()
    trainer.save_model()

    # Optional: final validation accuracy
    eval_res = trainer.evaluate()
    print("Validation metrics:", eval_res)


if __name__ == "__main__":
    main() 