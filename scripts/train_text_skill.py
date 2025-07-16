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
from datasets import Dataset
from peft import LoraConfig

from verifiers import GRPOConfig, GRPOTrainer, CRMArenaTextEnv  # type: ignore[attr-defined]
from verifiers.utils.model_utils import get_model_and_tokenizer

# CRM assets
from CRMArena.crm_sandbox.data.assets import TASKS_B2B  # type: ignore

TEXT_SKILL_CATEGORIES = {
    "knowledge_qa",
    "sales_insight_mining",
    "wrong_stage_rectification",
    "activity_priority",
    "named_entity_disambiguation",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _load_tasks_by_idx() -> Dict[int, Dict]:
    """Load B2B tasks filtered to text-skill categories keyed by `idx`."""
    return {
        task["idx"]: task for task in TASKS_B2B if task["task"] in TEXT_SKILL_CATEGORIES
    }


def _rows_from_indices(indices: List[int], tasks_by_idx: Dict[int, Dict]) -> List[Dict]:
    rows = []
    for idx in indices:
        t = tasks_by_idx[idx]
        rows.append({
            "question": t["query"],
            "answer": t["answer"],
            "task_name": t["task"],
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Train Llama-3.1-70B on CRMArena text-skill tasks (B2B)")
    parser.add_argument("--indices_json", type=Path, required=True, help="Path to JSON produced by split_text_skill_dataset.py")
    parser.add_argument("--config_yaml", type=Path, required=True, help="GRPO + LoRA config YAML")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    tasks_by_idx = _load_tasks_by_idx()

    with args.indices_json.open() as fp:
        idx_dict = json.load(fp)

    rows_train = _rows_from_indices(idx_dict["train"], tasks_by_idx)
    rows_val = _rows_from_indices(idx_dict["val"], tasks_by_idx)

    # Build Environment (train dataset; val dataset provided separately)
    dataset_train = Dataset.from_list(rows_train)
    dataset_val = Dataset.from_list(rows_val)
    env = CRMArenaTextEnv(tasks=tasks_by_idx, max_turns=10)
    # Override datasets prepared in __init__
    env.dataset = dataset_train
    env.eval_dataset = dataset_val

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