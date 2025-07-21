import asyncio
from typing import Dict, Any, List

import pytest

from verifiers.envs.crmarena_policy_env import CRMArenaPolicyEnv, POLICY_TOOLS


def test_reward_calculation_weights(monkeypatch):
    """Ensure weighted reward reflects GPT reward, tool-execution reward and format reward."""

    # Minimal single task required by CRMArenaPolicyEnv
    tasks: Dict[int, Dict[str, Any]] = {
        0: {
            "query": "Find the Account ID for Acme Corp.",
            "answer": "001XX000003GXXX",
            "task": "policy_task",
            "persona": "",
            "metadata": "",
            "reward_metric": "exact_match",
        }
    }

    # Build environment with full tool list (POLICY_TOOLS)
    env = CRMArenaPolicyEnv(tasks=tasks, tools=POLICY_TOOLS, max_turns=1)

    # Stub the GPT-4o evaluator so that _crm_reward (renamed gpt_reward_func)
    # returns a constant 1.0 regardless of input. This isolates the test from
    # external API calls and focuses purely on the weighting logic.
    monkeypatch.setattr(env._evaluator, "evaluate", lambda *_, **__: {"reward": 1.0})

    # Shortcut: grab the rubric that the trainer would use
    rubric = env.rubric

    # Craft a completion that follows the required <think> … </think> format
    completion: List[Dict[str, str]] = [
        {
            "role": "assistant",
            "content": "<think>\nSure – working on it.\n</think>\nHere you go.",
        }
    ]

    # No tools are called in this simple completion, so tool_execution_reward
    # will be zero.  format_reward_func returns 1.0 because the single assistant
    # message matches the XML pattern.
    scores = asyncio.run(
        rubric.score_rollout(
            prompt=[],
            completion=completion,
            answer="001XX000003GXXX",
            state={},
            task="policy_task",
            info={},
        )
    )

    # Locate per-function scores for clarity
    gpt_reward = scores.get("gpt_reward_func") or scores.get("_crm_reward")
    tool_exec_reward = scores.get("tool_execution_reward_func")
    format_reward = scores.get("format_reward_func")

    # Sanity checks – individual rewards make sense
    assert gpt_reward == 1.0
    assert tool_exec_reward == 0.0
    assert 0.0 <= format_reward <= 1.0

    # Weighted reward = gpt_reward*1.0 + tool_exec*0.1 + format_reward*0.1
    expected_total = 1.0 + format_reward * 0.1  # tool_exec contributes 0
    assert pytest.approx(scores["reward"], rel=1e-6) == expected_total 