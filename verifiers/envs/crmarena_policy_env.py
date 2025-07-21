from __future__ import annotations

"""crmarena_policy_env
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Environment wrapper for the four *policy* tasks that keeps CRMArena’s
GPT-4o-based exact-match scoring **and** augments it with Verifiers’ tool
execution / format rewards.

Compared to :class:`verifiers.envs.crmarena_env.CRMArenaEnv` it

* limits the helper-function set to the five actually used by the policy
  tasks (search_knowledge_articles, search_products, get_issues,
  get_issue_counts, respond) to shrink the action space;
* computes additional rewards via :class:`verifiers.rubrics.ToolRubric` on
  episode termination and returns the weighted sum.

Reward weights follow the user specification:
  answer-correctness  …… 1.0  (CRM evaluator)
  tool-success        …… 0.1
  format (XML)        …… 0.0
"""

from typing import Any, Dict, List, Tuple, Callable

from crm_sandbox.env.functions import (
    search_knowledge_articles,
    search_products,
    get_issues,
    get_issue_counts,
    respond,
)

from verifiers.envs.crmarena_env import CRMArenaEnv
from verifiers import XMLParser
from verifiers.rubrics import ToolRubric

# ---------------------------------------------------------------------------
# Helper: narrow tool list to policy-relevant subset
# ---------------------------------------------------------------------------
POLICY_TOOLS: List[Callable] = [
    search_knowledge_articles,
    search_products,
    get_issues,
    get_issue_counts,
    respond,
]

__all__ = ["CRMArenaPolicyEnv", "POLICY_TOOLS"]


class CRMArenaPolicyEnv(CRMArenaEnv):
    """Adapter that adds tool-success + format rewards on top of CRM scoring."""

    def __init__(
        self,
        *,
        tasks: List[Dict[str, Any]],
        tools: List[Callable] | None = None,
        task_index: int | None = None,
        tool_weight: float = 0.1,
        format_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if tools is None:
            tools = POLICY_TOOLS

        super().__init__(tasks=tasks, tools=tools, task_index=task_index, **kwargs)

        # ---------------------------------------------------------------
        # Instrument litellm.completion so we can count GPT-4o calls.
        # This is *local* to the environment instance and does not affect
        # other parts of the application.
        # ---------------------------------------------------------------
        import litellm  # import inside to avoid hard dep at module import time

        self._gpt_calls = 0

        _orig_completion = litellm.completion

        def _counting_completion(*args, **kw):  # type: ignore[override]
            self._gpt_calls += 1
            return _orig_completion(*args, **kw)

        litellm.completion = _counting_completion  # type: ignore[assignment]
        self._restore_litellm = lambda: setattr(litellm, "completion", _orig_completion)

        # Build rubric for auxiliary rewards
        self._parser = XMLParser(fields=["think", ("tool", "answer")], answer_field="answer")
        self._env_parser = XMLParser(fields=["result"])
        self._rubric = ToolRubric(parser=self._parser, env_parser=self._env_parser, tools=tools)

        # Overwrite first three weights: [ans, tool, format]
        self._rubric.reward_weights[:3] = [1.0, tool_weight, format_weight]

    # ------------------------------------------------------------------ API #
    def step(self, action: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:  # type: ignore[override]
        observation, crm_reward, done, info = super().step(action)

        # Combine rewards only at episode end; intermediate steps keep 0 reward
        if done:
            # ``info`` coming from underlying env already contains the trajectory
            trajectory: List[Dict[str, str]] = info.get("agent_actions", [])
            tool_score = self._rubric.tool_execution_reward_func(trajectory)
            fmt_score = self._parser.get_format_reward_func()(trajectory)

            final_reward = (
                1.0 * crm_reward
                + self._rubric.reward_weights[1] * tool_score
                + self._rubric.reward_weights[2] * fmt_score
            )

            info["reward_breakdown"] = {
                "crm_exact": crm_reward,
                "tool_success": tool_score,
                "format": fmt_score,
            }

            # expose how many GPT-4o (litellm) calls were made in this episode
            info["gpt4o_calls"] = self._gpt_calls
            # reset counter for next episode
            self._gpt_calls = 0
        else:
            final_reward = 0.0

        return observation, final_reward, done, info

    # ------------------------------------------------------------------ #
    def __del__(self):
        """Restore litellm.completion on garbage-collection."""
        if hasattr(self, "_restore_litellm"):
            self._restore_litellm() 