from __future__ import annotations

"""crmarena_policy_env
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Environment wrapper for the four *policy* tasks that keeps CRMArena’s
GPT-4o-based exact-match scoring **and** augments it with Verifiers’ tool
execution / format rewards.

It mirrors the structure of :class:`verifiers.envs.crmarena_text_env.CRMArenaTextEnv`
but narrows the helper-function set to the subset actually used by the policy
benchmarks and overrides the reward weights as requested by the user:

    answer-correctness … 1.0 (CRM evaluator)
    tool-success       … 0.1
    format (XML)       … 0.0
"""

from typing import Any, Dict, List, Callable, Tuple, Optional

from datasets import Dataset

# CRM sandbox imports -------------------------------------------------------
from crm_sandbox.env.functions import (
    search_knowledge_articles,
    search_products,
    get_issues,
    get_issue_counts,
    respond,
)
from crm_sandbox.env.env import ToolEnv as CRMSandboxToolEnv  # type: ignore

from .multiturn_env import MultiTurnEnv
from ..rubrics.tool_rubric import ToolRubric
from ..parsers.xml_parser import XMLParser
from verifiers.envs.tool_env import format_tool_descriptions  # human-readable schemas
from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE

import inspect
import crm_sandbox.env.functions as crm_fn_mod

__all__ = [
    "CRMArenaPolicyEnv",
    "POLICY_TOOLS",
    "ALL_CRM_TOOLS",
]

# ---------------------------------------------------------------------------
# Collect every CRM helper that declares an OpenAI-style JSON schema via
# the ``__info__`` attribute.  This ensures the policy is aware of *all*
# available tools during training without having to maintain a manual list.
# ---------------------------------------------------------------------------

def _collect_all_tools() -> list[Callable]:
    """Return every callable in ``crm_sandbox.env.functions`` that defines
    a JSON schema ("__info__" attribute)."""

    return [
        fn
        for _, fn in inspect.getmembers(crm_fn_mod, inspect.isfunction)
        if hasattr(fn, "__info__")
    ]


# Full exhaustive set of CRM tools
ALL_CRM_TOOLS: list[Callable] = _collect_all_tools()

# Back-compatibility alias used by external scripts (e.g. train_text_skill.py)
# Previously this contained a five-tool subset; it now points to the full set.
POLICY_TOOLS: list[Callable] = ALL_CRM_TOOLS

# ---------------------------------------------------------------------------
# Helper to convert CRMArena function-call schema to Verifiers flat schema
# ---------------------------------------------------------------------------

def _convert_crm_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a CRM __info__ dict to the flat format expected by Verifiers."""

    if "function" not in schema:  # already flattened
        return schema

    func = schema["function"]

    # Extract argument list
    arg_specs: Dict[str, Any] = {}
    params = func.get("parameters", {})
    for arg_name, arg_info in params.get("properties", {}).items():
        arg_specs[arg_name] = {
            "type": arg_info.get("type", "any"),
            "description": arg_info.get("description", ""),
        }

    return {
        "name": func.get("name", ""),
        "description": func.get("description", ""),
        "args": arg_specs,
        "returns": func.get("returns", {}).get("description", ""),
        "examples": [],
    }


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class CRMArenaPolicyEnv(MultiTurnEnv):
    """Verifiers environment for the four CRMArena-Pro *policy* tasks."""

    def __init__(
        self,
        *,
        tasks: Dict[int, Dict[str, Any]],
        tools: Optional[List[Callable]] = None,
        max_turns: int = 5,
        dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        seed: int = 42,
        tool_weight: float = 0.1,
        format_weight: float = 0.1,
        **kwargs: Any,
    ) -> None:
        if tools is None:
            tools = POLICY_TOOLS

        # ------------------------------------------------------------------
        # 1. Wrap the official CRM ToolEnv (executes functions & rewards)
        # ------------------------------------------------------------------
        self._crm_env = CRMSandboxToolEnv(tools=tools, tasks=tasks, org_type="original")

        # ------------------------------------------------------------------
        # 2. Build / adopt dataset(s)
        # ------------------------------------------------------------------
        if dataset is None:
            rows = [
                {
                    "question": task["query"],
                    "answer": task["answer"],
                    "task_name": task["task"],
                }
                for task in tasks.values()
            ]
            dataset = Dataset.from_list(rows)

        # ------------------------------------------------------------------
        # 3. Parser & rubric – override weights
        # ------------------------------------------------------------------
        self.parser = XMLParser(fields=["think", ("tool", "answer")], answer_field="answer")
        self.env_parser = XMLParser(fields=["result"])
        rubric = ToolRubric(parser=self.parser, env_parser=self.env_parser, tools=tools)
        rubric.reward_weights[:3] = [1.0, tool_weight, format_weight]

        # ------------------------------------------------------------------
        # 5. Replace default correct-answer reward with CRM Evaluator (GPT-4o)
        # ------------------------------------------------------------------
        from crm_sandbox.env.env import Evaluator  # local import to avoid heavy deps at module load time

        self._evaluator = Evaluator(model="gpt-4o-2024-08-06", provider="openai")
        # Build quick lookup -> task info so we can fetch reward_metric for fuzzy/exact etc.
        self._answer_lookup = {}
        def _norm(ans):
            # Lists are unhashable; convert to tuple. None -> "None" to keep hashable.
            if isinstance(ans, list):
                return tuple(ans)
            if ans is None:
                return ("None",)
            return (ans,)

        for _t in tasks.values():
            key = _norm(_t["answer"])
            if key not in self._answer_lookup:
                self._answer_lookup[key] = _t

        def _crm_reward(completion, answer, **kwargs):  # type: ignore
            """Use CRMArena's Evaluator (GPT-4o) to grade the final <answer>."""
            # Extract assistant <answer> text
            proposed_ans = ""
            try:
                parsed = self.parser.parse_answer(completion)
                if parsed is not None:
                    proposed_ans = str(parsed)
            except Exception:
                proposed_ans = ""

            task_info = self._answer_lookup.get(_norm(answer), {})
            reward_metric = task_info.get("reward_metric", "exact_match")
            task_name = task_info.get("task", "policy_task")

            res = self._evaluator.evaluate(
                proposed_ans,
                [answer],
                reward_metric,
                task_name,
                [m["content"] for m in completion if isinstance(m, dict)],
            )
            return res["reward"]

        _crm_reward.__name__ = "gpt_reward_func"
        rubric.reward_funcs[0] = _crm_reward

        # ------------------------------------------------------------------
        # 4. Build system prompt with *only* policy-tools described
        # ------------------------------------------------------------------
        tool_schemas = [_convert_crm_schema(s) for s in self._crm_env.tools_info]
        tool_descriptions = format_tool_descriptions(tool_schemas)
        formatted_prompt = DEFAULT_TOOL_PROMPT_TEMPLATE.format(tool_descriptions=tool_descriptions)

        super().__init__(
            system_prompt=formatted_prompt,
            parser=self.parser,
            rubric=rubric,
            dataset=dataset,
            eval_dataset=eval_dataset,
            message_type="chat",
            max_turns=max_turns,
            seed=seed,
            **kwargs,
        )

    # ------------------------------------------------------------------ MultiTurnEnv overrides
    def is_completed(self, messages, state, **kwargs):  # noqa: D401
        """Episode ends when <answer> tag appears or max_turns reached."""
        return self.parser.parse_answer(messages) is not None

    def env_response(self, messages, state, **kwargs):
        """Execute the selected tool via CRM ToolEnv and wrap result as user."""
        try:
            parsed = self.parser.parse(messages[-1]["content"])
            if hasattr(parsed, "tool") and parsed.tool is not None:
                result = self._crm_env.call_tool(parsed.tool)
                formatted = self.env_parser.format(result=result)
                return {"role": "user", "content": formatted}, state
        except Exception:
            pass  # fall through to error message
        return {"role": "user", "content": "Error: Tool command not found or invalid XML format."}, state

    # -------------- dataset helpers (used by GRPOTrainer) -----------------
    def get_dataset(self, *args, **kwargs):  # noqa: D401
        return super().get_dataset(*args, **kwargs)

    def get_eval_dataset(self, *args, **kwargs):  # noqa: D401
        return super().get_eval_dataset(*args, **kwargs) 