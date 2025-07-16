"""crmarena_text_env.py
Environment adapter that lets Verifiers' GRPOTrainer interact with the
Salesforce **CRMArena-Pro** *text-skill* tasks through the existing CRM
`ToolEnv` while preserving reward semantics.

Key points
----------
1.  Uses the exact helper-functions (`crm_sandbox.env.TOOLS`) and reward
    pipeline (`Evaluator.evaluate`) already defined by the benchmark.
2.  Presents a *chat* multi-turn interface so Verifiers can roll out
    conversations and obtain per-episode rewards.
3.  No reward-weight override – keeps default
    ``[1.0, 0.2, 0.2]`` (answer, tool-success, format).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from datasets import Dataset

# CRM sandbox imports (kept optional for static analysers)
from CRMArena.crm_sandbox.env import TOOLS  # type: ignore
from CRMArena.crm_sandbox.env.env import ToolEnv as CRMSandboxToolEnv  # type: ignore

from verifiers import MultiTurnEnv, ToolRubric, XMLParser, Message, Messages, State  # noqa: E501
from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE


class CRMArenaTextEnv(MultiTurnEnv):
    """Verifiers environment for the five CRMArena-Pro text-skill categories."""

    def __init__(
        self,
        tasks: Dict[int, Dict[str, Any]],
        org_type: str = "original",
        max_turns: int = 10,
        seed: int = 42,
        **kwargs,
    ) -> None:
        # ------------------------------------------------------------------
        # 1. Wrap the official CRM ToolEnv (executes functions & rewards)
        # ------------------------------------------------------------------
        self._crm_env = CRMSandboxToolEnv(tools=TOOLS, tasks=tasks, org_type=org_type)

        # ------------------------------------------------------------------
        # 2. Build Verifiers dataset (needed by base Environment class)
        # ------------------------------------------------------------------
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
        # 3. Parser & rubric (inherits default reward weights)
        # ------------------------------------------------------------------
        self.parser = XMLParser(fields=["think", ("tool", "answer")], answer_field="answer")
        self.env_parser = XMLParser(fields=["result"])
        rubric = ToolRubric(parser=self.parser, env_parser=self.env_parser, tools=TOOLS)

        # ------------------------------------------------------------------
        # 4. Build system prompt (textual schemas for non-native tool calls)
        # ------------------------------------------------------------------
        tool_descriptions = self._crm_env.format_tool_descriptions(self._crm_env.tool_schemas)  # type: ignore[attr-defined]
        formatted_prompt = DEFAULT_TOOL_PROMPT_TEMPLATE.format(tool_descriptions=tool_descriptions)

        super().__init__(
            system_prompt=formatted_prompt,
            parser=self.parser,
            rubric=rubric,
            dataset=dataset,
            message_type="chat",
            max_turns=max_turns,
            seed=seed,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # MultiTurnEnv required overrides
    # ------------------------------------------------------------------
    def is_completed(self, messages: Messages, state: State, **kwargs: Any) -> bool:  # noqa: D401
        """Episode ends when <answer> is present or max_turns reached."""
        return self.parser.parse_answer(messages) is not None

    def env_response(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> Tuple[Message, State]:
        """Execute the requested tool and return its output wrapped as user."""
        try:
            parsed = self.parser.parse(messages[-1]["content"])
            if hasattr(parsed, "tool") and parsed.tool is not None:
                result = self._crm_env.call_tool(parsed.tool)
                formatted = self.env_parser.format(result=result)
                return {"role": "user", "content": formatted}, state
        except Exception:
            # fall through to error message
            pass
        return {
            "role": "user",
            "content": "Error: Tool command not found or invalid XML format.",
        }, state

    # ------------------------------------------------------------------
    # Expose reward funcs/weights directly from rubric for trainer
    # ------------------------------------------------------------------
    def get_reward_funcs(self, **kwargs):  # noqa: D401
        return self.rubric.get_reward_funcs()

    def get_reward_weights(self, **kwargs):  # noqa: D401
        return self.rubric.get_reward_weights() 