"""
verifiers.envs.crmarena_env
~~~~~~~~~~~~~~~~~~~~~~~~~~~
A thin adapter that exposes CRMArena's ``ToolEnv`` under the generic
``Environment`` interface expected by Verifiers trainers.

Only the two core methods – ``reset`` and ``step`` – are forwarded.  All
core reward logic (including GPT-4o-based evaluation) continues to live in
CRMArena’s code-base, ensuring benchmark-faithful scoring.  The wrapper adds
no additional behaviour; it merely satisfies the structural expectations of
Verifiers components such as ``GRPOTrainer``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Callable

# CRMArena imports -----------------------------------------------------------
from crm_sandbox.env import TOOLS as _DEFAULT_TOOLS
from crm_sandbox.env.env import ToolEnv as _CRMToolEnv


class CRMArenaEnv:  # pylint: disable=too-few-public-methods
    """Adapter that makes CRMArena's :class:`ToolEnv` usable inside Verifiers.

    Parameters
    ----------
    tasks:
        A list of task dictionaries as expected by CRMArena (keys include
        ``query``, ``answer``, ``reward_metric``, etc.).
    tools:
        Optional explicit list of helper-function callables.  If *None*, the
        canonical ``crm_sandbox.env.functions.TOOLS`` list is used.
    task_index:
        If given, the underlying environment will start at this task; otherwise
        CRMArena’s own random selection logic applies.
    """

    # --------------------------------------------------------------------- #
    def __init__(
        self,
        *,
        tasks: List[Dict[str, Any]],
        tools: List[Callable] | None = None,
        task_index: int | None = None,
    ) -> None:
        if tools is None:
            tools = _DEFAULT_TOOLS

        # CRMArena's ToolEnv requires ``org_type='original'`` so that it uses
        # its internally configured Salesforce sandbox credentials.
        self._env: _CRMToolEnv = _CRMToolEnv(
            tools=tools,
            tasks=tasks,
            task_index=task_index,
            org_type="original",
        )

    # ------------------------------------------------------------------ API #
    def reset(self, task_index: int = 0) -> Tuple[str, Dict[str, Any]]:
        """Reset the underlying environment and return its initial observation.

        Returns
        -------
        observation:
            The user query string for the chosen task.
        metadata:
            Optional metadata dictionary provided by CRMArena.
        """

        return self._env.reset(task_index)

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Forward *action* to the underlying CRMArena environment."""

        return self._env.step(action) 