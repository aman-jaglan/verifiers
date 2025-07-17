"""Tests for `CRMArenaEnv` wrapper.

We monkey-patch :class:`crm_sandbox.env.connect_sandbox.SalesforceConnector`
to avoid real network calls and Salesforce credentials.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from verifiers.envs.crmarena_env import CRMArenaEnv


def _dummy_tasks():
    """Return a minimal single-task list compatible with CRMArena."""

    return [
        {
            "query": "Dummy question?",
            "answer": ["None"],
            "reward_metric": "exact_match",
            "task": "knowledge_qa",
            "persona": "",
        }
    ]


def test_reset_and_step():
    """CRMArenaEnv should proxy `reset` and `step` without errors."""

    with patch("crm_sandbox.env.connect_sandbox.SalesforceConnector") as mock_sf:
        # Prevent actual Salesforce calls – all we need is that the connector
        # instance exists and its .run_query returns a tuple in the right shape.
        mock_instance = MagicMock()
        mock_instance.run_query.return_value = ([], 1)
        mock_sf.return_value = mock_instance

        env = CRMArenaEnv(tasks=_dummy_tasks(), tools=[])

        obs, meta = env.reset(0)
        assert isinstance(obs, str)
        # CRMArena fills empty metadata with "", not None
        assert meta in ("", {})

        action = {"name": "respond", "arguments": {"content": "None"}}
        _, reward, done, info = env.step(action)

        assert done is True
        assert isinstance(reward, (int, float))
        assert "agent_actions" in info 