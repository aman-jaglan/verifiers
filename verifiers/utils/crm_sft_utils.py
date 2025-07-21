from __future__ import annotations

from typing import Dict, List

__all__: List[str] = [
    "build_prompt_completion",
]


def build_prompt_completion(question: str, answer: str, system_prompt: str) -> Dict[str, List[Dict[str, str]]]:
    """Create a *chat-style* prompt/completion mapping for supervised fine-tuning.

    The *Verifiers* RL environments expose a ``system_prompt`` that already
    contains textual tool descriptions.  For supervised fine-tuning (SFT) we
    want to present the model with a standard *system → user → assistant*
    sequence so it can learn to emit the correct assistant response without
    interacting with the environment.

    This helper converts a single (``question``, ``answer``) pair into a dict
    compatible with :class:`trl.SFTTrainer`, where the keys are:

    - ``prompt`` – a list of *system* and *user* messages.
    - ``completion`` – a list containing exactly one *assistant* message.

    Args:
        question: Natural-language query from the CRM task.
        answer:   Ground-truth assistant response (may include tool calls).
        system_prompt: The system prompt describing available tools and rules.

    Returns:
        A mapping with ``prompt`` and ``completion`` suitable for SFT.
    """

    prompt: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    completion: List[Dict[str, str]] = [
        {"role": "assistant", "content": answer},
    ]

    return {"prompt": prompt, "completion": completion} 