from __future__ import annotations

import time
import uuid
from typing import Any, List, Optional

import httpx
from openai.types.chat.chat_completion import ChatCompletion, Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from transformers import PreTrainedTokenizerBase


class _ChatCompletionsEndpoint:
    """Implements `client.chat.completions.create` against TRL's `/generate/` route."""

    def __init__(self, wrapper: "TRLVLLMClient") -> None:  # noqa: D401
        self._wrapper = wrapper

    async def create(
        self, *, model: str, messages: List[dict[str, str]], **sampling_args: Any
    ) -> ChatCompletion:  # noqa: D401
        # Concatenate contents naïvely – system/user separation is already in template
        prompt_str = "".join(m["content"] for m in messages)

        payload = {"prompts": [prompt_str], **sampling_args}

        async with httpx.AsyncClient(base_url=self._wrapper.base_url) as client:
            resp = await client.post("/generate/", json=payload, timeout=None)
            resp.raise_for_status()
            completion_ids: List[int] = resp.json()["completion_ids"][0]  # first prompt

        # Decode tokens to text when a tokenizer is supplied; otherwise leave blank.
        if self._wrapper.tokenizer is not None:
            # decode without special tokens to ensure round-trippable conversation text
            decoded_text: str = self._wrapper.tokenizer.decode(
                completion_ids, skip_special_tokens=True
            )
        else:
            decoded_text = ""

        # Build ChatCompletionTokenLogprob list
        logprob_tokens: List[ChatCompletionTokenLogprob] = [
            ChatCompletionTokenLogprob(
                token=f"id:{tid}", logprob=0.0, top_logprobs=[]  # type: ignore[arg-type]
            )
            for tid in completion_ids
        ]

        logprobs = ChoiceLogprobs(content=logprob_tokens)

        message = ChatCompletionMessage(role="assistant", content=decoded_text)

        choice = Choice(
            finish_reason="stop",  # best guess; vLLM stops at max_tokens or EOS
            index=0,
            logprobs=logprobs,
            message=message,
        )

        chat_completion = ChatCompletion(
            id=str(uuid.uuid4()),
            choices=[choice],
            created=int(time.time()),
            model=model,
            object="chat.completion",
        )

        return chat_completion


class _Chat:
    def __init__(self, wrapper: "TRLVLLMClient") -> None:  # noqa: D401
        self.completions = _ChatCompletionsEndpoint(wrapper)


class TRLVLLMClient:
    """Drop-in substitute for `openai.AsyncOpenAI` that targets TRL vLLM server."""

    def __init__(
        self, base_url: str, tokenizer: Optional[PreTrainedTokenizerBase] = None
    ) -> None:  # noqa: D401
        self.base_url = base_url.rstrip("/")
        self.tokenizer = tokenizer
        self.chat = _Chat(self) 