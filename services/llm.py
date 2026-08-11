"""Provider-agnostic chat-model wrapper. Structured output is produced via
LangChain's tool-calling-backed `with_structured_output`, used for
GroundedAnswer, the scope-classification guardrail, and the compliance
checklist schemas — this is what every call site (qa.py, guardrails.py,
checklist.py, ragas_harness.py) depends on, not a specific provider's SDK.

Default provider is Google Gemini's free tier (see ADR-0004) — free tiers are
rate-limited, so calls are wrapped with a retry/backoff for 429s rather than
failing a whole eval run on a transient rate-limit hit."""

import re
from functools import lru_cache
from typing import Literal, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt

from services.config import get_settings

ModelTier = Literal["cheap", "quality"]
T = TypeVar("T", bound=BaseModel)


def _is_rate_limit_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "429" in message or "rate limit" in message or "resource_exhausted" in message


# Google's 429 response embeds its own suggested wait, e.g. "retry_delay {\n
# seconds: 34\n}". LangChain's own internal retry (a separate layer below
# this one, active by default) ignores that and hammers again every ~2
# seconds regardless — which just re-triggers the same per-minute quota
# error dozens of times instead of actually waiting it out. This wait
# strategy reads Google's own number when present; ChatGoogleGenerativeAI is
# constructed with max_retries=1 below specifically so that number reaches
# here instead of being absorbed by LangChain's internal retry first.
_RETRY_DELAY_RE = re.compile(r"retry_delay\s*\{\s*seconds:\s*(\d+)")


def _wait_for_rate_limit(retry_state: RetryCallState) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is not None:
        match = _RETRY_DELAY_RE.search(str(exc))
        if match:
            return float(match.group(1)) + 2  # small buffer past the server's own estimate
    return min(5 * (2 ** (retry_state.attempt_number - 1)), 65)


_retry_on_rate_limit = retry(
    retry=retry_if_exception(_is_rate_limit_error),
    stop=stop_after_attempt(6),
    wait=_wait_for_rate_limit,
    reraise=True,
)


@lru_cache
def _get_chat_model(tier: ModelTier) -> BaseChatModel:
    settings = get_settings()
    model_name = settings.llm_model_cheap if tier == "cheap" else settings.llm_model_quality

    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.gemini_api_key,
            temperature=0,
            max_retries=1,  # disable LangChain's own internal retry — see _wait_for_rate_limit above
        )

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name, api_key=settings.anthropic_api_key, temperature=0)

    raise ValueError(f"Unsupported LLM_PROVIDER={settings.llm_provider!r} (expected 'gemini' or 'anthropic')")


def get_chat_model(tier: ModelTier = "cheap") -> BaseChatModel:
    """Public accessor for callers (e.g. the RAGAS judge) that need the raw
    chat model rather than a structured-output call."""
    return _get_chat_model(tier)


@_retry_on_rate_limit
def generate_structured(prompt: str, schema: type[T], tier: ModelTier = "cheap") -> T:
    model = _get_chat_model(tier).with_structured_output(schema)
    result = model.invoke(prompt)
    return result  # type: ignore[return-value]


@_retry_on_rate_limit
def generate_text(prompt: str, tier: ModelTier = "cheap") -> str:
    model = _get_chat_model(tier)
    return model.invoke(prompt).content
