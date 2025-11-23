from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import json
import requests
import streamlit as st

from db import increment_rate_counter
from utils import DEFAULT_SYSTEM_PROMPT, emit_chunks

logger = logging.getLogger(__name__)
API_TIMEOUT = 60


class RateLimitError(Exception):
    """Raised when a user exceeds the allowed call volume."""


@dataclass
class ProviderConfig:
    key: str
    label: str
    provider: str
    model: str
    secret: str
    handler: Callable[..., str]
    supports_stream: bool = False


class InMemoryRateLimiter:
    """Simple in-memory limiter to guard against bursts per session."""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self._cache: Dict[str, Dict[str, float]] = {}

    def check(self, session_id: str) -> Tuple[bool, int]:
        now = time.time()
        entry = self._cache.get(session_id, {"count": 0, "reset": now + self.window})
        if now >= entry["reset"]:
            entry = {"count": 0, "reset": now + self.window}
        if entry["count"] >= self.limit:
            remaining = int(entry["reset"] - now)
            return False, max(remaining, 0)
        entry["count"] += 1
        self._cache[session_id] = entry
        remaining_calls = self.limit - entry["count"]
        return True, remaining_calls


burst_limiter = InMemoryRateLimiter(limit=8, window_seconds=60)


def _secret(key: str) -> Optional[str]:
    """Fetch a secret without raising, returning None if missing."""
    try:
        return st.secrets[key]
    except Exception:
        return None


def _with_system_prompt(messages: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Prepend a system prompt unless user supplied one."""
    history = list(messages)
    if history and history[0].get("role") == "system":
        return history
    return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, *history]


def emit_or_collect(text: str, stream: bool, on_token) -> str:
    """Utility to support faux streaming."""
    if stream and on_token:
        emit_chunks(text, on_token)
    return text


def call_openai(
    messages: List[Dict[str, str]],
    *,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    stream: bool = False,
    on_token=None,
) -> str:
    api_key = _secret("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI model unavailable — please add OPENAI_API_KEY to Streamlit secrets."
    try:
        from openai import OpenAI
    except ImportError:
        return "OpenAI SDK is missing. Run `pip install openai`."

    client = OpenAI(api_key=api_key)
    prepared = _with_system_prompt(messages)

    try:
        if stream and on_token:
            stream_resp = client.chat.completions.create(
                model=model,
                messages=prepared,
                temperature=temperature,
                stream=True,
                timeout=API_TIMEOUT,
            )
            collected: List[str] = []
            for chunk in stream_resp:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    collected.append(delta)
                    on_token(delta)
            return "".join(collected)

        resp = client.chat.completions.create(
            model=model,
            messages=prepared,
            temperature=temperature,
            timeout=API_TIMEOUT,
        )
        text = resp.choices[0].message.content or ""
        return emit_or_collect(text, stream, on_token)
    except Exception as exc:
        logger.exception("OpenAI call failed: %s", exc)
        return "OpenAI provider error — try again in a moment."


def call_anthropic(
    messages: List[Dict[str, str]],
    *,
    model: str = "claude-sonnet-4-5",
    temperature: float = 0.2,
    stream: bool = False,
    on_token=None,
) -> str:
    api_key = _secret("ANTHROPIC_API_KEY")
    if not api_key:
        return "Anthropic model unavailable — set ANTHROPIC_API_KEY in secrets."
    try:
        import anthropic
    except ImportError:
        return "Anthropic SDK missing. Install `anthropic`."

    client = anthropic.Anthropic(api_key=api_key)
    prepared = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if msg["role"] in {"user", "assistant"}
    ]

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=prepared,
        )
        text_chunks: List[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                text_chunks.append(block.text)

        text = "\n\n".join(text_chunks) if text_chunks else "Claude returned no text blocks."
        return emit_or_collect(text, stream, on_token)
    except Exception as exc:
        logger.exception("Anthropic call failed: %s", exc)
        return "Anthropic provider error — try again later."


def call_gemini(
    messages: List[Dict[str, str]],
    *,
    model: str = "gemini-1.5-pro-latest",
    temperature: float = 0.2,
    stream: bool = False,
    on_token=None,
) -> str:
    api_key = _secret("GEMINI_API_KEY")
    if not api_key:
        return "Gemini model unavailable — add GEMINI_API_KEY to secrets."
    try:
        import google.generativeai as genai
    except ImportError:
        return "Google Generative AI SDK missing. Install `google-generativeai`."

    genai.configure(api_key=api_key)
    prompt = "\n".join(f"{msg['role'].title()}: {msg['content']}" for msg in messages)
    try:
        model_client = genai.GenerativeModel(f"models/{model}")
        response = model_client.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 8192},
            request_options={"timeout": API_TIMEOUT},
        )

        candidates = getattr(response, "candidates", [])
        if not candidates:
            finish_reason = getattr(response, "prompt_feedback", None)
            return f"Gemini: No candidates returned. Feedback: {finish_reason}"

        parts: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                snippet = getattr(part, "text", None)
                if snippet:
                    parts.append(snippet.strip())

        text = "\n\n".join(parts) if parts else f"Gemini returned empty response. Candidates: {len(candidates)}"
        return emit_or_collect(text, stream, on_token)
    except Exception as exc:
        logger.exception("Gemini call failed: %s", exc)
        return "Gemini provider error — please retry."


def call_qwen(
    messages: List[Dict[str, str]],
    *,
    model: str = "qwen-plus",
    temperature: float = 0.2,
    stream: bool = False,
    on_token=None,
) -> str:
    api_key = _secret("DASHSCOPE_API_KEY")
    if not api_key:
        return "Qwen model unavailable — add DASHSCOPE_API_KEY to secrets."
    try:
        from openai import OpenAI
    except ImportError:
        return "OpenAI SDK required for Qwen compatible calls."

    base_url = st.secrets.get("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)
    prepared = _with_system_prompt(messages)
    try:
        if stream and on_token:
            stream_resp = client.chat.completions.create(
                model=model,
                messages=prepared,
                stream=True,
                temperature=temperature,
                timeout=API_TIMEOUT,
            )
            collected: List[str] = []
            for chunk in stream_resp:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    collected.append(delta)
                    on_token(delta)
            return "".join(collected)

        resp = client.chat.completions.create(
            model=model,
            messages=prepared,
            temperature=temperature,
            max_tokens=512,
            timeout=API_TIMEOUT,
        )

        choices = getattr(resp, "choices", None) or []
        if not choices:
            return "Qwen returned no choices."

        message = getattr(choices[0], "message", None)
        content = ""
        if isinstance(message, dict):
            content = message.get("content") or ""
        else:
            content = getattr(message, "content", "") or ""

        content = content.strip()
        text = content or "Qwen returned an empty response."
        return emit_or_collect(text, stream, on_token)
    except Exception as exc:
        logger.exception("Qwen call failed: %s", exc)
        return "Qwen provider error — try again."


def call_groq(
    messages: List[Dict[str, str]],
    *,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    stream: bool = False,
    on_token=None,
) -> str:
    api_key = _secret("GROQ_API_KEY")
    if not api_key:
        return "Groq model unavailable — add GROQ_API_KEY to secrets."
    try:
        from groq import Groq
    except ImportError:
        return "Groq SDK missing. Install `groq`."

    client = Groq(api_key=api_key)
    prepared = _with_system_prompt(messages)
    try:
        if stream and on_token:
            stream_resp = client.chat.completions.create(
                model=model,
                messages=prepared,
                stream=True,
                temperature=temperature,
            )
            collected: List[str] = []
            for chunk in stream_resp:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    collected.append(delta)
                    on_token(delta)
            return "".join(collected)

        completion = client.chat.completions.create(
            model=model,
            messages=prepared,
            temperature=temperature,
            max_tokens=512,
        )

        choices = getattr(completion, "choices", None) or []
        if not choices:
            return "Groq returned no choices."

        message = getattr(choices[0], "message", None)
        content = ""
        if isinstance(message, dict):
            content = message.get("content") or ""
        else:
            content = getattr(message, "content", "") or ""

        content = content.strip()
        text = content or "Groq returned an empty response."
        return emit_or_collect(text, stream, on_token)
    except Exception as exc:
        logger.exception("Groq call failed: %s", exc)
        return "Groq provider error — try later."


def call_deepseek(
    messages: List[Dict[str, str]],
    *,
    model: str = "deepseek-reasoner",
    temperature: float = 0.2,
    stream: bool = False,
    on_token=None,
) -> str:
    api_key = _secret("DEEPSEEK_API_KEY")
    if not api_key:
        return "DeepSeek model unavailable — add DEEPSEEK_API_KEY."
    payload = {
        "model": model,
        "messages": _with_system_prompt(messages),
        "temperature": temperature,
        "max_tokens": 512,
        "stream": stream and bool(on_token),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            json=payload,
            headers=headers,
            timeout=API_TIMEOUT,
            stream=payload["stream"],
        )
        response.raise_for_status()
        if payload["stream"]:
            collected: List[str] = []
            for line in response.iter_lines():
                if not line or line == b"data: [DONE]":
                    continue
                if not line.startswith(b"data: "):
                    continue
                chunk = line.replace(b"data: ", b"", 1)
                try:
                    data = chunk.decode("utf-8")
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except Exception:
                    delta = None
                if delta:
                    collected.append(delta)
                    if on_token:
                        on_token(delta)
            return "".join(collected)
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return emit_or_collect(content, stream, on_token)
    except Exception as exc:
        logger.exception("DeepSeek call failed: %s", exc)
        return "DeepSeek provider error — try again."


MODEL_REGISTRY: Dict[str, ProviderConfig] = {
    "openai-gpt-4o": ProviderConfig(
        key="openai-gpt-4o",
        label="OpenAI · GPT-4o",
        provider="openai",
        model="gpt-4o",
        secret="OPENAI_API_KEY",
        handler=call_openai,
        supports_stream=True,
    ),
    "anthropic-claude": ProviderConfig(
        key="anthropic-claude",
        label="Anthropic · Claude Sonnet 4.5",
        provider="anthropic",
        model="claude-sonnet-4-5",
        secret="ANTHROPIC_API_KEY",
        handler=call_anthropic,
    ),
    "gemini-1.5-pro": ProviderConfig(
        key="gemini-1.5-pro",
        label="Google · Gemini 1.5 Pro Latest",
        provider="gemini",
        model="gemini-1.5-pro-latest",
        secret="GEMINI_API_KEY",
        handler=call_gemini,
    ),
    "qwen-plus": ProviderConfig(
        key="qwen-plus",
        label="Qwen · Plus",
        provider="qwen",
        model="qwen-plus",
        secret="DASHSCOPE_API_KEY",
        handler=call_qwen,
        supports_stream=True,
    ),
    "groq-llama33": ProviderConfig(
        key="groq-llama33",
        label="Groq · Llama 3.3 70B",
        provider="groq",
        model="llama-3.3-70b-versatile",
        secret="GROQ_API_KEY",
        handler=call_groq,
        supports_stream=True,
    ),
    "deepseek-reasoner": ProviderConfig(
        key="deepseek-reasoner",
        label="DeepSeek · Reasoner",
        provider="deepseek",
        model="deepseek-reasoner",
        secret="DEEPSEEK_API_KEY",
        handler=call_deepseek,
        supports_stream=True,
    ),
}


def available_models() -> List[ProviderConfig]:
    """Return registry entries with availability flag."""
    configs: List[ProviderConfig] = []
    for config in MODEL_REGISTRY.values():
        configs.append(config)
    return configs


def enforce_rate_limits(session_id: str) -> None:
    """Apply burst and persistent rate limiting."""
    allowed, remaining = burst_limiter.check(session_id)
    if not allowed:
        raise RateLimitError("Too many requests this minute. Please wait a few seconds.")

    db_allowed, _, reset_seconds = increment_rate_counter(
        session_id, limit=200, window_minutes=60 * 24
    )
    if not db_allowed:
        raise RateLimitError(
            f"You have reached the daily limit. Try again in {reset_seconds // 60} minutes."
        )


def call_model(
    model_key: str,
    messages: List[Dict[str, str]],
    *,
    session_id: str,
    stream: bool = False,
    on_token=None,
) -> str:
    """Entry point used by the UI."""
    config = MODEL_REGISTRY.get(model_key)
    if not config:
        return "Unknown model selected."

    if not _secret(config.secret):
        return f"{config.label} requires the secret {config.secret}."

    enforce_rate_limits(session_id)
    return config.handler(
        messages,
        model=config.model,
        stream=stream,
        on_token=on_token,
    )


def evaluate_responses(
    prompt: str,
    response_a: str,
    response_b: str,
    model_a_name: str,
    model_b_name: str,
) -> str:
    """
    Uses GPT-5 nano to evaluate which response is better.
    Returns a brief evaluation (max 30 words).
    """
    api_key = _secret("OPENAI_API_KEY")
    if not api_key:
        return "Evaluator unavailable — missing OPENAI_API_KEY."

    try:
        from openai import OpenAI
    except ImportError:
        return "Evaluator unavailable — OpenAI SDK missing."

    client = OpenAI(api_key=api_key)

    evaluation_prompt = f"""Compare these two responses to the question: "{prompt}"

Response 1 ({model_a_name}):
{response_a}

Response 2 ({model_b_name}):
{response_b}

Which response is better? Reply ONLY with: "Answer 1 is better: [reason]" or "Answer 2 is better: [reason]". Keep reason under 20 words."""

    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=evaluation_prompt
        )
        evaluation = response.output_text or "Unable to evaluate."
        # Ensure it's under 30 words
        words = evaluation.strip().split()
        if len(words) > 30:
            evaluation = " ".join(words[:30]) + "..."
        return evaluation
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        return "Evaluation error — try again."

