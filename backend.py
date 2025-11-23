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
    model: str = "gpt-4.1-mini",
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
    model: str = "claude-3-5-sonnet-20241022",
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
            system=DEFAULT_SYSTEM_PROMPT,
            model=model,
            max_tokens=1024,
            temperature=temperature,
            messages=prepared,
        )
        parts = [
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", "") == "text"
        ]
        text = "\n\n".join(parts).strip() or "Claude returned an empty response."
        return emit_or_collect(text, stream, on_token)
    except Exception as exc:
        logger.exception("Anthropic call failed: %s", exc)
        return "Anthropic provider error — try again later."


def call_gemini(
    messages: List[Dict[str, str]],
    *,
    model: str = "gemini-1.5-pro-002",
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
        model_client = genai.GenerativeModel(model_name=model)
        response = model_client.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": 2048},
            request_options={"timeout": API_TIMEOUT},
        )
        parts: List[str] = []
        for candidate in getattr(response, "candidates", []):
            for part in getattr(getattr(candidate, "content", None), "parts", []) or []:
                text = getattr(part, "text", "")
                if text:
                    parts.append(text.strip())
        text = "\n\n".join(parts) or "Gemini did not return text."
        return emit_or_collect(text, stream, on_token)
    except Exception as exc:
        logger.exception("Gemini call failed: %s", exc)
        return "Gemini provider error — please retry."


def call_qwen(
    messages: List[Dict[str, str]],
    *,
    model: str = "qwen2.5-72b-instruct",
    temperature: float = 0.2,
    stream: bool = False,
    on_token=None,
) -> str:
    api_key = _secret("QWEN_API_KEY")
    if not api_key:
        return "Qwen model unavailable — add QWEN_API_KEY to secrets."
    try:
        from openai import OpenAI
    except ImportError:
        return "OpenAI SDK required for Qwen compatible calls."

    client = OpenAI(api_key=api_key, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
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
            timeout=API_TIMEOUT,
        )
        text = resp.choices[0].message.content or ""
        return emit_or_collect(text, stream, on_token)
    except Exception as exc:
        logger.exception("Qwen call failed: %s", exc)
        return "Qwen provider error — try again."


def call_groq(
    messages: List[Dict[str, str]],
    *,
    model: str = "llama3-70b-8192",
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

        resp = client.chat.completions.create(
            model=model,
            messages=prepared,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        return emit_or_collect(text, stream, on_token)
    except Exception as exc:
        logger.exception("Groq call failed: %s", exc)
        return "Groq provider error — try later."


def call_deepseek(
    messages: List[Dict[str, str]],
    *,
    model: str = "deepseek-chat",
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
    "openai-gpt-4.1-mini": ProviderConfig(
        key="openai-gpt-4.1-mini",
        label="OpenAI · GPT-4.1-mini",
        provider="openai",
        model="gpt-4.1-mini",
        secret="OPENAI_API_KEY",
        handler=call_openai,
        supports_stream=True,
    ),
    "openai-gpt-4.1": ProviderConfig(
        key="openai-gpt-4.1",
        label="OpenAI · GPT-4.1",
        provider="openai",
        model="gpt-4.1",
        secret="OPENAI_API_KEY",
        handler=call_openai,
        supports_stream=True,
    ),
    "anthropic-claude": ProviderConfig(
        key="anthropic-claude",
        label="Anthropic · Claude 3.5 Sonnet",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        secret="ANTHROPIC_API_KEY",
        handler=call_anthropic,
    ),
    "gemini-1.5-pro": ProviderConfig(
        key="gemini-1.5-pro",
        label="Google · Gemini 1.5 Pro",
        provider="gemini",
        model="gemini-1.5-pro-002",
        secret="GEMINI_API_KEY",
        handler=call_gemini,
    ),
    "qwen-72b": ProviderConfig(
        key="qwen-72b",
        label="Qwen 2.5 · 72B",
        provider="qwen",
        model="qwen2.5-72b-instruct",
        secret="QWEN_API_KEY",
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
    "deepseek-r1": ProviderConfig(
        key="deepseek-r1",
        label="DeepSeek · R1",
        provider="deepseek",
        model="deepseek-chat",
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

