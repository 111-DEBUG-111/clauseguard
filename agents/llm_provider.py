"""
ClauseGuard — LLM Provider Configuration
==========================================
Centralised factory for the LangChain chat model.

Priority:
  1. Groq  (GROQ_API_KEY  — keys start with "gsk_")
  2. Grok / xAI  (XAI_API_KEY  — keys start with "xai-")
  3. OpenAI  (OPENAI_API_KEY  — keys start with "sk-")

The .env file is loaded automatically at import time via python-dotenv.
Copy .env.example → .env and fill in your key before running.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Auto-load .env from project root ─────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass   # python-dotenv not installed; rely on shell environment

from langchain_openai import ChatOpenAI


def _clean(key: str) -> str:
    """Strip stray whitespace and surrounding quotes from env var values."""
    return key.strip().strip('"').strip("'").strip()


def get_llm(temperature: float = 0.0, max_tokens: int = 2048) -> ChatOpenAI:
    """
    Return a LangChain-compatible ChatOpenAI instance.

    Provider priority
    -----------------
    1. Groq  (GROQ_API_KEY)  — fast open-source LLMs, keys start with "gsk_"
       Model: llama-3.3-70b-versatile (default, free tier available)
    2. xAI Grok  (XAI_API_KEY)  — keys start with "xai-"
       Model: grok-3
    3. OpenAI  (OPENAI_API_KEY)  — keys start with "sk-"
       Model: gpt-4o

    Parameters
    ----------
    temperature:
        0.0  → deterministic (recommended for classification tasks)
        >0.0 → more expressive (useful for explanation generation)
    max_tokens:
        Upper bound on response length.
    """
    # ── 1. Groq ───────────────────────────────────────────────────────────────
    groq_key = _clean(os.getenv("GROQ_API_KEY", ""))
    if groq_key:
        return ChatOpenAI(
            model="llama-3.3-70b-versatile",      # best free-tier Groq model
            openai_api_key=groq_key,
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ── 2. xAI / Grok ────────────────────────────────────────────────────────
    xai_key = _clean(os.getenv("XAI_API_KEY", ""))
    if xai_key:
        return ChatOpenAI(
            model="grok-3",
            openai_api_key=xai_key,
            openai_api_base="https://api.x.ai/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ── 3. OpenAI ─────────────────────────────────────────────────────────────
    openai_key = _clean(os.getenv("OPENAI_API_KEY", ""))
    if openai_key:
        return ChatOpenAI(
            model="gpt-4o",
            openai_api_key=openai_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise EnvironmentError(
        "\n\n❌  No LLM API key found.\n"
        "   Set ONE of the following in your .env file (copy .env.example → .env):\n\n"
        "     GROQ_API_KEY=gsk_...        ← Groq  (free tier at console.groq.com)\n"
        "     XAI_API_KEY=xai-...         ← xAI Grok\n"
        "     OPENAI_API_KEY=sk-...       ← OpenAI GPT-4o\n"
    )
