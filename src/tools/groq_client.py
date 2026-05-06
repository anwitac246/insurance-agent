"""
groq_client.py
--------------
Centralized Groq LLM factory with automatic API key rotation.
Supports both sync (get_llm) and async (get_async_llm) interfaces.

Reads GROQ_API_KEYS from .env as a comma-separated list:
    GROQ_API_KEYS=key1,key2,key3

Falls back to the legacy GROQ_API_KEY (single key) if GROQ_API_KEYS is not set.

Rotation policy
---------------
- A per-key consecutive-429 counter is maintained.
- If a key hits ROTATION_THRESHOLD (default 5) consecutive 429s, it is marked
  exhausted and the next available key is activated.
- On successful call, the counter for the active key resets to 0.
- If ALL keys are exhausted, a GroqKeysExhaustedError is raised immediately
  so callers fail fast rather than looping forever.
- Keys rotate in round-robin order. Exhausted keys re-enter the pool after
  COOLDOWN_SECONDS (default 60) to handle transient rate limits.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_FAST_MODEL = "llama-3.1-8b-instant"   # for simple classification tasks
ROTATION_THRESHOLD = 5
COOLDOWN_SECONDS = 60


class GroqKeysExhaustedError(RuntimeError):
    """Raised when every configured Groq API key is rate-limited simultaneously."""


def _load_keys() -> list[str]:
    multi = os.getenv("GROQ_API_KEYS", "")
    if multi:
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            logger.info("groq_client | loaded %d API key(s) from GROQ_API_KEYS", len(keys))
            return keys

    single = os.getenv("GROQ_API_KEY", "")
    if single:
        logger.info("groq_client | loaded 1 API key from GROQ_API_KEY (fallback)")
        return [single]

    raise EnvironmentError(
        "No Groq API keys found. Set GROQ_API_KEYS=key1,key2,key3 "
        "(or the legacy GROQ_API_KEY) in your .env file."
    )


class _KeyRotationManager:
    """Thread-safe Groq API key rotation manager with sync + async LLM instances."""

    def __init__(self):
        self._lock = threading.Lock()
        self._keys: list[str] = _load_keys()
        self._current_idx: int = 0
        self._consecutive_429s: list[int] = [0] * len(self._keys)
        self._exhausted_at: list[Optional[float]] = [None] * len(self._keys)
        self._llm: Optional[ChatGroq] = None
        self._async_llm: Optional[ChatGroq] = None
        self._fast_llm: Optional[ChatGroq] = None
        self._async_fast_llm: Optional[ChatGroq] = None
        self._build_llms()

    def _build_llms(self) -> None:
        key = self._keys[self._current_idx]
        shared_kwargs = dict(temperature=0, request_timeout=45, api_key=key)

        # Sync instances
        self._llm = ChatGroq(model=GROQ_MODEL, **shared_kwargs)
        self._fast_llm = ChatGroq(model=GROQ_FAST_MODEL, **shared_kwargs)

        # Async instances — same class, async methods called via ainvoke
        self._async_llm = ChatGroq(model=GROQ_MODEL, **shared_kwargs)
        self._async_fast_llm = ChatGroq(model=GROQ_FAST_MODEL, **shared_kwargs)

        logger.info(
            "groq_client | active key index=%d (***%s)",
            self._current_idx, key[-4:],
        )

    def _is_key_available(self, idx: int) -> bool:
        exhausted_at = self._exhausted_at[idx]
        if exhausted_at is None:
            return True
        if time.monotonic() - exhausted_at >= COOLDOWN_SECONDS:
            logger.info("groq_client | key index=%d cooled down, re-entering pool", idx)
            self._exhausted_at[idx] = None
            self._consecutive_429s[idx] = 0
            return True
        return False

    def _rotate(self) -> None:
        num_keys = len(self._keys)
        for offset in range(1, num_keys + 1):
            candidate = (self._current_idx + offset) % num_keys
            if self._is_key_available(candidate):
                old_idx = self._current_idx
                self._current_idx = candidate
                logger.warning(
                    "groq_client | rotated from key index=%d to index=%d",
                    old_idx, self._current_idx,
                )
                self._build_llms()
                return
        raise GroqKeysExhaustedError(
            f"All {num_keys} Groq API key(s) are currently rate-limited. "
            f"Wait ~{COOLDOWN_SECONDS}s and retry."
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_llm(self) -> ChatGroq:
        with self._lock:
            if self._llm is None:
                self._build_llms()
            return self._llm

    def get_async_llm(self) -> ChatGroq:
        with self._lock:
            if self._async_llm is None:
                self._build_llms()
            return self._async_llm

    def get_fast_llm(self) -> ChatGroq:
        with self._lock:
            if self._fast_llm is None:
                self._build_llms()
            return self._fast_llm

    def get_async_fast_llm(self) -> ChatGroq:
        with self._lock:
            if self._async_fast_llm is None:
                self._build_llms()
            return self._async_fast_llm

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_429s[self._current_idx] = 0

    def record_429(self) -> None:
        with self._lock:
            self._consecutive_429s[self._current_idx] += 1
            count = self._consecutive_429s[self._current_idx]
            logger.warning(
                "groq_client | 429 on key index=%d (consecutive=%d/%d)",
                self._current_idx, count, ROTATION_THRESHOLD,
            )
            if count >= ROTATION_THRESHOLD:
                self._exhausted_at[self._current_idx] = time.monotonic()
                logger.warning(
                    "groq_client | key index=%d marked exhausted after %d consecutive 429s",
                    self._current_idx, count,
                )
                self._rotate()


# ── Module-level singleton ─────────────────────────────────────────────────────
_manager: Optional[_KeyRotationManager] = None
_manager_lock = threading.Lock()


def _get_manager() -> _KeyRotationManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = _KeyRotationManager()
    return _manager


# ── Public interface ───────────────────────────────────────────────────────────

def get_llm() -> ChatGroq:
    """Return the currently active ChatGroq instance (sync)."""
    return _get_manager().get_llm()


def get_async_llm() -> ChatGroq:
    """Return the currently active ChatGroq instance for async calls."""
    return _get_manager().get_async_llm()


def get_fast_llm() -> ChatGroq:
    """Return the fast (8B) ChatGroq instance for simple classification tasks (sync)."""
    return _get_manager().get_fast_llm()


def get_async_fast_llm() -> ChatGroq:
    """Return the fast (8B) ChatGroq instance for async simple classification tasks."""
    return _get_manager().get_async_fast_llm()


def record_success() -> None:
    _get_manager().record_success()


def record_429() -> None:
    _get_manager().record_429()