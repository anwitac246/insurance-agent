"""
groq_client.py
--------------
Centralized Groq LLM factory with automatic API key rotation.

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

Usage
-----
    from src.tools.groq_client import get_llm, record_success, record_429

    llm = get_llm()   # always returns the current active ChatGroq instance

    # In your retry / exception handler:
    except groq.RateLimitError:
        record_429()
        llm = get_llm()   # may return a new instance on a fresh key
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

GROQ_MODEL = "llama-3.3-70b-versatile"
ROTATION_THRESHOLD = 5      # consecutive 429s before rotating
COOLDOWN_SECONDS = 60       # seconds before an exhausted key re-enters the pool


class GroqKeysExhaustedError(RuntimeError):
    """Raised when every configured Groq API key is rate-limited simultaneously."""


def _load_keys() -> list[str]:
    """Parse GROQ_API_KEYS (comma-separated) or fall back to GROQ_API_KEY."""
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
    """Thread-safe Groq API key rotation manager."""

    def __init__(self):
        self._lock = threading.Lock()
        self._keys: list[str] = _load_keys()
        self._current_idx: int = 0
        self._consecutive_429s: list[int] = [0] * len(self._keys)
        self._exhausted_at: list[Optional[float]] = [None] * len(self._keys)
        self._llm: Optional[ChatGroq] = None
        self._build_llm()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_llm(self) -> None:
        """(Re)build the ChatGroq instance for the current active key."""
        key = self._keys[self._current_idx]
        self._llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0,
            request_timeout=45,
            api_key=key,
        )
        logger.info(
            "groq_client | active key index=%d (***%s)",
            self._current_idx,
            key[-4:],
        )

    def _is_key_available(self, idx: int) -> bool:
        """Return True if the key at idx is not exhausted (or cooldown has passed)."""
        exhausted_at = self._exhausted_at[idx]
        if exhausted_at is None:
            return True
        if time.monotonic() - exhausted_at >= COOLDOWN_SECONDS:
            # Cooldown passed — reset this key
            logger.info(
                "groq_client | key index=%d cooled down, re-entering pool", idx
            )
            self._exhausted_at[idx] = None
            self._consecutive_429s[idx] = 0
            return True
        return False

    def _rotate(self) -> None:
        """Switch to the next available key. Raises if all keys are exhausted."""
        num_keys = len(self._keys)
        for offset in range(1, num_keys + 1):
            candidate = (self._current_idx + offset) % num_keys
            if self._is_key_available(candidate):
                old_idx = self._current_idx
                self._current_idx = candidate
                logger.warning(
                    "groq_client | rotated from key index=%d to index=%d",
                    old_idx,
                    self._current_idx,
                )
                self._build_llm()
                return

        raise GroqKeysExhaustedError(
            f"All {num_keys} Groq API key(s) are currently rate-limited. "
            f"Wait ~{COOLDOWN_SECONDS}s and retry."
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_llm(self) -> ChatGroq:
        with self._lock:
            if self._llm is None:
                self._build_llm()
            return self._llm

    def record_success(self) -> None:
        """Call after a successful LLM response to reset the 429 counter."""
        with self._lock:
            self._consecutive_429s[self._current_idx] = 0

    def record_429(self) -> None:
        """
        Call when a 429 / RateLimitError is caught.
        Increments the counter; rotates the key if ROTATION_THRESHOLD is reached.
        """
        with self._lock:
            self._consecutive_429s[self._current_idx] += 1
            count = self._consecutive_429s[self._current_idx]
            logger.warning(
                "groq_client | 429 on key index=%d (consecutive=%d/%d)",
                self._current_idx,
                count,
                ROTATION_THRESHOLD,
            )

            if count >= ROTATION_THRESHOLD:
                self._exhausted_at[self._current_idx] = time.monotonic()
                logger.warning(
                    "groq_client | key index=%d marked exhausted after %d consecutive 429s",
                    self._current_idx,
                    count,
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
    """Return the currently active ChatGroq instance."""
    return _get_manager().get_llm()


def record_success() -> None:
    """Signal a successful LLM call (resets the 429 counter for the active key)."""
    _get_manager().record_success()


def record_429() -> None:
    """
    Signal a 429 / RateLimitError.
    After ROTATION_THRESHOLD consecutive calls, the key is rotated automatically.
    """
    _get_manager().record_429()