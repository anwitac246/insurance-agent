"""
llm_client.py
--------------
Centralized LLM factory using Ollama.
Supports both sync (get_llm) and async (get_async_llm) interfaces.
"""

from __future__ import annotations

import logging
import time

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

OLLAMA_MODEL = "llama3:latest"

def get_llm() -> ChatOllama:
    return ChatOllama(model=OLLAMA_MODEL, temperature=0, request_timeout=45)

def get_async_llm() -> ChatOllama:
    return ChatOllama(model=OLLAMA_MODEL, temperature=0, request_timeout=45)

def get_fast_llm() -> ChatOllama:
    return ChatOllama(model=OLLAMA_MODEL, temperature=0, request_timeout=45)

def get_async_fast_llm() -> ChatOllama:
    return ChatOllama(model=OLLAMA_MODEL, temperature=0, request_timeout=45)

def record_success() -> None:
    pass

def record_429() -> None:
    pass

def adaptive_sleep(base_delay: float) -> float:
    time.sleep(base_delay)
    return base_delay
