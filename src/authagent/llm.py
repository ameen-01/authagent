"""LLM client: any OpenAI-compatible endpoint (Groq, OpenRouter, Ollama, vLLM...), with a disk
cache, a PHI guard and usage metering.

* Cache: every response is stored in cache/llm_cache.jsonl keyed by a hash of the request.
  Re-runs, teammates' runs and the live demo cost nothing; AUTHAGENT_OFFLINE=1 forbids any
  network call, so results are exactly reproducible from the committed cache.
* PHI guard: if a request contains any of the patient's identifiers, it is refused before it
  leaves the machine.
* Metering: calls, cache hits, tokens and latency are counted per case for the cost tables.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .common import CACHE_DIR
from .redact import find_leaks

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass


class PHILeakError(RuntimeError):
    pass


class CacheMiss(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    cached: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class Meter:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    by_purpose: dict = field(default_factory=dict)

    def add(self, purpose: str, r: LLMResponse) -> None:
        self.calls += 1
        self.cache_hits += int(r.cached)
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.latency_s += r.latency_s
        self.by_purpose[purpose] = self.by_purpose.get(purpose, 0) + 1

    def as_dict(self) -> dict:
        return {"llm_calls": self.calls, "cache_hits": self.cache_hits, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "llm_latency_s": round(self.latency_s, 3),
                "calls_by_purpose": dict(self.by_purpose)}


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates code fences and chatter)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON object")


class LLMClient:
    """OpenAI-compatible chat client. Configure with env vars:

    LLM_BASE_URL  e.g. https://api.groq.com/openai/v1  or  http://localhost:11434/v1 (Ollama)
    LLM_API_KEY   provider key ("ollama" for Ollama)
    LLM_MODEL     e.g. llama-3.3-70b-versatile  or  qwen2.5:7b-instruct
    """

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 600, cache_path: Path | None = None,
                 offline: bool | None = None, json_mode: bool | None = None, min_interval_s: float | None = None):
        self.model = model or os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.offline = offline if offline is not None else os.environ.get("AUTHAGENT_OFFLINE") == "1"
        self.json_mode = json_mode if json_mode is not None else os.environ.get("LLM_JSON_MODE", "1") == "1"
        self.min_interval_s = min_interval_s if min_interval_s is not None else float(os.environ.get("LLM_MIN_INTERVAL_S", "0"))
        self.cache_path = cache_path or (CACHE_DIR / "llm_cache.jsonl")
        self.cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._client = None
        self.meter = Meter()
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.cache[row["key"]] = row

    @property
    def name(self) -> str:
        return self.model

    def _key(self, messages: list[dict]) -> str:
        payload = json.dumps({"model": self.model, "messages": messages, "temperature": self.temperature,
                              "max_tokens": self.max_tokens, "json": self.json_mode}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _api(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key or "none", max_retries=6)
        return self._client

    def chat(self, messages: list[dict], purpose: str = "chat", phi: dict | None = None) -> LLMResponse:
        if phi:
            leaks = find_leaks(json.dumps(messages), phi)
            if leaks:
                raise PHILeakError(f"Refusing to send identifiers to the LLM: {leaks}")
        key = self._key(messages)
        hit = self.cache.get(key)
        if hit:
            r = LLMResponse(hit["text"], True, hit.get("prompt_tokens", 0), hit.get("completion_tokens", 0),
                            hit.get("latency_s", 0.0))
            self.meter.add(purpose, r)
            return r
        if self.offline:
            raise CacheMiss("AUTHAGENT_OFFLINE=1 and this request is not in the cache.")

        wait = self.min_interval_s - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.time()
        try:
            out = self._api().chat.completions.create(**kwargs)
        except Exception as exc:  # some providers/models reject response_format
            if self.json_mode and "response_format" in str(exc).lower():
                kwargs.pop("response_format")
                out = self._api().chat.completions.create(**kwargs)
            else:
                raise
        latency = time.time() - t0
        self._last_call = time.time()
        text = out.choices[0].message.content or ""
        usage = getattr(out, "usage", None)
        r = LLMResponse(text, False, getattr(usage, "prompt_tokens", 0) or 0,
                        getattr(usage, "completion_tokens", 0) or 0, latency)
        row = {"key": key, "model": self.model, "purpose": purpose, "text": text,
               "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "latency_s": round(latency, 3)}
        with self._lock:
            self.cache[key] = row
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.meter.add(purpose, r)
        return r


def get_llm(name: str | None = None):
    """'mock' -> scripted offline stand-in (plumbing tests only); anything else -> LLMClient."""
    name = name or os.environ.get("AUTHAGENT_LLM", "api")
    if name == "mock":
        from .mock_llm import MockLLM

        return MockLLM()
    return LLMClient()
