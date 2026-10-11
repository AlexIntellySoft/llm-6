"""YandexGPT через OpenAI-совместимый API + дисковый кэш ответов.

Каждый ответ сохраняется в outputs/llm_cache/<sha256>.json (ключ — модель,
температура и полный список сообщений). В режиме replay сеть не нужна: ноутбук
перезапускается бесплатно и после истечения ключа.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from functools import lru_cache

from src import config


@dataclass
class LLMReply:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    cached: bool

    @property
    def cost_rub(self) -> float:
        return (self.prompt_tokens + self.completion_tokens) / 1000 * config.price_per_1k(self.model)


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    if not (config.YC_API_KEY and config.YC_FOLDER_ID):
        raise RuntimeError("Нет YC_API_KEY / YC_FOLDER_ID в .env — доступен только LLM_MODE=replay")
    return OpenAI(api_key=config.YC_API_KEY, base_url=config.YC_OPENAI_BASE_URL,
                  default_headers={"OpenAI-Project": config.YC_FOLDER_ID},
                  max_retries=4, timeout=120)


def _key(model: str, temperature: float, max_tokens: int, messages: list[dict]) -> str:
    raw = json.dumps([model, temperature, max_tokens, messages], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def chat(messages: list[dict], model: str = config.YC_MODEL,
         temperature: float = config.TEMPERATURE, max_tokens: int = config.MAX_TOKENS) -> LLMReply:
    key = _key(model, temperature, max_tokens, messages)
    path = config.LLM_CACHE / f"{key}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        return LLMReply(d["text"], model, d["prompt_tokens"], d["completion_tokens"],
                        d["seconds"], cached=True)
    if config.LLM_MODE != "live":
        raise RuntimeError(f"LLM_MODE={config.LLM_MODE}: ответа нет в кэше ({key})")

    t0 = time.perf_counter()
    resp = _client().chat.completions.create(model=config.model_uri(model), messages=messages,
                                             temperature=temperature, max_tokens=max_tokens)
    secs = time.perf_counter() - t0
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    reply = LLMReply(text, model, usage.prompt_tokens, usage.completion_tokens,
                     round(secs, 2), cached=False)
    config.LLM_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "temperature": temperature,
                                "messages": messages, "text": text,
                                "prompt_tokens": reply.prompt_tokens,
                                "completion_tokens": reply.completion_tokens,
                                "seconds": reply.seconds}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return reply
