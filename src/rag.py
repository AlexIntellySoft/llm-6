"""RAG-цепочка: вопрос → retriever (top-K с метаданными) → контекст с источниками → YandexGPT.

Для многотемных вопросов («расскажи про отпуска, пароли, VPN и дресс-код»)
один вектор запроса тянет чанки только по одной-двум темам, и модель честно
отвечает «Я не знаю» на остальные. Поэтому есть режим decompose: LLM сначала
разбивает вопрос на подвопросы, по каждому ищется top-k_each, результаты
объединяются без дублей.
"""
from __future__ import annotations

import re

from src import config
from src.citations import Citation, coverage, parse_citations
from src.config import ChunkConfig
from src.index import Hit, retrieve
from src.llm import LLMReply, chat
from src.prompts import build_messages

DECOMPOSE_PROMPT = """Разбей вопрос сотрудника на независимые короткие поисковые подвопросы — по одному на каждую тему, которую он затрагивает. Не больше 8. Выведи только подвопросы, каждый с новой строки, без нумерации и пояснений."""


def subqueries(question: str, model: str = config.YC_MODEL) -> list[str]:
    reply = chat([{"role": "system", "content": DECOMPOSE_PROMPT},
                  {"role": "user", "content": question}], model=model, temperature=0.0,
                 max_tokens=300)
    lines = [re.sub(r"^[\s\-–•*\d.)]+", "", ln).strip() for ln in reply.text.splitlines()]
    return [ln for ln in lines if len(ln) > 3][:8]


def retrieve_decomposed(question: str, k_each: int = 2, cfg: ChunkConfig = config.DEFAULT_CHUNKS,
                        limit: int = 14) -> tuple[list[Hit], list[str]]:
    subs = subqueries(question)
    seen, hits = set(), []
    for sq in subs:
        for h in retrieve(sq, k=k_each, cfg=cfg):
            if h.meta["chunk_id"] not in seen:
                seen.add(h.meta["chunk_id"])
                hits.append(h)
    return hits[:limit], subs


def answer(question: str, k: int = config.TOP_K, cfg: ChunkConfig = config.DEFAULT_CHUNKS,
           system: str = "strict", style: str = "", model: str = config.YC_MODEL,
           decompose: bool = False, hits: list[Hit] | None = None,
           with_ref: bool = True) -> dict:
    subs: list[str] = []
    if hits is None:
        if decompose:
            hits, subs = retrieve_decomposed(question, cfg=cfg)
        else:
            hits = retrieve(question, k=k, cfg=cfg)
    messages = build_messages(question, hits, system=system, answer_style=style,
                              with_ref=with_ref)
    reply: LLMReply = chat(messages, model=model)
    cites: list[Citation] = parse_citations(reply.text, hits)
    return {
        "question": question,
        "subqueries": subs,
        "answer": reply.text,
        "hits": [{"ref": h.ref(), "header": h.meta["header"], "score": h.score,
                  "text": h.text, "page_label": h.meta.get("page_label", "")} for h in hits],
        "citations": [c.__dict__ for c in cites],
        "coverage": coverage(reply.text),
        "usage": {"prompt": reply.prompt_tokens, "completion": reply.completion_tokens,
                  "seconds": reply.seconds, "cost_rub": round(reply.cost_rub, 4),
                  "cached": reply.cached, "model": model},
        "messages": messages,
    }


def answer_by_sections(question: str, k_each: int = 3, cfg: ChunkConfig = config.DEFAULT_CHUNKS,
                       system: str = "strict", model: str = config.YC_MODEL) -> dict:
    """Длинный ответ как сумма коротких: каждый подвопрос — отдельный вызов со своим
    контекстом. Короткий ответ модель не «забывает» подписать, а в итоговом тексте
    каждая ссылка стоит рядом со своим фактом."""
    subs = subqueries(question, model=model)
    parts, all_hits, usage = [], [], {"prompt": 0, "completion": 0, "seconds": 0.0, "cost_rub": 0.0}
    for sq in subs:
        hits = retrieve(sq, k=k_each, cfg=cfg)
        reply = chat(build_messages(sq, hits, system=system), model=model)
        parts.append(f"**{sq}**\n{reply.text.strip()}")
        all_hits += hits
        usage["prompt"] += reply.prompt_tokens
        usage["completion"] += reply.completion_tokens
        usage["seconds"] = round(usage["seconds"] + reply.seconds, 2)
        usage["cost_rub"] = round(usage["cost_rub"] + reply.cost_rub, 4)
    text = "\n\n".join(parts)
    cites = parse_citations(text, all_hits)
    return {
        "question": question, "subqueries": subs, "answer": text,
        "hits": [{"ref": h.ref(), "header": h.meta["header"], "score": h.score,
                  "text": h.text, "page_label": h.meta.get("page_label", "")} for h in all_hits],
        "citations": [c.__dict__ for c in cites], "coverage": coverage(text),
        "usage": {**usage, "model": model, "calls": len(subs)},
    }


if __name__ == "__main__":
    import sys

    from src.citations import linkify

    r = answer(" ".join(sys.argv[1:]) or "За сколько дней нужно писать заявление на отпуск?")
    for h in r["hits"]:
        print(f"  {h['score']:.3f} {h['ref']} · {h['header']}")
    print("\n" + linkify(r["answer"]))
    print("\n", r["citations"], r["usage"])
