"""Эксперименты. Каждый пишет outputs/<имя>.json; ноутбук строится из этих файлов.

    python -m src.experiments retrieval   # chunk_size × стратегия → точность страницы (без LLM)
    python -m src.experiments main        # 12 вопросов, строгий и базовый промпты
    python -m src.experiments long        # теряются ли ссылки в длинном ответе
    python -m src.experiments pages       # «на какой странице?» при колонтитулах и оглавлении в контексте
    python -m src.experiments chunks_llm  # крупные «сквозные» чанки → цитируемая страница
"""
from __future__ import annotations

import json
import re
import sys
from statistics import mean

from src import config
from src.citations import chunk_contains, citation_supports, needle_pages, Citation
from src.config import ChunkConfig
from src.eval_set import PROBES, QUESTIONS, UNKNOWN_RE
from src.index import all_chunks, build, retrieve
from src.rag import answer

OUT = config.OUT_DIR


def save(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                      encoding="utf-8")


# --------------------------------------------------------------------------- #
# 1. Ретривер: размер чанка и стратегия нарезки против точности номера страницы
# --------------------------------------------------------------------------- #
SIZES = [200, 400, 600, 1000, 1500, 2500]


def probe_config(cfg: ChunkConfig, k: int = 5) -> dict:
    build(cfg)
    rows = []
    for q, needle in PROBES:
        gold = needle_pages(needle)[0]
        hits = retrieve(q, k=k, cfg=cfg)
        rank = next((i for i, h in enumerate(hits) if chunk_contains(h.text, needle)), None)
        page_ok = None
        if rank is not None:
            h = hits[rank]
            page_ok = (h.source, h.page) == gold
        rows.append({"q": q, "needle": needle, "gold": gold, "rank": rank, "page_ok": page_ok,
                     "got": (hits[rank].source, hits[rank].page) if rank is not None else None})
    chunks = all_chunks(cfg)
    pdf_chunks = [c for c in chunks if c.meta["doc_type"] == "pdf"]
    found = [r for r in rows if r["rank"] is not None]
    return {
        "strategy": cfg.strategy, "chunk_size": cfg.chunk_size, "clean": cfg.clean,
        "chunks": len(chunks),
        "avg_len": round(mean(len(c.text) for c in chunks)),
        "multi_page_share": round(mean(c.meta["page_end"] != c.meta["page"] for c in pdf_chunks), 3),
        "recall@1": round(mean(r["rank"] == 0 for r in rows), 3),
        "recall@3": round(mean(r["rank"] is not None and r["rank"] < 3 for r in rows), 3),
        "recall@5": round(len(found) / len(rows), 3),
        "page_acc": round(mean(r["page_ok"] for r in found), 3) if found else None,
        "page_acc_all": round(sum(bool(r["page_ok"]) for r in rows) / len(rows), 3),
        "rows": rows,
    }


def exp_retrieval() -> list[dict]:
    res = []
    for strategy in ("page", "flow"):
        for size in SIZES:
            cfg = ChunkConfig(chunk_size=size, chunk_overlap=min(100, size // 5), strategy=strategy)
            r = probe_config(cfg)
            print(f"{strategy:5s} {size:5d} chunks={r['chunks']:4d} r@1={r['recall@1']:.2f} "
                  f"r@5={r['recall@5']:.2f} page_acc={r['page_acc']} multi={r['multi_page_share']}")
            res.append(r)
    # Отдельно: без чистки колонтитулов и оглавления (страничная нарезка, 400)
    r = probe_config(ChunkConfig(strategy="page", clean=False))
    print(f"raw   400 chunks={r['chunks']} r@1={r['recall@1']:.2f} page_acc={r['page_acc']}")
    res.append(r)
    save("exp_retrieval", res)
    return res


# --------------------------------------------------------------------------- #
# Оценка одного ответа
# --------------------------------------------------------------------------- #
def grade(q: dict, r: dict) -> dict:
    text = r["answer"]
    cites = [Citation(**c) for c in r["citations"]]
    said_unknown = bool(re.search(UNKNOWN_RE, text, re.I))
    facts = []
    for rx, needles in q["facts"]:
        needles = [needles] if isinstance(needles, str) else needles
        stated = bool(re.search(rx, text, re.I))
        # Ссылка подтверждает факт, если цитируемая страница реально содержит needle.
        # Факт, которого нет в ответе, подтверждённым не считается.
        supported = stated and any(citation_supports(c, n) for c in cites for n in needles)
        facts.append({"rx": rx, "needle": needles[0],
                      "gold": [p for n in needles for p in needle_pages(n)],
                      "stated": stated, "supported": supported})
    bad = [c for c in cites if c.status in ("fabricated", "bad_page", "malformed")]
    g = {
        "id": q["id"], "type": q["type"],
        "said_unknown": said_unknown,
        "facts_total": len(facts),
        "facts_stated": sum(f["stated"] for f in facts),
        "facts_supported": sum(f["supported"] for f in facts),
        "citations": len(cites),
        "citations_ok": sum(c.status == "ok" for c in cites),
        "citations_not_in_context": sum(c.status == "not_in_context" for c in cites),
        "citations_bad": len(bad),
        "distinct_sources": len({c.source for c in cites}),
        "facts": facts,
    }
    if q["type"] == "ood":
        g["passed"] = said_unknown and not cites
    else:
        g["passed"] = (g["facts_stated"] == g["facts_total"]
                       and g["facts_supported"] == g["facts_total"] and not bad)
    return g


def run_questions(system: str = "strict", cfg: ChunkConfig = config.DEFAULT_CHUNKS,
                  ids: list[str] | None = None, model: str = config.YC_MODEL) -> list[dict]:
    build(cfg)
    out = []
    for q in QUESTIONS:
        if ids and q["id"] not in ids:
            continue
        r = answer(q["q"], k=q.get("k", config.TOP_K), cfg=cfg, system=system, model=model,
                   decompose=q.get("decompose", False))
        r.update({"id": q["id"], "type": q["type"], "showcase": q.get("showcase", False),
                  "grade": grade(q, r)})
        g = r["grade"]
        print(f"{q['id']} {system:6s} passed={g['passed']!s:5s} facts {g['facts_stated']}/{g['facts_supported']}/"
              f"{g['facts_total']} cites ok={g['citations_ok']} bad={g['citations_bad']} "
              f"nic={g['citations_not_in_context']} unknown={g['said_unknown']}")
        out.append(r)
    return out


def exp_main() -> None:
    save("exp_main", {"strict": run_questions("strict"), "basic": run_questions("basic")})


# --------------------------------------------------------------------------- #
# 3. Длинные ответы: теряется ли ссылка
# --------------------------------------------------------------------------- #
LONG_STYLES = {
    "short": "Ответь коротко: 3–5 пунктов.",
    "default": "",
    "detailed": "Ответ должен быть максимально подробным и развёрнутым: распиши каждое правило отдельным абзацем, с пояснениями.",
}


def exp_long() -> None:
    from src.rag import answer_by_sections
    res = []
    for q in [x for x in QUESTIONS if x["type"] == "long"]:
        runs = [("top8", "default", "strict"), ("top8", "detailed", "strict"),
                ("decompose", "default", "strict"), ("decompose", "detailed", "strict"),
                ("decompose", "default", "basic"), ("sections", "per-section", "strict")]
        for retrieval, style, system in runs:
            if retrieval == "sections":
                # Исправление: длинный ответ собирается из коротких ответов на подвопросы.
                r = answer_by_sections(q["q"], system=system)
            else:
                r = answer(q["q"], k=8, system=system, style=LONG_STYLES[style],
                           decompose=retrieval == "decompose")
            r.update({"id": q["id"], "retrieval": retrieval, "style": style, "system": system,
                      "grade": grade(q, r)})
            cov, g = r["coverage"], r["grade"]
            print(f"{q['id']} {retrieval:9s} {style:11s} {system:6s} tok={r['usage']['completion']:4d} "
                  f"units {cov['units_cited']:2d}/{cov['units']:2d} blocks {cov['blocks_cited']}/{cov['blocks']} "
                  f"ref_lines={cov['ref_lines']} facts {g['facts_stated']}/{g['facts_supported']}/{g['facts_total']} "
                  f"bad={g['citations_bad']}")
            res.append(r)
    save("exp_long", res)


# --------------------------------------------------------------------------- #
# 4. «На какой странице…?»: что назовёт модель, если рядом с текстом есть
#    печатные номера (колонтитулы «— 2 —», строки оглавления «Глава 3 … 2»)
# --------------------------------------------------------------------------- #
PAGE_CONFIGS = {
    # чистые чанки + готовая ссылка в заголовке блока (основная конфигурация)
    "clean_strict": (config.DEFAULT_CHUNKS, "strict", True),
    # колонтитулы и оглавление в индексе, промпт и заголовки те же
    "raw_strict": (ChunkConfig(clean=False), "strict", True),
    # колонтитулы и оглавление + базовый промпт, в заголовке блока только «стр. N»
    "raw_basic": (ChunkConfig(clean=False), "basic", False),
    # наивный пайплайн: get_text() без метаданных страниц — номер только из текста
    "naive_nometa": (ChunkConfig(chunk_size=1000, chunk_overlap=100, strategy="naive",
                                 clean=False), "strict", False),
}
_PAGE_MENTION = re.compile(r"(?:стр\.?|страниц\w*)\s*(\d{1,3})", re.I)


def page_verdict(text: str, source: str, physical: int) -> dict:
    """Какие номера страниц прозвучали в ответе и чем они являются."""
    nums = [int(n) for n in _PAGE_MENTION.findall(text)]
    printed = physical - config.FRONT_PAGES
    kinds = ["physical" if n == physical else "printed" if n == printed else "other" for n in nums]
    return {"physical": physical, "printed": printed, "mentioned": nums, "kinds": kinds,
            "verdict": ("correct" if kinds and all(k == "physical" for k in kinds) else
                        "printed" if "printed" in kinds and "physical" not in kinds else
                        "mixed" if "physical" in kinds else
                        "none" if not kinds else "wrong")}


def exp_pages() -> None:
    from src.eval_set import PAGE_QUESTIONS
    res = {}
    for name, (cfg, system, with_ref) in PAGE_CONFIGS.items():
        build(cfg)
        rows = []
        for q in PAGE_QUESTIONS:
            source, physical = next((s, p) for s, p in needle_pages(q["needle"])
                                    if p > config.FRONT_PAGES)
            r = answer(q["q"], cfg=cfg, system=system, with_ref=with_ref)
            r.update({"id": q["id"], "source": source,
                      "toc_in_context": any(h["header"] == "Содержание" for h in r["hits"]),
                      "pages": page_verdict(r["answer"], source, physical)})
            print(f"{name:12s} {q['id']} {r['pages']['verdict']:8s} phys={physical} "
                  f"said={r['pages']['mentioned']} toc_ctx={r['toc_in_context']}")
            rows.append(r)
        res[name] = rows
    save("exp_pages", res)


# --------------------------------------------------------------------------- #
# 5. Крупные сквозные чанки: страница начала чанка ≠ страница факта
# --------------------------------------------------------------------------- #
def exp_chunks_llm() -> None:
    res = {}
    for cfg in (config.DEFAULT_CHUNKS,
                ChunkConfig(chunk_size=2500, chunk_overlap=100, strategy="flow")):
        res[cfg.key] = run_questions("strict", cfg=cfg, ids=["D1", "D2", "D3", "D4", "A2", "A4"])
    save("exp_chunks_llm", res)


if __name__ == "__main__":
    {"retrieval": exp_retrieval, "main": exp_main, "long": exp_long,
     "pages": exp_pages, "chunks_llm": exp_chunks_llm}[sys.argv[1]]()
