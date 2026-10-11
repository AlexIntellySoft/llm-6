"""docs/test_log.md — документ для сдачи: тестовые запросы и полные ответы ассистента со ссылками.

    python -m tools.build_test_log

Собирается из outputs/exp_main.json (реальные ответы YandexGPT, строгий промпт).
"""
from __future__ import annotations

import json
from pathlib import Path

from src import config
from src.citations import linkify
from src.prompts import SYSTEM_STRICT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "test_log.md"

TYPE_RU = {"direct": "Прямой вопрос", "aggregation": "Вопрос с агрегацией",
           "ood": "Провокация (Out-of-Domain)", "long": "Длинный ответ"}
STATUS_RU = {"ok": "есть в контексте", "not_in_context": "не было в контексте",
             "bad_page": "нет такой страницы", "fabricated": "источник выдуман",
             "malformed": "без номера страницы"}


def block(r: dict) -> list[str]:
    g = r["grade"]
    out = [f"### {r['id']}. {TYPE_RU[r['type']]}", "", f"**Вопрос:** {r['question']}", "",
           "**Ответ ассистента:**", ""]
    out += ["> " + line for line in linkify(r["answer"]).strip().split("\n")]
    out += ["", "<details><summary>Что нашёл ретривер и как проверены ссылки</summary>", ""]
    if r.get("subqueries"):
        out += ["Подвопросы для поиска: " + " · ".join(r["subqueries"]), ""]
    out += ["| # | источник | раздел | близость |", "|---|---|---|---|"]
    out += [f"| {i} | `{h['ref']}` | {h['header']} | {h['score']:.3f} |"
            for i, h in enumerate(r["hits"], 1)]
    out.append("")
    if r["citations"]:
        out += ["Ссылки в ответе: " + "; ".join(f"`[{c['raw']}]` — {STATUS_RU[c['status']]}"
                                               for c in r["citations"]), ""]
    for f in g["facts"]:
        where = ", ".join(f"{s}, стр. {p}" if p else s for s, p in f["gold"])
        mark = "✅ подтверждён ссылкой" if f["supported"] else (
            "⚠️ в ответе есть, ссылки на источник нет" if f["stated"] else "❌ в ответе нет")
        out.append(f"- «{f['needle']}» (на самом деле: {where}) — {mark}")
    out += ["", "</details>", "",
            f"**Итог:** {'✅ пройден' if g['passed'] else '❌ не пройден'}", "", "---", ""]
    return out


def main() -> None:
    main_runs = json.loads((config.OUT_DIR / "exp_main.json").read_text(encoding="utf-8"))["strict"]
    showcase = [r for r in main_runs if r["showcase"]]
    rest = [r for r in main_runs if not r["showcase"]]
    passed = sum(r["grade"]["passed"] for r in main_runs)
    cites = [c for r in main_runs for c in r["citations"]]
    lines = [
        "# Corporate Knowledge Assistant: лог тестовых запросов",
        "",
        f"Модель: YandexGPT 5 Lite (`{config.YC_MODEL}`), temperature {config.TEMPERATURE}. "
        f"Поиск: multilingual-e5-base + ChromaDB, top-{config.TOP_K}, чанки "
        f"{config.DEFAULT_CHUNKS.chunk_size} символов в пределах страницы. Ответы ниже — реальные "
        "ответы модели без правки; ссылки сделаны кликабельными (PDF открывается на нужной странице).",
        "",
        f"**Итог по {len(main_runs)} вопросам:** пройдено {passed}; ссылок {len(cites)}, "
        f"из них на выдуманные файлы или страницы — "
        f"{sum(c['status'] in ('fabricated', 'bad_page', 'malformed') for c in cites)}. "
        "Подробный разбор — в [`notebooks/report.ipynb`](../notebooks/report.ipynb).",
        "",
        "Как проверяется каждый ответ: (1) ссылка указывает на существующий файл и страницу, и "
        "этот фрагмент был в контексте модели; (2) цитируемая страница PDF действительно "
        "содержит факт — сверка по тексту страницы в файле, а не по метаданным.",
        "",
        "<details><summary>Системный промпт</summary>", "", "```text", SYSTEM_STRICT, "```", "",
        "</details>", "",
        f"## Основные тестовые запросы ({len(showcase)})", "",
    ]
    for r in showcase:
        lines += block(r)
    lines += [f"## Остальные запросы ({len(rest)})", ""]
    for r in rest:
        lines += block(r)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {len(main_runs)} вопросов")


if __name__ == "__main__":
    main()
