"""Разбор и проверка ссылок в ответе модели.

Каждая ссылка вида [Файл.pdf, стр. X] или [Название Wiki-статьи] получает статус:

* ok           — источник и страница есть среди фрагментов, переданных модели;
* not_in_context — файл существует и страница в нём есть, но такого фрагмента
                 модели не давали (ссылка «по памяти» или перепутанная страница);
* bad_page     — файл есть, а страницы с таким номером в нём нет;
* fabricated   — такого файла / Wiki-статьи в базе нет вообще (галлюцинация источника);
* malformed    — похоже на ссылку на PDF, но без номера страницы.

Отдельно проверяется, что цитируемая страница действительно содержит факт
(needle): по тексту самой страницы PDF, а не по метаданным чанка.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import pymupdf

from src import config
from src.index import Hit
from src.parsing import pdf_page_count, wiki_titles

_BRACKET = re.compile(r"\[([^\[\]\n]{2,200})\](?!\()")
_PDF_REF = re.compile(r"^\s*(?P<file>[^,\[\]]+?\.pdf)\s*(?:,\s*(?:стр\.?|с\.|страница)\s*(?P<page>\d+))?",
                      re.I)


@dataclass
class Citation:
    raw: str
    kind: str              # pdf | wiki | other
    source: str            # имя файла PDF или файла Wiki
    page: int | None
    status: str


def _norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[«»\"'“”]", "", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=1)
def _wiki_index() -> dict[str, str]:
    return {_norm(t): f for t, f in wiki_titles().items()}


def parse_citations(answer: str, hits: list[Hit]) -> list[Citation]:
    ctx_pages = {(h.source, h.page) for h in hits if h.meta["doc_type"] == "pdf"}
    ctx_wiki = {h.source for h in hits if h.meta["doc_type"] == "wiki"}
    out: list[Citation] = []
    for m in _BRACKET.finditer(answer):
        raw = m.group(1).strip()
        pm = _PDF_REF.match(raw)
        if pm:
            file = pm.group("file").strip()
            page = int(pm.group("page")) if pm.group("page") else None
            if file not in config.PDF_BOOKS:
                status = "fabricated"
            elif page is None:
                status = "malformed"
            elif not 1 <= page <= pdf_page_count(file):
                status = "bad_page"
            elif (file, page) in ctx_pages:
                status = "ok"
            else:
                status = "not_in_context"
            out.append(Citation(raw, "pdf", file, page, status))
            continue
        key = _norm(re.sub(r"^wiki\s*", "", raw, flags=re.I))
        wiki = _wiki_index()
        if key in wiki:
            f = wiki[key]
            out.append(Citation(raw, "wiki", f, None, "ok" if f in ctx_wiki else "not_in_context"))
        elif re.search(r"стр\.|\.pdf|wiki|вики", raw, re.I):
            out.append(Citation(raw, "other", raw, None, "fabricated"))
        # прочие [..] — не ссылки (например, «[имя руководителя]» из самого документа)
    return out


# --------------------------------------------------------------------------- #
# Где факт лежит на самом деле
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _page_texts(file: str) -> tuple[str, ...]:
    with pymupdf.open(config.PDF_DIR / file) as d:
        return tuple(_norm(p.get_text()) for p in d)


@lru_cache(maxsize=None)
def _wiki_text(file: str) -> str:
    from src.parsing import parse_html
    return _norm(" ".join(s.text for s in parse_html(config.WIKI_DIR / file)))


def needle_pages(needle: str) -> list[tuple[str, int]]:
    """Все (файл, страница), на которых встречается needle. Страница Wiki = 0."""
    n = _norm(needle)
    found = [(f, i + 1) for f in config.PDF_BOOKS for i, t in enumerate(_page_texts(f)) if n in t]
    found += [(f, 0) for f in config.WIKI_PAGES if n in _wiki_text(f)]
    return found


def citation_supports(c: Citation, needle: str) -> bool:
    if c.kind == "pdf" and c.page:
        texts = _page_texts(c.source)
        return 1 <= c.page <= len(texts) and _norm(needle) in texts[c.page - 1]
    if c.kind == "wiki":
        return _norm(needle) in _wiki_text(c.source)
    return False


def chunk_contains(text: str, needle: str) -> bool:
    return _norm(needle) in _norm(text)


# --------------------------------------------------------------------------- #
# Покрытие ответа ссылками и оформление
# --------------------------------------------------------------------------- #
def coverage(answer: str) -> dict:
    """Насколько плотно ответ подписан ссылками.

    * units  — строки-утверждения (абзац или пункт списка);
    * blocks — смысловые блоки: группа строк до пустой строки или подзаголовка;
               блок без единой ссылки = факты, источник которых потерян;
    * ref_lines — строки, где кроме ссылок ничего нет («Ссылки: [..][..]»):
               источник отъехал от фактов в конец ответа.
    """
    has = lambda s: bool(_BRACKET.search(s))
    units, blocks, block, ref_lines = [], [], [], 0
    for line in answer.split("\n") + [""]:
        u = line.strip()
        if not u or _is_heading(u):
            if block:
                blocks.append(block)
            block = []
            continue
        if "я не знаю" in u.lower():
            continue
        if has(u) and len(re.sub(r"(ссылк\w*|источник\w*)\s*:?", "", _BRACKET.sub("", u),
                                 flags=re.I).strip(" .,;:-–*")) < 3:
            ref_lines += 1          # строка из одних ссылок не делает фактом то, что выше
            continue
        units.append(u)
        block.append(u)
    sentences = [s for u in units for s in re.split(r"(?<=[.!?])\s+(?=[А-ЯA-Z«])", u) if len(s) > 25]
    return {
        "units": len(units),
        "units_cited": sum(map(has, units)),
        "sentences": len(sentences),
        "sentences_cited": sum(map(has, sentences)),
        "blocks": len(blocks),
        "blocks_cited": sum(any(map(has, b)) for b in blocks),
        "ref_lines": ref_lines,
    }


def _is_heading(line: str) -> bool:
    """«1. **Отпуска**:», «### Пароли», «**Онбординг:**» — подзаголовки, а не факты."""
    core = re.sub(r"^[#>\s]*(\d+[.)]\s*)?", "", line).strip()
    if _BRACKET.search(core) or len(core) > 90:
        return False
    return bool(re.fullmatch(r"\**[^.!]*?\**:?\**:?", core)) and (core.endswith(":") or
                                                                 core.endswith("**") or line.startswith("#"))


def linkify(answer: str, rel: str = "../data") -> str:
    """Ссылки → кликабельные markdown-ссылки на файл (PDF — сразу на нужную страницу)."""
    def repl(m: re.Match) -> str:
        raw = m.group(1).strip()
        pm = _PDF_REF.match(raw)
        if pm and pm.group("file").strip() in config.PDF_BOOKS and pm.group("page"):
            f, p = pm.group("file").strip(), pm.group("page")
            return f"**[\\[{f}, стр. {p}\\]]({rel}/pdf/{f}#page={p})**"
        key = _norm(re.sub(r"^wiki\s*", "", raw, flags=re.I))
        if key in _wiki_index():
            return f"**[\\[{raw}\\]]({rel}/wiki/{_wiki_index()[key]})**"
        if re.search(r"стр\.|\.pdf|wiki", raw, re.I):
            return f"**\\[{raw}\\]** ⚠️ *источника нет в базе*"
        return m.group(0)
    return _BRACKET.sub(repl, answer)
