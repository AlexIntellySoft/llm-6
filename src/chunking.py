"""Нарезка на чанки с жёстко привязанными метаданными.

Каждый чанк несёт словарь
    {"source": "Справочник_сотрудника_HR.pdf", "page": 4, "header": "Основные факты", ...}
который без изменений уходит в ChromaDB вместе с вектором и возвращается при поиске.

Две стратегии (см. ChunkConfig.strategy):

* page — режем каждый Segment отдельно. Segment по построению лежит на одной
  странице и в одном разделе, значит, у любого чанка ровно одна страница и
  один заголовок. Плата — абзац, разорванный переносом страницы, попадает в
  два чанка.
* flow — склеиваем документ в сплошной текст и режем его целиком; странице
  чанка присваивается страница его первого символа. Так работает большинство
  «наивных» пайплайнов (PyPDFLoader → splitter), и чем крупнее чанк, тем
  чаще факт на самом деле лежит на следующей странице.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ChunkConfig
from src.parsing import Segment

SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    chunk_id: str
    text: str              # то, что видит модель и пользователь
    meta: dict             # source, page, page_end, page_label, header, chapter, doc_type, title

    @property
    def embed_text(self) -> str:
        # «Хлебная крошка» в эмбеддинге: короткий фрагмент списка без названия
        # главы почти не ищется («• Срок подачи заявки…» — заявки на что?).
        crumbs = " / ".join(x for x in (self.meta["chapter"], self.meta["header"]) if x)
        return f"{crumbs}\n{self.text}" if crumbs else self.text


def _splitter(cfg: ChunkConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_size,
                                          chunk_overlap=cfg.chunk_overlap,
                                          separators=SEPARATORS, add_start_index=True)


def _meta(seg: Segment, page: int | None = None, page_end: int | None = None,
          header: str | None = None, label: str | None = None) -> dict:
    page = seg.page if page is None else page
    return {
        "source": seg.source,
        "page": page,
        "page_end": page if page_end is None else page_end,
        "page_label": seg.page_label if label is None else label,
        "header": seg.header if header is None else header,
        "chapter": seg.chapter,
        "title": seg.title,
        "doc_type": seg.doc_type,
    }


def chunk_page(segments: list[Segment], cfg: ChunkConfig) -> list[Chunk]:
    sp = _splitter(cfg)
    out: list[Chunk] = []
    for seg in segments:
        for piece in sp.split_text(seg.text):
            out.append(Chunk("", piece, _meta(seg)))
    return _number(out)


def chunk_flow(segments: list[Segment], cfg: ChunkConfig) -> list[Chunk]:
    """Документ целиком → сплошной текст → чанки; страница = страница начала чанка."""
    sp = _splitter(cfg)
    out: list[Chunk] = []
    by_doc: dict[str, list[Segment]] = {}
    for seg in segments:
        by_doc.setdefault(seg.source, []).append(seg)

    for segs in by_doc.values():
        starts, text = [], ""
        for seg in segs:
            starts.append(len(text))
            text += seg.text + "\n\n"
        for doc in sp.create_documents([text]):
            a = doc.metadata["start_index"]
            b = a + len(doc.page_content) - 1
            i = bisect.bisect_right(starts, a) - 1
            j = bisect.bisect_right(starts, b) - 1
            first = segs[i]
            out.append(Chunk("", doc.page_content,
                             _meta(first, page=first.page, page_end=segs[j].page)))
    return _number(out)


def _number(chunks: list[Chunk]) -> list[Chunk]:
    counter: dict[str, int] = {}
    for c in chunks:
        n = counter.get(c.meta["source"], 0)
        counter[c.meta["source"]] = n + 1
        c.chunk_id = f"{c.meta['source']}#{n:03d}"
        c.meta["chunk_id"] = c.chunk_id
    return chunks


def make_chunks(segments: list[Segment], cfg: ChunkConfig) -> list[Chunk]:
    return chunk_flow(segments, cfg) if cfg.strategy == "flow" else chunk_page(segments, cfg)
