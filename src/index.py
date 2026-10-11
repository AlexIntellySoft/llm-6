"""Векторизация (multilingual-e5-base) и хранение в ChromaDB вместе с метаданными.

Под каждую конфигурацию нарезки — своя коллекция (имя = ChunkConfig.key),
поэтому эксперименты с chunk_size не перетирают друг друга, а повторный запуск
ноутбука не пересчитывает эмбеддинги.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from chromadb.config import Settings

from src import config
from src.chunking import Chunk, make_chunks
from src.config import ChunkConfig
from src.parsing import load_corpus, naive_corpus


@lru_cache(maxsize=1)
def embedder():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(config.EMBED_MODEL, device=device)


def embed(texts: list[str], prefix: str) -> list[list[float]]:
    vecs = embedder().encode([prefix + t for t in texts], batch_size=32,
                             normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()


@lru_cache(maxsize=1)
def client() -> chromadb.ClientAPI:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR),
                                     settings=Settings(anonymized_telemetry=False))


def build(cfg: ChunkConfig = config.DEFAULT_CHUNKS, recreate: bool = False) -> dict:
    """Парсинг → чанки → эмбеддинги → Chroma. Возвращает статистику."""
    name = cfg.key
    if not recreate:
        try:
            col = client().get_collection(name)
            if col.count():
                return {"collection": name, "chunks": col.count(), "cached": True}
        except Exception:
            pass
    try:
        client().delete_collection(name)
    except Exception:
        pass

    segments = naive_corpus() if cfg.strategy == "naive" else load_corpus(clean=cfg.clean)
    chunks = make_chunks(segments, cfg)
    t0 = time.perf_counter()
    vectors = embed([c.embed_text for c in chunks], config.PASSAGE_PREFIX)
    secs = time.perf_counter() - t0
    col = client().create_collection(name, metadata={"hnsw:space": "cosine"})
    col.add(ids=[c.chunk_id for c in chunks], embeddings=vectors,
            documents=[c.text for c in chunks], metadatas=[c.meta for c in chunks])
    return {"collection": name, "chunks": len(chunks), "embed_seconds": round(secs, 1),
            "cached": False}


@dataclass
class Hit:
    text: str
    meta: dict
    score: float           # косинусная близость, 1 — совпадение

    @property
    def source(self) -> str:
        return self.meta["source"]

    @property
    def page(self) -> int:
        return int(self.meta["page"])

    def ref(self) -> str:
        """Ссылка в том формате, который требуем от модели."""
        if self.meta["doc_type"] == "pdf" and not self.page:
            return f"[{self.source}]"          # naive: страница неизвестна
        if self.meta["doc_type"] == "pdf":
            return f"[{self.source}, стр. {self.page}]"
        return f"[{self.meta['title']}]"


def retrieve(query: str, k: int = config.TOP_K,
             cfg: ChunkConfig = config.DEFAULT_CHUNKS) -> list[Hit]:
    """Retriever: топ-K чанков вместе с метаданными."""
    col = client().get_collection(cfg.key)
    res = col.query(query_embeddings=embed([query], config.QUERY_PREFIX), n_results=k,
                    include=["documents", "metadatas", "distances"])
    return [Hit(d, m, round(1 - dist, 4)) for d, m, dist in
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


def all_chunks(cfg: ChunkConfig = config.DEFAULT_CHUNKS) -> list[Chunk]:
    col = client().get_collection(cfg.key)
    got = col.get(include=["documents", "metadatas"])
    return [Chunk(i, d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])]


if __name__ == "__main__":
    print(build(recreate=True))
    for h in retrieve("За сколько дней нужно подавать заявку на отпуск?"):
        print(h.score, h.ref(), h.meta["header"], "|", h.text[:80].replace("\n", " "))
