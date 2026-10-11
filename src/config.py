"""Конфигурация: пути, состав корпуса, параметры поиска и модели.

Имена переменных Yandex Cloud совпадают с reference-проектом курса
(otus-practice-yc-gpt-aug): YC_API_KEY / YC_API_KEY_ID / YC_FOLDER_ID.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
SOURCE_MD = DATA / "source_md"      # исходный корпус преподавателя (markdown)
PDF_DIR = DATA / "pdf"              # собранные PDF-инструкции
WIKI_DIR = DATA / "wiki"            # выгрузка Wiki в HTML
CHROMA_DIR = DATA / "chroma"
OUT_DIR = ROOT / "outputs"
LLM_CACHE = OUT_DIR / "llm_cache"

# --------------------------------------------------------------------------- #
# Состав базы знаний.
#
# Факты целиком взяты из корпуса РобоТех преподавателя
# (github.com/Ilia2704/rag-llamaindex, data/knowledge_base). Своего текста
# не добавлено: меняется только упаковка — markdown-документы собраны в
# PDF-сборники (как настоящие корпоративные инструкции: титул, оглавление,
# колонтитулы, сквозная нумерация) и в HTML-выгрузку Wiki.
# --------------------------------------------------------------------------- #
PDF_BOOKS: dict[str, dict] = {
    "Справочник_сотрудника_HR.pdf": {
        "title": "Справочник сотрудника",
        "subtitle": "Кадровые регламенты и памятки",
        "department": "PeopleOps",
        "chapters": [
            "HR_policy_2026.md", "HR_Remote_Work_2026_Advanced.md",
            "Holiday_Schedule.md", "Doc_Filler_15.md", "Doc_Filler_3.md",
            "Doc_Filler_20.md", "Doc_Filler_8.md", "Doc_Filler_10.md",
            "Doc_Filler_22.md", "Doc_Filler_6.md", "Doc_Filler_18.md",
        ],
    },
    "Регламенты_ИТ_и_безопасности.pdf": {
        "title": "Регламенты ИТ и информационной безопасности",
        "subtitle": "Доступ, пароли, рабочее окружение",
        "department": "Security",
        "chapters": [
            "Security_Passwords.md", "IT_Zero_Trust_2026.md",
            "Doc_Filler_19.md", "Doc_Filler_7.md", "Doc_Filler_17.md",
            "Doc_Filler_5.md", "Doc_Filler_11.md", "Doc_Filler_23.md",
            "Doc_Filler_1.md", "Doc_Filler_13.md", "Doc_Filler_9.md",
            "Doc_Filler_21.md",
        ],
    },
    "Инструкции_по_офису_и_закупкам.pdf": {
        "title": "Инструкции по офису и закупкам",
        "subtitle": "Безопасность, переговорные, оборудование",
        "department": "Operations",
        "chapters": [
            "Doc_Filler_12.md", "Doc_Filler_24.md", "Doc_Filler_4.md",
            "Doc_Filler_16.md", "Doc_Filler_2.md", "Doc_Filler_14.md",
            "Procurement_Device_Catalog_2026.md",
        ],
    },
    "Архив_HR_2024.pdf": {
        "title": "Архив кадровых документов 2024",
        "subtitle": "Только для исторической справки",
        "department": "PeopleOps",
        "chapters": ["HR_policy_2024.md", "HR_Remote_Work_2024_Archive.md"],
    },
}

WIKI_PAGES: dict[str, dict] = {
    "Project_Atlas.html": {"source": "Project_Atlas_Graph_2026.md", "space": "AI Platform"},
    "Meeting_Atlas_Security_2026-03-14.html": {"source": "Meeting_Atlas_Security_2026.md",
                                               "space": "AI Platform"},
    "Project_Red_Eye.html": {"source": "Project_Red_Eye.md", "space": "R&D"},
    "Meeting_Corporate_NewYear_2025.html": {"source": "Meeting_Notes_Corp.md",
                                            "space": "Общее"},
}

# Сколько страниц в начале каждого PDF не имеют печатного номера
# (титул + оглавление). Печатная «стр. 1» = физическая страница FRONT_PAGES + 1.
FRONT_PAGES = 2

# --------------------------------------------------------------------------- #
# Эмбеддинги и поиск
# --------------------------------------------------------------------------- #
EMBED_MODEL = "intfloat/multilingual-e5-base"
# e5 обучен с префиксами: без них качество поиска заметно падает.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
TOP_K = int(os.getenv("TOP_K", "5"))


@dataclass(frozen=True)
class ChunkConfig:
    """Параметры нарезки.

    strategy:
      * "page"  — чанк не пересекает границу страницы и раздела: номер
                  страницы у чанка всегда один и точный;
      * "flow"  — документ склеивается в сплошной текст, страница чанка —
                  та, где он начинается (так режут «наивные» пайплайны);
      * "naive" — PDF целиком через page.get_text() без метаданных страниц:
                  номер страницы модель может взять только из текста
                  (колонтитулы «— 2 —», строки оглавления).
    clean: убирать ли колонтитулы и страницы оглавления.
    """
    chunk_size: int = 400
    chunk_overlap: int = 80
    strategy: str = "page"
    clean: bool = True

    @property
    def key(self) -> str:
        return f"{self.strategy}-{self.chunk_size}-{self.chunk_overlap}-{'clean' if self.clean else 'raw'}"


DEFAULT_CHUNKS = ChunkConfig()

# --------------------------------------------------------------------------- #
# LLM: YandexGPT через OpenAI-совместимый эндпоинт Yandex AI Studio
# --------------------------------------------------------------------------- #
YC_API_KEY = os.getenv("YC_API_KEY")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")
YC_OPENAI_BASE_URL = "https://llm.api.cloud.yandex.net/v1"
YC_MODEL = os.getenv("YC_MODEL", "yandexgpt-lite/latest")
# Низкая температура: ассистенту нужна воспроизводимость, а не креатив.
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
# live   — запросы в API, ответы пишутся в кэш;
# replay — только из кэша (ключ не нужен, ноутбук перезапускается бесплатно).
LLM_MODE = os.getenv("LLM_MODE", "live" if YC_API_KEY else "replay")

# Тариф YandexGPT Lite, синхронный режим: 0,20 ₽ за 1000 токенов, Pro — 1,20 ₽.
PRICE_PER_1K_RUB = {"yandexgpt-lite": 0.20, "yandexgpt": 1.20}


def model_uri(model: str) -> str:
    return f"gpt://{YC_FOLDER_ID}/{model}"


def price_per_1k(model: str) -> float:
    return PRICE_PER_1K_RUB.get(model.split("/")[0], PRICE_PER_1K_RUB["yandexgpt-lite"])
