"""Парсинг базы знаний с метаданными: PDF (PyMuPDF) и HTML-выгрузка Wiki (BeautifulSoup).

Результат — список Segment: непрерывный кусок текста одного раздела на одной
странице. Номер страницы, печатная метка и заголовок раздела берутся из
структуры документа, а не из текста, и дальше «приклеены» к каждому чанку.

Что делает PDF-парсер сверх page.get_text():

* читает текст построчно со шрифтами (get_text("dict")) и по размеру/жирности
  отличает заголовки глав и разделов от основного текста — так появляется
  поле header;
* выкидывает колонтитулы по геометрии: строки в верхних и нижних 7% страницы
  («РобоТех · Справочник сотрудника», «— 4 —»);
* пропускает титул и оглавление: на странице оглавления номера страниц стоят
  рядом с названиями глав и сбивают и поиск, и модель;
* заголовок раздела переносится через границу страницы: если раздел начался
  на стр. 4 и продолжился на стр. 5, у текста со стр. 5 header тот же.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup

from src import config

MARGIN_ZONE = 0.07          # доля высоты страницы под колонтитулы сверху и снизу
H1_MIN_SIZE = 15.0          # глава: 16 pt bold
H2_MIN_SIZE = 12.0          # раздел: 12.5 pt bold; основной текст 10.5 pt
TOC_TITLES = {"содержание", "оглавление"}


@dataclass
class Segment:
    source: str              # имя файла: "Справочник_сотрудника_HR.pdf" / "Project_Atlas.html"
    doc_type: str            # pdf | wiki
    title: str               # название сборника или Wiki-статьи
    page: int                # физическая страница PDF, с 1; у Wiki — 0
    page_label: str          # напечатанный на странице номер ("2"), у титула "i"
    chapter: str             # глава (для Wiki — название статьи)
    header: str              # ближайший заголовок раздела
    text: str

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    y0: float
    y1: float


def _blocks(page: pymupdf.Page) -> list[list[_Line]]:
    """Текстовые блоки страницы как списки строк со шрифтовыми признаками."""
    out = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        if block.get("type") != 0:
            continue
        lines = []
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            size = max(s["size"] for s in spans)
            bold = all("Bold" in s["font"] for s in spans)
            lines.append(_Line(text, size, bold, line["bbox"][1], line["bbox"][3]))
        if lines:
            out.append(lines)
    return out


def _is_toc_page(blocks: list[list[_Line]]) -> bool:
    first = next((ln for b in blocks for ln in b if ln.size >= H2_MIN_SIZE), None)
    if first and first.text.strip().lower() in TOC_TITLES:
        return True
    lines = [ln.text for b in blocks for ln in b]
    numbered = sum(bool(re.search(r"\s\d{1,3}$|^\d{1,3}$", t)) for t in lines)
    return len(lines) > 5 and numbered / len(lines) > 0.4


def parse_pdf(path: Path, clean: bool = True) -> list[Segment]:
    doc = pymupdf.open(path)
    title = doc.metadata.get("title") or path.stem
    segments: list[Segment] = []
    chapter, header = "", ""

    for pno in range(doc.page_count):
        page = doc[pno]
        label = page.get_label() or str(pno + 1)
        h = page.rect.height
        blocks = _blocks(page)
        # Титул и оглавление: либо метка страницы не число (i, ii), либо страница
        # похожа на оглавление (для PDF без PageLabels).
        if clean and (not label.isdigit() or _is_toc_page(blocks)):
            continue

        buf: list[str] = []

        def flush() -> None:
            text = "\n".join(buf).strip()
            if text:
                segments.append(Segment(path.name, "pdf", title, pno + 1, label,
                                        chapter, header, text))
            buf.clear()

        for lines in blocks:
            if clean:
                lines = [ln for ln in lines
                         if ln.y0 > h * MARGIN_ZONE and ln.y1 < h * (1 - MARGIN_ZONE)]
                if not lines:
                    continue
            first = lines[0]
            text = " ".join(ln.text for ln in lines)
            if first.bold and first.size >= H1_MIN_SIZE:
                flush()
                chapter = header = re.sub(r"^Глава \d+\.\s*", "", text)
            elif first.bold and first.size >= H2_MIN_SIZE:
                flush()
                header = text
            else:
                buf.append(_join_wrapped(lines))
        flush()
    doc.close()
    return segments


def _join_wrapped(lines: list[_Line]) -> str:
    """Склеивает строки одного абзаца, снимая переносы по дефису."""
    out = ""
    for ln in lines:
        if out.endswith("-") and ln.text[:1].islower():
            out = out[:-1] + ln.text
        else:
            out = f"{out} {ln.text}" if out else ln.text
    return out


def naive_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Базовый вариант для сравнения: page.get_text() как есть, страница целиком."""
    with pymupdf.open(path) as doc:
        return [(i + 1, p.get_text()) for i, p in enumerate(doc)]


# --------------------------------------------------------------------------- #
# HTML (Wiki)
# --------------------------------------------------------------------------- #
NOISE_TAGS = ["script", "style", "header", "nav", "aside", "footer"]


def parse_html(path: Path, clean: bool = True) -> list[Segment]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    article = soup.find("article") or soup.body
    h1 = article.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else soup.title.get_text().split(" - ")[0]

    if not clean:
        # Без чистки: весь видимый текст страницы, включая меню и боковую панель.
        text = soup.body.get_text("\n", strip=True)
        return [Segment(path.name, "wiki", title, 0, "", title, "", text)]

    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for tag in soup.select(".breadcrumbs"):
        tag.decompose()

    segments: list[Segment] = []
    header, buf = title, []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            segments.append(Segment(path.name, "wiki", title, 0, "", title, header, text))
        buf.clear()

    for el in article.find_all(["h2", "h3", "p", "li"]):
        txt = el.get_text(" ", strip=True)
        if el.name in ("h2", "h3"):
            flush()
            header = txt
        elif txt:
            buf.append(("• " if el.name == "li" else "") + txt)
    flush()
    return segments


def wiki_meta(path: Path) -> dict:
    """Атрибуты страницы из <meta name="wiki:*">: год, подразделение, группа доступа."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    return {m["name"].split(":", 1)[1]: m["content"]
            for m in soup.find_all("meta", attrs={"name": re.compile(r"^wiki:")})}


# --------------------------------------------------------------------------- #
def load_corpus(clean: bool = True) -> list[Segment]:
    segs: list[Segment] = []
    for name in config.PDF_BOOKS:
        segs += parse_pdf(config.PDF_DIR / name, clean=clean)
    for name in config.WIKI_PAGES:
        segs += parse_html(config.WIKI_DIR / name, clean=clean)
    return segs


def naive_corpus() -> list[Segment]:
    """Базовая линия: каждый PDF — один сплошной текст из page.get_text(), без страниц.

    Колонтитулы, титул и оглавление остаются в тексте; Wiki — как обычно."""
    segs = []
    for name in config.PDF_BOOKS:
        path = config.PDF_DIR / name
        text = "\n".join(t for _, t in naive_pdf_pages(path))
        segs.append(Segment(name, "pdf", config.PDF_BOOKS[name]["title"], 0, "", "", "", text))
    for name in config.WIKI_PAGES:
        segs += parse_html(config.WIKI_DIR / name)
    return segs


def pdf_page_count(source: str) -> int:
    with pymupdf.open(config.PDF_DIR / source) as d:
        return d.page_count


def wiki_titles() -> dict[str, str]:
    """Название Wiki-статьи → имя файла (по названию модель и ссылается)."""
    return {parse_html(config.WIKI_DIR / n)[0].title: n for n in config.WIKI_PAGES}
