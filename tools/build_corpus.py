"""Сборка базы знаний: markdown преподавателя → PDF-сборники + HTML-выгрузка Wiki.

    python -m tools.build_corpus

Текст документов не меняется, меняется упаковка. PDF собраны так, как выглядят
настоящие корпоративные инструкции, — со всеми ловушками для парсера:

* титульный лист и оглавление без печатного номера, поэтому печатная «стр. 1»
  стоит на физической странице 3 (и это отражено в PDF PageLabels: i, ii, 1, 2…);
* колонтитулы на каждой странице («РобоТех · Справочник сотрудника», «— 4 —»);
* главы идут сплошным потоком, раздел может начаться внизу одной страницы и
  закончиться на следующей.

Wiki-страницы похожи на экспорт Confluence: шапка с навигацией, «хлебные
крошки», боковая панель со ссылками на соседние статьи, подвал. Полезный текст —
только внутри <article>.
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
import pymupdf
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer)
from reportlab.platypus.tableofcontents import TableOfContents

from src import config
from src.markdown_blocks import MdDoc, inline_html, parse

# DejaVu идёт в комплекте с matplotlib: кириллица есть, сборка переносима.
_FONTS = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(_FONTS / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(_FONTS / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(_FONTS / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-BoldOblique", str(_FONTS / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                              italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique")

# Размеры шрифтов парсер потом использует, чтобы отличить заголовки от текста.
S_BODY = ParagraphStyle("body", fontName="DejaVu", fontSize=10.5, leading=15, spaceAfter=5)
S_LI = ParagraphStyle("li", parent=S_BODY, leftIndent=16, bulletIndent=4, spaceAfter=3)
S_META = ParagraphStyle("meta", parent=S_BODY, fontSize=8.5, leading=11,
                        textColor=colors.HexColor("#666666"), spaceAfter=8)
S_H1 = ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=16, leading=20,
                      spaceBefore=14, spaceAfter=10)
S_H2 = ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=12.5, leading=16,
                      spaceBefore=8, spaceAfter=5)
S_TITLE = ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=26, leading=32,
                         alignment=TA_CENTER, spaceAfter=18)
S_SUB = ParagraphStyle("sub", fontName="DejaVu", fontSize=14, leading=18,
                       alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
S_TOC_HEAD = ParagraphStyle("tochead", parent=S_H1, fontSize=18)

MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 2.2 * cm, 2.4 * cm, 2.2 * cm


class Book(BaseDocTemplate):
    def __init__(self, path: Path, title: str) -> None:
        super().__init__(str(path), pagesize=A4, title=title, author="РобоТех",
                         leftMargin=MARGIN_X, rightMargin=MARGIN_X,
                         topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM)
        self.book_title = title
        frame = Frame(MARGIN_X, MARGIN_BOTTOM, A4[0] - 2 * MARGIN_X,
                      A4[1] - MARGIN_TOP - MARGIN_BOTTOM, id="f")
        self.addPageTemplates([
            PageTemplate("front", [frame]),                    # без колонтитулов
            PageTemplate("body", [frame], onPage=self._decorate),
        ])

    def _decorate(self, canv, doc) -> None:
        printed = doc.page - config.FRONT_PAGES
        w, h = A4
        canv.saveState()
        canv.setFont("DejaVu", 8.5)
        canv.setFillColor(colors.HexColor("#777777"))
        canv.drawString(MARGIN_X, h - 1.4 * cm, f"РобоТех · {self.book_title}")
        canv.drawRightString(w - MARGIN_X, h - 1.4 * cm, "Внутренний документ")
        canv.line(MARGIN_X, h - 1.55 * cm, w - MARGIN_X, h - 1.55 * cm)
        canv.drawCentredString(w / 2, 1.2 * cm, f"— {printed} —")
        canv.restoreState()

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name == "h1":
            # В оглавление попадает печатный номер — как в реальных документах.
            self.notify("TOCEntry", (0, flowable.getPlainText(),
                                     self.page - config.FRONT_PAGES))


def _chapter_flowables(n: int, doc: MdDoc) -> list:
    out: list = [CondPageBreak(5 * cm), Paragraph(f"Глава {n}. {html.escape(doc.title)}", S_H1)]
    if doc.front:
        attrs = " · ".join(f"{k}: {v}" for k, v in doc.front.items() if k != "graph_ready")
        out.append(Paragraph(f"Атрибуты документа: {html.escape(attrs)}", S_META))
    for b in doc.blocks:
        if b.kind == "h2":
            out += [CondPageBreak(2.5 * cm), Paragraph(inline_html(b.text), S_H2)]
        elif b.kind == "li":
            style = ParagraphStyle(f"li{b.level}", parent=S_LI, leftIndent=16 + 14 * b.level,
                                   bulletIndent=4 + 14 * b.level)
            out.append(Paragraph(inline_html(b.text), style, bulletText=b.num))
        else:
            out.append(Paragraph(inline_html(b.text), S_BODY))
    return out


def build_pdf(name: str, spec: dict) -> dict:
    path = config.PDF_DIR / name
    book = Book(path, spec["title"])
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("toc0", fontName="DejaVu", fontSize=11, leading=17,
                                      leftIndent=0, rightIndent=30)]
    story: list = [
        Spacer(1, 7 * cm), Paragraph("РобоТех", S_SUB), Spacer(1, 0.6 * cm),
        Paragraph(spec["title"], S_TITLE), Paragraph(spec["subtitle"], S_SUB),
        Spacer(1, 1.2 * cm), Paragraph(f"Владелец: {spec['department']} · Редакция 2026", S_SUB),
        PageBreak(),
        Paragraph("Содержание", S_TOC_HEAD), toc,
        NextPageTemplate("body"), PageBreak(),
    ]
    for n, md_name in enumerate(spec["chapters"], 1):
        story += _chapter_flowables(n, parse(config.SOURCE_MD / md_name))
    book.multiBuild(story)

    # PageLabels: viewer показывает i, ii, 1, 2… — так же, как напечатано на страницах.
    pdf = pymupdf.open(path)
    pdf.set_page_labels([
        {"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1},
        {"startpage": config.FRONT_PAGES, "prefix": "", "style": "D", "firstpagenum": 1},
    ])
    pdf.set_metadata({**pdf.metadata, "title": spec["title"], "author": "РобоТех"})
    pages = pdf.page_count
    pdf.saveIncr()
    pdf.close()
    return {"file": name, "pages": pages, "chapters": len(spec["chapters"])}


WIKI_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>{title} - РобоТех Wiki</title>
<meta name="wiki:space" content="{space}">
{meta}
<style>
body {{ font-family: sans-serif; margin: 0; color: #172b4d; }}
header, footer {{ background: #f4f5f7; padding: 8px 24px; }}
.layout {{ display: flex; }} aside {{ width: 220px; padding: 16px; background: #fafbfc; }}
article {{ padding: 16px 32px; max-width: 760px; }}
</style>
</head>
<body>
<header>
  <nav><a href="/">РобоТех Wiki</a> | <a href="/spaces">Пространства</a> | <a href="/people">Люди</a> | <a href="/search">Поиск</a></nav>
</header>
<div class="layout">
<aside class="sidebar">
  <h4>Пространство {space}</h4>
  <ul>
{sidebar}
  </ul>
</aside>
<main>
<div class="breadcrumbs">РобоТех Wiki / {space} / {title}</div>
<article id="main-content">
<h1>{title}</h1>
{body}
</article>
</main>
</div>
<footer>
  <p>Экспорт из корпоративной Wiki РобоТех. Страница доступна сотрудникам с учётной записью SSO.</p>
  <p><a href="/help">Справка</a> · <a href="/privacy">Политика конфиденциальности</a></p>
</footer>
</body>
</html>
"""


def _wiki_body(doc: MdDoc) -> str:
    out: list[str] = []
    in_list = False
    for b in doc.blocks:
        if b.kind != "li" and in_list:
            out.append("</ul>")
            in_list = False
        if b.kind == "h2":
            out.append(f"<h2>{inline_html(b.text)}</h2>")
        elif b.kind == "li":
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"  <li>{inline_html(b.text)}</li>")
        else:
            out.append(f"<p>{inline_html(b.text)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def build_wiki() -> list[dict]:
    docs = {name: parse(config.SOURCE_MD / spec["source"])
            for name, spec in config.WIKI_PAGES.items()}
    res = []
    for name, spec in config.WIKI_PAGES.items():
        doc = docs[name]
        # Боковая панель — ссылки на соседние статьи: шум, который парсер обязан выкинуть,
        # иначе в чанк попадают чужие заголовки.
        sidebar = "\n".join(f'    <li><a href="{other}">{html.escape(d.title)}</a></li>'
                            for other, d in docs.items() if other != name)
        meta = "\n".join(f'<meta name="wiki:{k}" content="{html.escape(v)}">'
                         for k, v in doc.front.items())
        page = WIKI_TEMPLATE.format(title=html.escape(doc.title), space=spec["space"],
                                    meta=meta, sidebar=sidebar, body=_wiki_body(doc))
        (config.WIKI_DIR / name).write_text(page, encoding="utf-8")
        res.append({"file": name, "title": doc.title, "blocks": len(doc.blocks)})
    return res


def main() -> None:
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    config.WIKI_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in config.PDF_BOOKS.items():
        print("PDF ", build_pdf(name, spec))
    for r in build_wiki():
        print("WIKI", r)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
