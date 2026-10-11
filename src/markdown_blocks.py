"""Минимальный разбор markdown корпуса РобоТех в список блоков.

Корпус неоднороден (документы писались разными генераторами): заголовки
встречаются как `# ...`, как setext-подчёркивания `====`, как строка целиком
в `**жирном**`. Здесь всё это сводится к трём уровням: title / h2 / p / li.
Нужен только для сборки PDF и Wiki из исходников — RAG-пайплайн этот модуль
не использует, он читает уже готовые PDF и HTML.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Block:
    kind: str          # title | h2 | p | li
    text: str          # inline-разметка: **жирный**, _курсив_
    num: str = ""      # маркер пункта списка: "1." или "•"
    level: int = 0     # вложенность пункта списка


@dataclass
class MdDoc:
    name: str
    title: str
    front: dict = field(default_factory=dict)
    blocks: list[Block] = field(default_factory=list)


_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)
_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_BOLD_LINE = re.compile(r"^\*\*(.+?)\*\*:?$")
_OL = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
_UL = re.compile(r"^(\s*)[-*]\s+(.*)$")


def parse(path: Path) -> MdDoc:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip() + "\n"
    front: dict = {}
    m = _FRONT.match(text)
    if m:
        for line in m.group(1).splitlines():
            k, _, v = line.partition(":")
            front[k.strip()] = v.strip()
        text = text[m.end():]

    lines = text.split("\n")
    blocks: list[Block] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(Block("p", " ".join(s.strip() for s in para)))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if not line.strip():
            flush()
        elif re.fullmatch(r"-{3,}|\*{3,}", line.strip()):
            flush()                                    # горизонтальная линия
        elif nxt and re.fullmatch(r"[=]{3,}|[-]{3,}", nxt):
            flush()                                    # setext-заголовок
            blocks.append(Block("h2", line.strip()))
            i += 1
        elif m := _ATX.match(line.strip()):
            flush()
            blocks.append(Block("h2", m.group(2)))
        elif (m := _BOLD_LINE.match(line.strip())) and not para:
            flush()
            blocks.append(Block("h2", m.group(1)))
        elif m := _OL.match(line):
            flush()
            blocks.append(Block("li", m.group(3), f"{m.group(2)}.", len(m.group(1)) // 3))
        elif m := _UL.match(line):
            flush()
            blocks.append(Block("li", m.group(2), "•", len(m.group(1)) // 3))
        elif line.startswith(" ") and blocks and blocks[-1].kind == "li" and not para:
            blocks[-1].text += " " + line.strip()     # продолжение пункта списка
        else:
            para.append(line)
        i += 1
    flush()

    # Первый заголовок документа становится его названием.
    title = path.stem
    for j, b in enumerate(blocks):
        if b.kind == "h2":
            title = b.text
            del blocks[j]
            break
    return MdDoc(path.name, _plain(title), front, blocks)


def _plain(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", s)
    return s.strip(" *:")


def inline_html(s: str) -> str:
    """markdown-инлайн → безопасный HTML (подходит и для reportlab Paragraph)."""
    s = html.escape(s, quote=False)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    return s
