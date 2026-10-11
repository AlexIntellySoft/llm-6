# База знаний

| папка | что внутри | откуда |
| --- | --- | --- |
| `source_md/` | 36 markdown-документов вымышленной компании РобоТех | материалы урока: [Ilia2704/rag-llamaindex](https://github.com/Ilia2704/rag-llamaindex), `data/knowledge_base`, коммит `c3aff8e` (2026-10-07). Текст не изменён; исходное описание корпуса — `source_md/README_upstream.md` |
| `pdf/` | 4 PDF-сборника | собраны из `source_md` скриптом `python -m tools.build_corpus` |
| `wiki/` | 4 HTML-страницы (экспорт Wiki) | собраны тем же скриптом |
| `chroma/` | векторная база (в git не попадает) | `python -m src.index` |

Какой документ в какой сборник попал — `PDF_BOOKS` и `WIKI_PAGES` в `src/config.py`.
