---
year: 2026
category: Projects
department: AI_Platform
doc_type: meeting_notes
access_group: engineering
graph_ready: true
---
# Встреча Atlas и Security по графовому RAG

Дата: 2026-03-14.

Участники: Иван Петров, Мария Соколова, Анна Ким, команда Security, команда AI Platform.

Решения:
- Для графового поиска используется Neo4j PropertyGraphIndex.
- Для точных кодов политик используется hybrid search с sparse-сигналом BM25.
- Для финального порядка источников используется Cross-Encoder Reranking.
- Для вопросов про текущие HR-правила обязателен metadata filter year=2026 и category=HR.

Иван Петров должен подготовить демонстрацию Graph Retrieval для вопроса: "Кто отвечает за проект Atlas и какие системы он использует?"

Мария Соколова должна проверить корректность ответа по политике HR-REMOTE-2026.

