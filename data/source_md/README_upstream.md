# RoboTech Knowledge Base

`data/knowledge_base/` содержит готовый корпус markdown-документов для RAG-демо.

Тематика общая: внутренняя база знаний вымышленной компании РобоТех.
Внутри есть несколько доменов, чтобы retrieval-задачи были более реалистичными:

- `HR`: гибридная работа, отпуска, архивные и актуальные политики;
- `IT_Security`: пароли, VPN, Zero Trust Access;
- `Projects`: проектные документы и встречи;
- `Procurement`: закупка устройств и требования к рабочим станциям;
- `General`: общие корпоративные инструкции и filler-документы.

Часть документов содержит front matter metadata: `year`, `category`, `department`,
`doc_type`, `access_group`, `graph_ready`. LlamaIndex-загрузчик читает эти поля и
использует их для metadata filtering и graph demo.

