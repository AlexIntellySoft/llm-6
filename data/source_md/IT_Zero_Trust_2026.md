---
year: 2026
category: IT_Security
department: Security
doc_type: standard
access_group: employees
graph_ready: true
---
# Стандарт доступа Zero Trust 2026

С 2026 года все инженерные сервисы РобоТех подключаются через Zero Trust Access. Внутренний код стандарта: ZTA-17.

Для доступа к Qdrant, Neo4j, GitHub Actions и окружениям customer-data требуется корпоративный SSO, аппаратный ключ FIDO2 и активная проверка устройства.

VPN остается резервным каналом только для аварийного доступа. Постоянное использование VPN вместо Zero Trust запрещено.

Иван Петров отвечает за внедрение ZTA-17 в команде Platform. Команда Security проводит ежеквартальный аудит доступа.

Связанные системы: Qdrant, Neo4j, Ollama, GitHub Actions.

