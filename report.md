# Полный анализ структуры программы legale-bot

## Обзор

Legale-bot - это RAG (Retrieval-Augmented Generation) бот для Telegram с поддержкой гибридного поиска (FTS5 + векторный поиск). Проект использует Python 3.11+, SQLite для хранения данных, ChromaDB для векторного хранилища.

**Дата анализа:** 2024

## Структура каталогов

```
legale-bot/
├── legale.py                    # Главная точка входа CLI
├── pyproject.toml               # Конфигурация Poetry
├── README.md                    # Документация
├── src/                         # Исходный код
│   ├── __init__.py
│   ├── adapters/                # Адаптеры для внешних систем
│   │   ├── __init__.py
│   │   ├── embedding/           # Адаптеры для эмбеддингов
│   │   │   ├── __init__.py       # Экспортирует EmbedderAdapter
│   │   │   └── embedder_adapter.py  # ✅ Используется через __init__.py
│   │   ├── llm/                 # Адаптеры для LLM
│   │   │   ├── __init__.py       # Экспортирует LLMAdapter
│   │   │   └── llm_adapter.py    # ✅ Используется через __init__.py
│   │   ├── persistence/         # Адаптеры для SQLite
│   │   │   ├── __init__.py       # Экспортирует SqliteMessageStore, SqliteChunkStore, SqliteFTSIndex
│   │   │   ├── sqlite_message_store.py  # ✅ Используется через __init__.py
│   │   │   ├── sqlite_chunk_store.py    # ✅ Используется через __init__.py
│   │   │   └── sqlite_fts_index.py      # ✅ Используется через __init__.py
│   │   └── vector/              # Адаптеры для векторного хранилища
│   │       ├── __init__.py       # Экспортирует ChromaVectorIndex
│   │       └── chroma_vector_index.py  # ✅ Используется через __init__.py
│   ├── app/                     # Bootstrap и конфигурация приложения
│   │   ├── __init__.py
│   │   ├── bootstrap.py          # ✅ Создание зависимостей (DI)
│   │   └── main_cli.py           # ⚠️ Импортирует отсутствующий use_cases
│   ├── bot/                      # Telegram бот
│   │   ├── __init__.py
│   │   ├── admin.py              # ✅ AdminManager
│   │   ├── admin_commands.py     # ✅ Команды админ-панели
│   │   ├── admin_router.py        # ✅ Роутер команд
│   │   ├── admin_tasks.py         # ✅ Менеджер задач
│   │   ├── cli.py                 # ✅ Интерактивный CLI чат
│   │   ├── command_parser.py     # ✅ Парсер команд
│   │   ├── config.py             # ✅ Конфигурация бота
│   │   ├── core.py                # ✅ Основная логика бота (LegaleBot)
│   │   ├── tgbot.py               # ✅ Telegram webhook daemon
│   │   └── utils/                 # Утилиты бота
│   │       ├── __init__.py        # Экспортирует все утилиты
│   │       ├── access_control.py  # ✅ Используется через __init__.py
│   │       ├── command_validator.py  # ✅ Используется через __init__.py
│   │       ├── database_stats.py  # ✅ Используется через __init__.py
│   │       ├── frequency_controller.py  # ✅ Используется через __init__.py
│   │       ├── health_checker.py   # ✅ Используется через __init__.py
│   │       ├── response_formatter.py  # ✅ Используется через __init__.py
│   │       ├── telegram_common.py # ✅ Используется напрямую
│   │       └── telegram_links.py # ✅ Используется через __init__.py
│   ├── core/                     # Ядро приложения
│   │   ├── __init__.py
│   │   ├── chunk_utils.py         # ⚠️ Не используется напрямую
│   │   ├── cli_parser.py           # ✅ Парсер CLI команд
│   │   ├── dispatcher.py           # ✅ Диспетчер команд
│   │   ├── distance_utils.py       # ❌ НЕ ИСПОЛЬЗУЕТСЯ
│   │   ├── domain.py                # ✅ Доменные модели
│   │   ├── embedding.py            # ✅ Клиент эмбеддингов
│   │   ├── interfaces.py           # ✅ Протоколы/интерфейсы
│   │   ├── llm.py                  # ✅ LLM клиент
│   │   ├── message_search.py       # ✅ Поиск сообщений
│   │   ├── prompt.py               # ✅ Генерация промптов
│   │   └── use_cases/              # ❌ ОТСУТСТВУЕТ (критическая проблема!)
│   │       ├── hybrid_retrieval.py  # ❌ Отсутствует
│   │       ├── search.py            # ❌ Отсутствует
│   │       ├── commands.py         # ❌ Отсутствует
│   │       └── admin_commands.py   # ❌ Отсутствует
│   ├── ingestion/                 # Пайплайн обработки данных
│   │   ├── __init__.py
│   │   ├── chunker.py              # ✅ Разбиение на чанки
│   │   ├── parser.py               # ✅ Парсинг JSON
│   │   ├── pipeline.py             # ✅ Основной пайплайн
│   │   └── telegram.py             # ✅ Загрузка из Telegram
│   ├── lib/                       # Вспомогательные библиотеки
│   │   ├── __init__.py
│   │   ├── argparse2.py            # ❌ НЕ ИСПОЛЬЗУЕТСЯ
│   │   └── syslog2.py             # ✅ Система логирования
│   └── storage/                   # Работа с хранилищами
│       ├── __init__.py
│       ├── db.py                   # ✅ SQLite база данных
│       ├── vector_store.py          # ✅ ChromaDB векторное хранилище
│       └── migrations/              # Миграции БД
│           ├── __init__.py
│           └── drop_legacy_tables.py  # ⚠️ Утилита миграции
├── tests/                         # Тесты
│   ├── __init__.py
│   ├── conftest.py                 # ✅ Конфигурация pytest
│   └── test_*.py                   # ✅ 87 тестовых файлов
├── scripts/                       # Скрипты
│   ├── fix_permissions.sh          # Утилита
│   └── reset_legale_bot_user.sh   # Утилита
└── profiles/                      # Профили (gitignored)
    └── <profile_name>/
        ├── legale_bot.db
        ├── chroma_db/
        └── config.json
```

## Детальный анализ использования файлов

### Корневой уровень

| Файл | Статус | Использование |
|------|--------|---------------|
| `legale.py` | ✅ Используется | Главная точка входа CLI, импортирует `src.core.cli_parser`, `src.lib.syslog2` |

### src/adapters/

Все адаптеры экспортируются через `__init__.py` и используются в `src/app/bootstrap.py`:

- **embedding/embedder_adapter.py** → Используется через `src/adapters/embedding/__init__.py` в `bootstrap.py`
- **llm/llm_adapter.py** → Используется через `src/adapters/llm/__init__.py` в `bootstrap.py`
- **persistence/sqlite_message_store.py** → Используется через `src/adapters/persistence/__init__.py` в `bootstrap.py`
- **persistence/sqlite_chunk_store.py** → Используется через `src/adapters/persistence/__init__.py` в `bootstrap.py`
- **persistence/sqlite_fts_index.py** → Используется через `src/adapters/persistence/__init__.py` в `bootstrap.py`
- **vector/chroma_vector_index.py** → Используется через `src/adapters/vector/__init__.py` в `bootstrap.py`

### src/app/

| Файл | Статус | Использование |
|------|--------|---------------|
| `bootstrap.py` | ✅ Используется | Импортируется в `src/bot/core.py`, создает зависимости |
| `main_cli.py` | ⚠️ Проблема | Импортирует отсутствующий `src.core.use_cases.commands` и `src.core.use_cases.admin_commands` |

### src/bot/

| Файл | Статус | Использование |
|------|--------|---------------|
| `admin.py` | ✅ Используется | Импортируется в `tgbot.py`, `main_cli.py` |
| `admin_commands.py` | ✅ Используется | Импортируется в `tgbot.py` |
| `admin_router.py` | ✅ Используется | Импортируется в `tgbot.py`, `main_cli.py` |
| `admin_tasks.py` | ✅ Используется | Импортируется в `tgbot.py` |
| `cli.py` | ✅ Используется | Вызывается из `legale.py` (cmd_chat) |
| `command_parser.py` | ✅ Используется | Импортируется в `tgbot.py`, `cli.py` |
| `config.py` | ✅ Используется | Импортируется в `core.py`, `legale.py` |
| `core.py` | ✅ Используется | Импортируется в `tgbot.py`, `cli.py`, `main_cli.py` |
| `tgbot.py` | ✅ Используется | Вызывается из `legale.py` (cmd_bot) |

**src/bot/utils/** - все файлы экспортируются через `__init__.py` и используются:
- `access_control.py` → через `__init__.py` в `tgbot.py`, `admin_commands.py`
- `command_validator.py` → через `__init__.py` в тестах
- `database_stats.py` → через `__init__.py` в `admin_commands.py`
- `frequency_controller.py` → через `__init__.py` в `tgbot.py`
- `health_checker.py` → через `__init__.py` в тестах
- `response_formatter.py` → через `__init__.py` в `admin_commands.py`
- `telegram_common.py` → напрямую в `message_search.py`
- `telegram_links.py` → через `__init__.py` в `message_search.py`, `admin_commands.py`

### src/core/

| Файл | Статус | Использование |
|------|--------|---------------|
| `chunk_utils.py` | ⚠️ Не используется напрямую | Возможно используется косвенно |
| `cli_parser.py` | ✅ Используется | Импортируется в `legale.py` |
| `dispatcher.py` | ✅ Используется | Импортируется в `main_cli.py`, `tgbot.py` |
| `distance_utils.py` | ❌ НЕ ИСПОЛЬЗУЕТСЯ | Нигде не импортируется |
| `domain.py` | ✅ Используется | Импортируется в `interfaces.py`, адаптерах |
| `embedding.py` | ✅ Используется | Импортируется в `bootstrap.py`, `core.py`, `pipeline.py` |
| `interfaces.py` | ✅ Используется | Импортируется во всех адаптерах |
| `llm.py` | ✅ Используется | Импортируется в `core.py`, `bootstrap.py` |
| `message_search.py` | ✅ Используется | Импортируется в `tgbot.py` |
| `prompt.py` | ✅ Используется | Импортируется в `core.py` |
| `use_cases/` | ❌ ОТСУТСТВУЕТ | **КРИТИЧЕСКАЯ ПРОБЛЕМА** |

### src/ingestion/

| Файл | Статус | Использование |
|------|--------|---------------|
| `chunker.py` | ✅ Используется | Импортируется в `pipeline.py` |
| `parser.py` | ✅ Используется | Импортируется в `pipeline.py` |
| `pipeline.py` | ✅ Используется | Вызывается из `legale.py` (cmd_ingest) |
| `telegram.py` | ✅ Используется | Вызывается из `legale.py` (cmd_telegram) |

### src/lib/

| Файл | Статус | Использование |
|------|--------|---------------|
| `argparse2.py` | ❌ НЕ ИСПОЛЬЗУЕТСЯ | Нигде не импортируется |
| `syslog2.py` | ✅ Используется | Импортируется почти везде |

### src/storage/

| Файл | Статус | Использование |
|------|--------|---------------|
| `db.py` | ✅ Используется | Импортируется в `bootstrap.py`, `core.py`, `pipeline.py` |
| `vector_store.py` | ✅ Используется | Импортируется в `bootstrap.py`, `core.py`, `pipeline.py` |
| `migrations/drop_legacy_tables.py` | ⚠️ Утилита | Используется для миграций |

## Критические проблемы

### 1. Отсутствует каталог `src/core/use_cases/`

**Проблема:** Каталог `src/core/use_cases/` не существует, но импортируется в следующих файлах:

- `src/bot/core.py` → `from src.core.use_cases.hybrid_retrieval import HybridRetrievalService`
- `src/core/message_search.py` → `from src.core.use_cases.hybrid_retrieval import HybridRetrievalService`
- `src/app/main_cli.py` → `from src.core.use_cases.commands import ...` и `from src.core.use_cases.admin_commands import ...`
- `src/app/bootstrap.py` → `from src.core.use_cases.search import HybridSearch` и `from src.core.use_cases.hybrid_retrieval import HybridRetrievalService`
- `tests/test_hybrid_search_unit.py` → `from src.core.use_cases.search import HybridSearch`
- `tests/test_fts_retrieval.py` → `from src.core.use_cases.hybrid_retrieval import HybridRetrievalService`
- `tests/test_build_fts5_queries.py` → `from src.core.use_cases.hybrid_retrieval import build_fts5_queries`

**Требуемые файлы:**
- `src/core/use_cases/__init__.py`
- `src/core/use_cases/hybrid_retrieval.py` (содержит `HybridRetrievalService`, `build_fts5_queries`)
- `src/core/use_cases/search.py` (содержит `HybridSearch`)
- `src/core/use_cases/commands.py` (содержит `StartCommandHandler`, `HelpCommandHandler`, `ResetCommandHandler`, `TokensCommandHandler`, `ModelCommandHandler`, `FindCommandHandler`)
- `src/core/use_cases/admin_commands.py` (содержит `AdminSetCommandHandler`, `AdminGetCommandHandler`, `AdminCommandHandler`)

**Влияние:** Приложение не может запуститься из-за отсутствующих модулей.

### 2. Отсутствует модуль `src/core/retrieval.py`

**Проблема:** Модуль `src.core.retrieval` не существует, но импортируется в тестах:

- `tests/test_retrieval.py` → `from src.core.retrieval import RetrievalService`
- `tests/test_retrieval_debug.py` → `from src.core.retrieval import RetrievalService`
- `tests/test_message_search.py` → `from src.core.retrieval import RetrievalService`

**Статус:** Судя по комментариям в коде (`# RetrievalService removed - legacy RAG code`), это устаревший модуль, который был заменен на `HybridRetrievalService`. Тесты требуют обновления.

### 3. Отсутствует модуль `src/ai/clustering.py`

**Проблема:** Модуль `src.ai.clustering` не существует, но импортируется:

- `legale.py` → `from src.ai.clustering import TopicClusterer`
- `tests/test_clustering.py` → `from src.ai.clustering import TopicClusterer`

**Статус:** Судя по комментариям в коде, кластеризация была удалена (`# clustering is deprecated`). Команды topics в `legale.py` также отключены.

## Неиспользуемые файлы

### Полностью неиспользуемые

1. **`src/lib/argparse2.py`**
   - Статус: ❌ Не используется
   - Описание: Альтернативный парсер аргументов, заменен на `cli_parser.py`
   - Рекомендация: Удалить или переместить в архив

2. **`src/core/distance_utils.py`**
   - Статус: ❌ Не используется
   - Описание: Утилиты для конвертации distance/similarity
   - Рекомендация: Удалить или использовать в векторном поиске

### Потенциально неиспользуемые

3. **`src/core/chunk_utils.py`**
   - Статус: ⚠️ Не используется напрямую
   - Описание: Утилиты для работы с чанками
   - Рекомендация: Проверить косвенное использование

4. **`src/storage/migrations/drop_legacy_tables.py`**
   - Статус: ⚠️ Утилита миграции
   - Описание: Скрипт для удаления устаревших таблиц
   - Рекомендация: Оставить как утилиту для миграций

## Устаревшие тесты

Следующие тесты ссылаются на удаленные модули:

1. **`tests/test_retrieval.py`** - использует `src.core.retrieval.RetrievalService` (удален)
2. **`tests/test_retrieval_debug.py`** - использует `src.core.retrieval.RetrievalService` (удален)
3. **`tests/test_message_search.py`** - использует `src.core.retrieval.RetrievalService` (удален)
4. **`tests/test_clustering.py`** - использует `src.ai.clustering.TopicClusterer` (удален)
5. **`tests/test_bot_core_models.py`** - мокирует `src.bot.core.RetrievalService` (удален)
6. **`tests/test_bot_core_chat.py`** - мокирует `src.bot.core.RetrievalService` (удален)
7. **`tests/test_bot_core_context.py`** - мокирует `src.bot.core.RetrievalService` (удален)

**Рекомендация:** Обновить тесты для использования `HybridRetrievalService` вместо `RetrievalService`.

## Архитектура зависимостей

### Основные потоки данных

1. **CLI Entry Point:**
   ```
   legale.py
   ├── src.core.cli_parser (парсинг команд)
   ├── src.lib.syslog2 (логирование)
   └── Маршрутизация команд:
       ├── cmd_ingest → src.ingestion.pipeline
       ├── cmd_bot → src.bot.tgbot
       ├── cmd_chat → src.bot.cli
       └── cmd_telegram → src.ingestion.telegram
   ```

2. **Bot Core:**
   ```
   src.bot.core (LegaleBot)
   ├── src.app.bootstrap (создание зависимостей)
   │   ├── src.adapters.* (все адаптеры)
   │   └── src.core.use_cases.* (❌ отсутствует!)
   ├── src.core.embedding
   ├── src.core.llm
   └── src.core.prompt
   ```

3. **Ingestion Pipeline:**
   ```
   src.ingestion.pipeline
   ├── src.ingestion.parser
   ├── src.ingestion.chunker
   ├── src.storage.db
   ├── src.storage.vector_store
   └── src.core.embedding
   ```

## Рекомендации

### Критичные (блокируют работу)

1. **Создать каталог `src/core/use_cases/`** с требуемыми модулями:
   - `hybrid_retrieval.py` - основной сервис гибридного поиска
   - `search.py` - базовый поиск
   - `commands.py` - обработчики команд
   - `admin_commands.py` - обработчики админ-команд

2. **Исправить импорты** в `src/app/main_cli.py` или создать недостающие модули

### Важные (влияют на тесты)

3. **Обновить тесты** для использования `HybridRetrievalService` вместо `RetrievalService`

4. **Удалить или обновить** тесты кластеризации (`test_clustering.py`)

### Опциональные (очистка кода)

5. **Удалить неиспользуемые файлы:**
   - `src/lib/argparse2.py`
   - `src/core/distance_utils.py` (или интегрировать в векторный поиск)

6. **Проверить использование:**
   - `src/core/chunk_utils.py` - возможно используется косвенно

## Статистика

- **Всего Python файлов:** 58
- **Файлов с импортами из src:** 32
- **Критических проблем:** 1 (отсутствует use_cases)
- **Неиспользуемых файлов:** 2
- **Устаревших тестов:** 7

## Заключение

Проект имеет хорошо структурированную архитектуру с четким разделением на слои (adapters, core, bot, ingestion, storage). Основная проблема - отсутствие критически важного каталога `src/core/use_cases/`, который импортируется в нескольких ключевых модулях. Это блокирует запуск приложения.

Рекомендуется:
1. Немедленно создать недостающий каталог `use_cases` с требуемыми модулями
2. Обновить устаревшие тесты
3. Удалить неиспользуемые файлы для упрощения кодовой базы

