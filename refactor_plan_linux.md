# План рефакторинга Legale Bot (Linux Philosophy)

## Раздел 1. Краткий обзор проблем

Текущая архитектура страдает от классического "over-engineering". 
Поток выполнения операций (например, "загрузить сообщения из файла" или "выполнить админскую команду") размазан по 3-4 слоям: CLI -> App/Bot -> Core UseCase -> Interface -> Adapter -> Implementation. 
Это заставляет прыгать по файлам, чтобы понять простую логику.
Каталог `src/adapters` содержит множество файлов-пустышек (`embedder_adapter.py`, `llm_adapter.py`), которые лишь делегируют вызовы, не добавляя ценности.
Слой `src/core/ingest_use_cases` имитирует "чистую архитектуру", но по факту усложняет простой ETL-скрипт.
Логика админских команд дублируется и размазана между `src/app/admin_commands.py`, `src/bot/admin_commands.py` и `src/core/commands/admin.py`, создавая путаницу ("где *на самом деле* выполняется команда?").
В `legale.py` (1000+ строк) смешаны CLI-аргументы, инициализация и бизнес-логика.

Цель рефакторинга: сделать код "плоским". Если функция `ingest` нужна, она должна лежать в `ingest.py` и делать работу, а не вызывать `UseCase`, который вызывает `Service`, который вызывает `Adapter`.

## Раздел 2. План рефакторинга

### 1. Уплощение структуры и удаление "прокладок" (Adapters & Interfaces)

- [x] file=src/adapters/embedding/embedder_adapter.py func=EmbedderAdapter type=inline HIGH RISK
  - **Действие**: Удалить класс-обертку. Использовать `EmbeddingClient` и `LocalEmbeddingClient` напрямую (или через единый фабричный метод).
  - **Цель**: `src/core/embedding.py` (или просто `src/embedding.py`) должен содержать всю логику эмбеддинга.

- [x] file=src/adapters/llm/llm_adapter.py func=LLMAdapter type=inline
  - **Действие**: Удалить. Перенести логику адаптации (если есть) прямо в `src/core/llm.py` (или `src/llm.py`).

- [x] file=src/adapters/persistence/* type=merge target=src/storage/sqlite.py
  - **Действие**: Перенести `sqlite_chunk_store.py`, `sqlite_message_store.py`, `sqlite_fts_index.py` в `src/storage` (или `src/db`).
  - **Детали**: Вместо кучи мелких файлов "store" сделать один модуль `src/db/sqlite.py` (или `storage.py`), где явно видны все операции с базой.
  - **Примечание**: Удалить саму директорию `src/adapters`.

- [x] file=src/storage/vector_store.py file=src/adapters/vector/chroma_vector_index.py type=merge target=src/storage/vector.py
  - **Действие**: Слить реализацию ChromaDB и абстракцию в один файл `src/storage/vector.py`.
  - **Обоснование**: У нас одна векторная БД. Интерфейс не нужен, если реализация единственная и жестко привязана к проекту.

### 2. Удаление Use Cases и упрощение Ingestion

- [x] file=src/core/ingest_use_cases/ingest_messages.py type=functionize target=src/ingestion/actions.py
  - **Действие**: Превратить класс `IngestMessages` в простую функцию `ingest_messages_from_file(...)`.
  - **Цель**: Убрать папку `src/core/ingest_use_cases`. Вся логика инжеста должна быть в `src/ingestion/`.

- [x] file=src/core/ingest_use_cases/* type=delete
  - **Действие**: Удалить остальные use-case классы, перенеся логику в функции внутри `src/ingestion/pipeline.py` или `src/ingestion/actions.py`.

- [ ] file=src/ingestion/pipeline.py type=simplify
  - **Действие**: Убрать лишние абстракции. Пайплайн должен быть простым скриптом, вызывающим функции шагов последовательно.

### 3. Консолидация логики бота и команд

- [ ] file=src/app/admin_commands.py file=src/bot/admin_commands.py file=src/core/commands/admin.py type=merge target=src/bot/admin.py HIGH RISK
  - **Действие**: Собрать ВСЮ логику админских команд в один файл `src/bot/admin.py`.
  - **Проблема**: Сейчас `src/app/admin_commands.py` (56KB) - это "монстр", который лежит не там. `core/commands/admin.py` - пустышка.
  - **Результат**: Один файл, где определены все хендлеры `/admin`.

- [x] file=src/bot/tgbot.py type=simplify
  - **Действие**: Вынести часть логики (например, `RuntimeContext`, глобальные состояния) в `src/bot/context.py` или `src/state.py`, если `tgbot.py` останется слишком большим. Но приоритет - простота. Если файл читается линейно, размер допустим.

### 4. Реструктуризация Core и Utils

- [ ] file=src/core/interfaces.py type=prune
  - **Действие**: Удалить интерфейсы, у которых ровно одна реализация и которые не помогают читать код. Оставить только те, что реально нужны для подмены (например, если планируется support нескольких LLM, но реализация переключается конфигом, а не DI-контейнером).

- [ ] file=src/bot/utils/* type=merge target=src/utils.py
  - **Действие**: Слить мелкие утилиты (`telegram_links.py`, `error_handler.py`, и т.д.) в один модуль `src/lib/utils.py` или `src/utils.py`, если они не привязаны жестко к боту.
  - **Философия**: Меньше файлов -> проще искать.

- [ ] file=legale.py type=simplify
  - **Действие**: Убрать реализацию команд из `legale.py`. Сделать его чистым диспетчером: "распарсил аргументы -> вызвал функцию из `src/cli/` или `src/main.py`".

### 5. Итоговая структура (Target State)

```text
src/
  bot/          # Логика Telegram бота
    admin.py    # ВСЕ админские команды
    server.py   # Webhook/Polling (бывший tgbot.py)
    handlers.py # Обработчики сообщений
  ingest/       # ETL пайплайн
    pipeline.py
    parser.py
    chunker.py
  storage/      # Работа с данными
    sqlite.py   # SQL база (сообщения, чанки)
    vector.py   # ChromaDB
  lib/          # Общие библиотеки
    llm.py      # Клиенты к LLM
    embedding.py # Клиенты эмбеддингов
    utils.py    # Утилиты
  config.py     # Конфигурация и профили
```
