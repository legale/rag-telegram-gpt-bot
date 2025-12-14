# План рефакторинга: Core + Interfaces + Adapters

Этот документ описывает план перехода архитектуры проекта `legale-bot` на принципы Clean Architecture (Core + Interfaces + Adapters).

## 1. Инвентаризация и декомпозиция домена

### Доменные сущности (Entities)
Эти объекты живут в `src/core/domain/` и не зависят от БД.

*   **Message**: id, chat_id, from_id, text, timestamp, meta (dict).
*   **Chunk**: id, text, msg_ids (start/end), valid_period (ts_from/to), embedding (optional), metadata (topic_ids).
*   **SearchResult**: chunk, score, original_messages (list), topics (list).
*   **IngestionJob**: status, stage, stats.
*   **ProfileConfig**: model_name, embedding_provider, paths.

### Слои
1.  **Core (Use Cases)**: Бизнес-логика (Ingest, Search, Command handling). Зависит только от Interfaces.
2.  **Interfaces (Ports)**: Абстракции (Protocols) для Store, Vector, LLM.
3.  **Adapters**: Реализации интерфейсов (SQLite, Chroma, OpenAI).
4.  **App (Composition)**: Точка входа, DI контейнер, связывание адаптеров и юзкейсов.

### Анализ связности (Current State)
*   **Tight Coupling**: `IngestionPipeline` жестко импортирует `Database` (SQLite) и `VectorStore` (ChromaWrappers).
*   **Leaking Abstractions**: `RAGSearch` знает про `ChromaDB` collection structure.
*   **Mixed Logic**: `tgbot.py` содержит логику поиска (`handle_find_command`) и форматирования.

## 2. Целевые интерфейсы (Interfaces)

Расположение: `src/core/interfaces/`

### Persistence
```python
class MessageStore(Protocol):
    def save_batch(self, messages: List[Message]) -> int: ...
    def get_by_chat(self, chat_id: str, limit: int, offset: int) -> List[Message]: ...
    def get_context(self, chat_id: str, time_point: datetime, window_sec: int) -> List[Message]: ...
    def count(self) -> int: ...

class ChunkStore(Protocol):
    def save_batch(self, chunks: List[Chunk]) -> int: ...
    def get_by_ids(self, ids: List[str]) -> List[Chunk]: ...
    def update_topics(self, updates: Dict[str, TopicUpdate]) -> None: ... # chunk_id -> topic info
    def clear(self) -> None: ...
```

### Retrieval & AI
```python
class VectorIndex(Protocol):
    def upsert(self, items: List[VectorDoc]) -> None: ... # VectorDoc: id, vector, meta
    def query(self, vector: List[float], top_k: int, filter: Dict = None) -> List[ScoredDoc]: ...
    def delete(self, ids: List[str]) -> None: ...

class Embedder(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...
    # Implementation handles batching and local/remote switching internally OR via adapters

class LLM(Protocol):
    def complete(self, prompt: str, system: str = None, **kwargs) -> str: ...
    # Streaming interface can be added later
```

### Infrastructure
```python
class ConfigProvider(Protocol):
    def get_profile_config(self, profile_name: str) -> ProfileConfig: ...

class TransactionManager(Protocol):
    # Context manager for atomic operations across stores
    def atomic(self): ...
```

## 3. Core Use Cases

Расположение: `src/core/use_cases/`

### Ingestion
*   **`IngestMessages`**:
    *   Input: `SourceIterator` (provides raw messages)
    *   Logic: Parse -> `MessageStore.save_batch`
*   **`ProcessChunks`**:
    *   Input: None (pulls from Store) or triggering event
    *   Logic: `MessageStore.iterate` -> Chunker -> `ChunkStore.save` -> `Embedder` -> `VectorIndex.upsert`
*   **`PipelineOrchestrator`**:
    *   Replaces current `IngestionPipeline`. Calls simpler use cases sequentially.

### Search
*   **`HybridSearch`**:
    *   Input: query, threshold, limits
    *   Logic:
        1.  `Embedder.embed_query(query)`
        2.  `VectorIndex.query(vec)` -> `candidates` (ids)
        3.  `ChunkStore.get_by_ids(candidates)` -> `chunks`
        4.  `MessageStore.get_context(...)` (enrichment)
        5.  Pack result into `SearchResult` DTO.

### Command Dispatching
*   **`CommandDispatcher`**:
    *   Input: `CommandContext` (user_id, chat_id, command_name, args)
    *   Logic: Switch on command name -> execute specific Use Case -> Return `CommandResult` (status, data/text).
    *   Этот слой не знает про Telegram API или `sys.argv`.

## 4. Adapters

Расположение: `src/adapters/`

### Persistence (SQLAlchemy)
*   **`SqliteMessageStore`**: Implements `MessageStore`. Maps domain `Message` to ORM `MessageModel`.
*   **`SqliteChunkStore`**: Implements `ChunkStore`. Maps domain `Chunk` to ORM `ChunkModel`.

### Vector (Chroma)
*   **`ChromaVectorIndex`**: Implements `VectorIndex`. Wraps `chromadb.Client`. Handles collection logic.

### AI
*   **`LocalEmbedderAdapter`**: Wraps `sentence-transformers`.
*   **`RemoteEmbedderAdapter`**: Wraps OpenAI/OpenRouter API.
*   **`OpenAILLMAdapter`**: Wraps `openai` client.

### Interface Adapters (Inbound)
*   **`TelegramHandler`**:
    *   Received `Update` -> parse to `CommandContext` -> `Dispatcher.dispatch` -> Format result to Telegram Message -> Send.
*   **`CLIHandler`**:
    *   `sys.argv` -> parse to `CommandContext` -> `Dispatcher.dispatch` -> Print result.

## 5. Структура директорий

```
src/
├── core/                  # Pure Python, no framework deps
│   ├── domain/            # Data classes (Message, Chunk)
│   ├── interfaces/        # Protocols
│   ├── use_cases/         # Ingest, Search, Commands
│   └── errors.py          # Domain exceptions
│   └── argparse2.py       # arguments parsing (argparse2)
│   └── syslog2.py         # Logging (syslog2)
├── adapters/              # Implementations
│   ├── persistence/       # SQLAlchemy models & stores
│   ├── vector/            # ChromaDB adapter
│   ├── embedding/         # Local/Remote embedders
│   ├── llm/               # LLM clients
│   └── telegram/          # Telegram bot specific logic
├── app/                   # Composition root
│   ├── config.py
│   ├── bootstrap.py       # DI container / wiring
│   └── main_cli.py        # Entry point for CLI
└── lib/                   # Shared utilities (syslog2, text tools)
```

## 6. План миграции (Инкрементальный)

### Шаг 1: Foundation
Создать структуру папок. Определить `core/domain` и `core/interfaces`.
*   Files: `src/core/domain.py`, `src/core/interfaces.py`, `src/core/argparse2.py`, `src/core/syslog2.py`.
*   Risk: None. Code addition only.

### Шаг 2: Adapters Wrapping
Обернуть текущий код (`db.py`, `vector_store.py`) в адаптеры, реализующие интерфейсы.
*   Files: `src/adapters/persistence/sqlite_impl.py`, `src/adapters/vector/chroma_impl.py`.
*   Logic: Просто делегирование вызовов старым классам пока, или перенос кода.
*   Risk: Low.

### Шаг 3: Core Extraction - Search
Переписать `RAGSearch` как Use Case `HybridSearch`, используя интерфейсы.
*   Files: `src/core/use_cases/search.py`.
*   Tests: Unit tests с mock-интерфейсами.

### Шаг 4: App Wiring for Search
Обновить `legale.py` и `tgbot.py` чтобы использовать `HybridSearch` use case (через bootstrap), вместо прямого вызова `RAGSearch`.
*   Verify: RAG поиск работает как раньше.

### Шаг 5: Dispatcher & CLI
Создать `CommandDispatcher`. Перевести CLI команды (`/find`) на него.
*   Files: `src/core/dispatcher.py`, `src/bot/cli_adapter.py`.

### Шаг 6: Ingestion Pipeline Migration
Разбить `IngestionPipeline` на Use Cases (`IngestMessages`, `ProcessChunks`).
*   Files: `src/core/use_cases/ingest.py`.
*   Refactor: Удалить прямые вызовы DB из старого pipeline.

### Шаг 7: Telegram Bot Migration
Перевести `tgbot.py` на использование `CommandDispatcher`.
*   Integration: Бот получает сообщения но логику отдает диспетчеру.

## 7. Тестовая стратегия
*   **Unit Tests**: Для `core/use_cases` используем `FakeMessageStore` (in-memory, dict based) и `FakeVectorIndex`.
*   **Contract Tests**: Набор тестов, которые запускаются и на `FakeMessageStore`, и на `SqliteMessageStore`, проверяя идентичность поведения.
*   **Manual**: Проверка команд `/find` и `/ingest` после каждого этапа миграции.

## 8. Риски
1.  **DB Schema**: Попытка изменить схему одновременно с рефакторингом кода.
    *   *Mitigation*: Не менять схему БД на этом этапе. Использовать существующие таблицы.
2.  **Vector Sync**: Рассинхрон при изменении логики ID.
    *   *Mitigation*: Строго сохранить логику генерации ID чанков.
3.  **Performance**: Оверхед на абстракции.
    *   *Mitigation*: Python protocols имеют минимальный оверхед. DI делать статическим (в bootstrap), а не динамическим при каждом вызове.

---

## Список задач (Tasks)

### Critical (Architecture)

1.  **Define Core Domain & Interfaces**
    *   File: `src/core/interfaces.py`, `src/core/domain.py`
    *   Impl: Создать dataclasses и Protocols для Stores, Index, Embedder.
    *   Done: Файлы созданы, типы проверены.

2.  **Implement Storage Adapters**
    *   File: `src/adapters/persistence.py`, `src/adapters/vector.py`
    *   Impl: Реализовать `MessageStore` через `src/storage/db.py`. Реализовать `VectorIndex` через `src/storage/vector_store.py`.
    *   Done: Адаптеры проходят тесты интерфейсов.

3.  **Implement Search Use Case**
    *   File: `src/core/use_cases/search.py`
    *   Impl: Логика поиска (`HybridSearch`) перенесена из `rag_search.py` в чистую функцию/класс, зависящую от интерфейсов.
    *   Done: Unit тесты Use Case работают.

4.  **Implement Command Dispatcher**
    *   File: `src/core/dispatcher.py`
    *   Impl: Реестр команд и маппинг на use cases.
    *   Done: Dispatcher обрабатывает тестовые команды.

5.  **Refactor CLI to use Dispatcher**
    *   File: `src/bot/cli.py` -> `src/app/main_cli.py`
    *   Impl: CLI парсит args и вызывает Dispatcher.
    *   Done: Команда `/find` работает через новый стек.

6.  **Refactor Telegram Bot to use Dispatcher**
    *   File: `src/bot/tgbot.py`
    *   Impl: Handlers вызывают Dispatcher.
    *   Done: Бот отвечает на команды через новый стек.

### Non-Critical (Cleanup & Optimization)

7.  **Refactor Ingestion to Use Cases**
    *   Goal: Декомпозировать pipeline.
    *   Impl: Разбить `IngestionPipeline` на отдельные шаги-классы в `src/core/use_cases/ingest/`.

8.  **Feature Flag for Clustering**
    *   Goal: Сделать кластеризацию опциональной.
    *   Impl: Вынести логику clustered search в отдельный декоратор или стратегию.

9.  **Expand Tests Coverage**
    *   Goal: Покрыть адаптеры контрактными тестами.
