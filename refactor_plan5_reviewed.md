# План рефакторинга: Core + Interfaces + Adapters (Reviewed)

Этот документ описывает уточненный план перехода архитектуры проекта `legale-bot` на принципы Clean Architecture (Core + Interfaces + Adapters).

> [!NOTE]
> Review Status: План был пересмотрен на основе текущего состояния репозитория. Статусы "Done" изменены на "Todo", уточнены пути к файлам и добавлены шаги по исправлению импортов.

## 1. Инвентаризация и декомпозиция домена

### Доменные сущности (Entities)
Эти объекты живут в `src/core/domain.py` (или `src/core/domain/` при росте) и не зависят от БД.

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

Расположение: `src/core/interfaces.py`

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

class LLM(Protocol):
    def complete(self, prompt: str, system: str = None, **kwargs) -> str: ...
```

### Infrastructure
```python
class ConfigProvider(Protocol):
    def get_profile_config(self, profile_name: str) -> ProfileConfig: ...

class TransactionManager(Protocol):
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
    *   Logic: Switch on command name -> execute specific Use Case -> Return `CommandResult`.

## 4. Adapters

Расположение: `src/adapters/`

### Persistence (SQLAlchemy)
*   **`SqliteMessageStore`**: Implements `MessageStore`. Maps domain `Message` to ORM `MessageModel`.
*   **`SqliteChunkStore`**: Implements `ChunkStore`. Maps domain `Chunk` to ORM `ChunkModel`.

### Vector (Chroma)
*   **`ChromaVectorIndex`**: Implements `VectorIndex`. Wraps `chromadb.Client`.

### AI
*   **`LocalEmbedderAdapter`**: Wraps `sentence-transformers` (Currently in `src/core/embedding.py`).
*   **`RemoteEmbedderAdapter`**: Wraps OpenAI/OpenRouter API.
*   **`OpenAILLMAdapter`**: Wraps `openai` client.

### Interface Adapters (Inbound)
*   **`TelegramHandler`**:
    *   Received `Update` -> parse to `CommandContext` -> `Dispatcher.dispatch` -> Format result -> Send.
*   **`CLIHandler`**:
    *   `sys.argv` -> parse to `CommandContext` -> `Dispatcher.dispatch` -> Print result.

## 5. Структура директорий

```
src/
├── core/                  # Pure Python, no framework deps
│   ├── domain.py          # Data classes (Message, Chunk) - SINGLE FILE INITIALLY
│   ├── interfaces.py      # Protocols - SINGLE FILE INITIALLY
│   ├── use_cases/         # Ingest, Search, Commands
│   └── errors.py          # Domain exceptions
├── adapters/              # Implementations (Currently Missing)
│   ├── persistence/       # SQLAlchemy models & stores
│   ├── vector/            # ChromaDB adapter
│   ├── embedding/         # Local/Remote embedders
│   ├── llm/               # LLM clients
│   └── telegram/          # Telegram bot specific logic
├── app/                   # Composition root
│   ├── config.py
│   ├── bootstrap.py       # DI container / wiring
│   └── main_cli.py        # Entry point for CLI
└── lib/                   # Shared utilities
    ├── syslog2.py         # Logging (Move from core)
    ├── argparse2.py       # Arguments parsing (Move from core)
    └── text_tools.py      # Text utilities
```

## 6. План миграции (Инкрементальный)

### Шаг 1: Foundation ✅
Создать структуру папок и базовые файлы домена.
*   **Files**: `src/core/domain.py`, `src/core/interfaces.py`. ✅
*   **Move**: Перенести `src/core/syslog2.py` и `src/core/argparse2.py` в `src/lib/` (создать `src/lib/__init__.py`). ✅
*   **Risk**: Requires fixing imports in existing code (`tgbot.py`, etc.). ✅ Resolved

### Шаг 2: Adapters Wrapping ✅
Обернуть текущий код (`src/storage/db.py`, `src/storage/vector_store.py`) в адаптеры.
*   **Files**: `src/adapters/persistence/sqlite_message_store.py`, `src/adapters/persistence/sqlite_chunk_store.py`, `src/adapters/vector/chroma_vector_index.py`. ✅
*   **Action**: Create new adapter classes that call existing logic. Do *not* delete old files yet. ✅
*   **Risk**: Low. ✅

### Шаг 3: Core Extraction - Search
Переписать `RAGSearch` как Use Case `HybridSearch`.
*   **Files**: `src/core/use_cases/search.py`.
*   **Action**: Создать чистую логику поиска, используя `FakeMessageStore` для тестов.

### Шаг 4: App Wiring for Search
Обновить `legale.py` и `tgbot.py`, чтобы подменить старый `RAGSearch` на новый `HybridSearch` (через Dependency Injection в `src/app/bootstrap.py`).
*   **Verification**: Команда `/find` должна работать идентично.

### Шаг 5: Dispatcher & CLI
Создать `CommandDispatcher` и перевести CLI.
*   **Files**: `src/core/dispatcher.py`, `src/app/main_cli.py`.

### Шаг 6: Ingestion Pipeline Migration
Разбить `IngestionPipeline` на Use Cases.
*   **Files**: `src/core/use_cases/ingest.py`.

### Шаг 7: Telegram Bot Migration
Перевести `tgbot.py` на использование Dispatcher.

## 7. Риски и Mitigation
1.  **Import Hell**: Перемещение `syslog2` может сломать много файлов.
    *   *Mitigation*: Сделать массовый refactor imports с помощью IDE или script (sed).
2.  **Circular Deps**: `core` не может зависеть от `adapters`.
    *   *Mitigation*: Строгий контроль импортов. Использовать `mypy`.

---

## Список задач (Tasks)

### Critical (Architecture)

1.  **Define Core Domain & Interfaces** ✅
    *   [x] Create `src/core/domain.py`
    *   [x] Create `src/core/interfaces.py`
    *   [x] Move `src/core/syslog2.py` -> `src/lib/syslog2.py` & fix imports
    *   [x] Move `src/core/argparse2.py` -> `src/lib/argparse2.py` & fix imports

2.  **Implement Storage Adapters** ✅
    *   [x] Create `src/adapters/persistence/`
    *   [x] Create `src/adapters/vector/`
    *   [x] Implement `SqliteMessageStore` (using `src/storage/db.py`)
    *   [x] Implement `SqliteChunkStore` (using `src/storage/db.py`)
    *   [x] Implement `ChromaVectorIndex` (using `src/storage/vector_store.py`)

3.  **Implement Search Use Case**
    *   [ ] Create `src/core/use_cases/search.py` (`HybridSearch` class)
    *   [ ] Write unit tests with Fake adapters

4.  **Implement Command Dispatcher**
    *   [ ] Create `src/core/dispatcher.py` implementation

5.  **Refactor CLI to use Dispatcher**
    *   [ ] Create `src/app/main_cli.py`
    *   [ ] Wire up in `legale.py` (CLI entry point)

6.  **Refactor Telegram Bot**
    *   [ ] Update `src/bot/tgbot.py` to use `Dispatcher`

### Non-Critical

7.  **Refactor Ingestion to Use Cases**
    *   [ ] Split `IngestionPipeline` into Use Cases in `src/core/use_cases/ingest/`

8.  **Expand Tests Coverage**
    *   [ ] Add contract tests for Adapters
