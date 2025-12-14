# Анализ архитектуры Librarian Chatbot (Legale Bot)

## Общее описание

Система представляет собой продвинутый RAG (Retrieval-Augmented Generation) бот с двухуровневой иерархией тем и гибридной системой поиска. Архитектура построена вокруг концепции **профилей**, позволяя изолировать данные и конфигурации для разных задач.

### RAG Архитектура
Система использует 10-этапный конвейер (pipeline) обработки данных:
1.  **Ingestion & Parsing**: Сообщения Telegram парсятся из JSON-дампов или текстовых файлов.
2.  **Chunking**: Сообщения группируются в смысловые блоки (чанки) на основе лимитов токенов (используя `tiktoken`) с перекрытием (overlap) и временными метками.
3.  **Embeddings**: Для чанков генерируются векторные представления. Поддерживается **гибридный режим**:
    *   **Local**: `sentence-transformers` (e.g., `paraphrase-multilingual-mpnet-base-v2`) для оффлайн работы.
    *   **Remote**: OpenAI-совместимый API (OpenRouter/OpenAI) (e.g., `text-embedding-3-small`).
4.  **Hierarchical Clustering**:
    *   **L1 Topics (Fine-grained)**: Векторы чанков кластеризуются алгоритмом **HDBSCAN**.
    *   **L2 Topics (Super-topics)**: Центроиды L1 тем повторно кластеризуются для создания супер-категорий.
5.  **Topic Naming**: LLM генерирует человекочитаемые названия для кластеров.

### Retrieval (Поиск)
Поиск реализован как двухступенчатый процесс:
1.  **Vector Search**: Поиск топ-K похожих чанков по косинусному расстоянию в векторной БД (ChromaDB).
2.  **Context Enrichment**: Обогащение чанков контекстом из SQLite (соседние сообщения, принадлежность к темам L1/L2) для формирования полного ответа.

---

## Пофайловая структура

### Корень проекта

#### `legale.py`
Главный оркестратор CLI. Управляет профилями, запуском бота и пайплайном загрузки.

*   `class ProfileManager`
    *   `__init__(self, project_root: Path)`: Инициализация менеджера.
    *   `get_current_profile(self) -> str`: Получает имя активного профиля из .env.
    *   `set_current_profile(self, profile_name: str)`: Устанавливает активный профиль.
    *   `create_profile(self, profile_name: str, set_active: bool = False) -> Path`: Создает структуру директорий нового профиля.
    *   `list_profiles(self)`: Выводит список всех профилей.
    *   `delete_profile(self, profile_name: str, force: bool = False)`: Удаляет профиль и его данные.
    *   `get_profile_paths(self, profile_name: Optional[str] = None) -> dict`: Возвращает пути к БД и конфигам профиля.

*   `def cmd_ingest(args, profile_manager: ProfileManager)`: Маршрутизатор команд инжеста (stage0-stage9).
*   `def cmd_bot(args, profile_manager: ProfileManager)`: Запуск вебхука или поллинг-демона.
*   `def cmd_test_embedding(args)`: Утилита для теста генерации эмбеддингов.

### `src/ingestion`

#### `src/ingestion/pipeline.py`
Ядро ETL процесса. Оркестрирует весь пайплайн от сырого JSON до векторов.

*   `class IngestionPipeline`
    *   `__init__(self, db_url: str, vector_db_path: str, collection_name: str, profile_dir: Optional[str])`: Инициализация с загрузкой конфига профиля.
    *   `run_all(self, file_path: str, model: Optional[str], batch_size: int, **clustering_params)`: Запуск всех этапов последовательно.
    *   `run_stage0(self, file_path: str)`: Парсинг и сохранение сообщений в SQLite.
    *   `run_stage1(self)`: Создание чанков (chunking).
    *   `run_stage2(self, model: Optional[str], batch_size: int)`: Генерация эмбеддингов и сохранение в SQLite.
    *   `run_stage3(self)`: Синхронизация чанков в векторную БД (ChromaDB).
    *   `run_stage4(self, **clustering_params)`: Кластеризация L1 (HDBSCAN).
    *   `run_stage5(self)`: Синхронизация L1 топиков в векторную БД.
    *   `run_stage6(self, **clustering_params)`: Кластеризация L2.
    *   `run_stage7(self)`: Синхронизация L2 топиков в векторную БД.
    *   `run_stage8(self, only_unnamed: bool, rebuild: bool)`: Именование L1 тем через LLM.
    *   `run_stage9(self, only_unnamed: bool, rebuild: bool)`: Именование L2 тем через LLM.
    *   `clear_all(self)`: Полная очистка данных.

#### `src/ingestion/chunker.py`
Логика разбиения сообщений на чанки.

*   `class MessageChunker`
    *   `__init__(self, chunk_token_min: int, chunk_token_max: int, chunk_overlap_ratio: float)`: Настройка токенизатора `tiktoken`.
    *   `chunk_messages(self, messages: List[ChatMessage]) -> List[EnhancedTextChunk]`: Группирует сообщения в чанки с учетом лимитов токенов и временных меток.

#### `src/ingestion/parser.py`
Парсер дампов чатов.

*   `class ChatParser`
    *   `parse_file(self, file_path: str) -> List[ChatMessage]`: Читает JSON дамп (или текстовый файл) и возвращает список объектов `ChatMessage`.

### `src/core`

#### `src/core/rag_search.py`
Высокоуровневая логика поиска.

*   `class RAGSearch`
    *   `__init__(self, db: Database, vector_store: VectorStore, embedding_client: EmbeddingClient)`
    *   `search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]`: Выполняет векторный поиск и джойнит результаты с метаданными из SQL базы (сообщения, топики).

#### `src/core/embedding.py`
Клиенты для генерации эмбеддингов.

*   `class EmbeddingClient`: Клиент для OpenAI/OpenRouter API.
    *   `get_embeddings(self, texts: List[str]) -> List[List[float]]`.
*   `class LocalEmbeddingClient`: Клиент на базе `sentence-transformers`.
    *   `get_embeddings(self, texts: List[str]) -> List[List[float]]`: Генерирует эмбеддинги локально.
*   `def get_embedding_function(provider: Optional[str], model: Optional[str]) -> Optional[EmbeddingFunction]`: Фабрика для ChromaDB.

#### `src/core/llm.py`
Обертка над API LLM.

*   `class LLMClient`
    *   `complete(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str`: Синхронная генерация ответа.
    *   `stream_complete(self, messages: List[Dict[str, str]], temperature: float) -> Generator`: Потоковая генерация.

### `src/ai`

#### `src/ai/clustering.py`
Иерархическая кластеризация.

*   `class TopicClusterer`
    *   `perform_l1_clustering(self, min_cluster_size: int, ...)`: Кластеризация векторов чанков через HDBSCAN.
    *   `perform_l2_clustering(self, min_cluster_size: int, ...)`: Кластеризация центроидов L1 топиков.
    *   `assign_l1_topics_to_chunks(self)`: Присвоение ID топиков чанкам в БД.
    *   `name_topics(self, ...)`: Использование LLM для генерации названий топиков на основе репрезентативных чанков.

### `src/storage`

#### `src/storage/db.py`
ORM модели (SQLAlchemy) и методы работы с SQLite.

*   **Models**: `MessageModel`, `ChunkModel`, `TopicL1Model`, `TopicL2Model`.
*   `class Database`
    *   `__init__(self, db_url: str)`: Инициализация и авто-миграции схемы.
    *   `add_messages_batch(self, messages: List[dict]) -> int`: Пакетная вставка сообщений.
    *   `create_topic_l1(...) -> int`: Создание L1 топика.
    *   `create_topic_l2(...) -> int`: Создание L2 топика.
    *   `get_chunk_link_info(self, chunk_id: str) -> Tuple`: Получение информации для ссылки на сообщение (chat_id, msg_id).
    *   `update_chunk_topics(...)`: Обновление привязки чанка к топикам.

#### `src/storage/vector_store.py`
Обертка над ChromaDB.

*   `class VectorStore`
    *   `__init__(self, persist_directory: str, collection_name: str, ...)`
    *   `add_documents_with_embeddings(self, ids, documents, embeddings, metadatas, ...)`: Сохранение векторов.
    *   `query(self, query_texts: List[str], n_results: int) -> Dict`: Поиск ближайших соседей.
    *   `get_all_embeddings(self) -> Dict`: Выгрузка всех векторов (для кластеризации).
