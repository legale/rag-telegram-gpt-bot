# Legale Bot - Architectural Overview

## Purpose

Legale Bot is a RAG (Retrieval-Augmented Generation) Telegram bot that analyzes chat history and answers questions based on ingested Telegram conversations. The system:

- Ingests Telegram chat history (via Telethon API or JSON dumps)
- Splits messages into semantic chunks with token-based chunking
- Generates embeddings using local models (sentence-transformers) or API services (OpenRouter/Voyage)
- Stores chunks in SQLite (metadata) + ChromaDB (vector embeddings)
- Provides hybrid search combining FTS5 (SQLite) and vector similarity (ChromaDB)
- Answers user queries using LLM (OpenRouter) with retrieved context
- Supports multiple bot profiles with separate databases
- Runs as FastAPI webhook daemon or interactive CLI

## How to Run

### Prerequisites
- Python 3.11+
- Poetry for dependency management
- Environment variables in `.env`:
  - `TELEGRAM_API_ID`, `TELEGRAM_API_HASH` (from my.telegram.org)
  - `TELEGRAM_BOT_TOKEN` (from @BotFather)
  - `OPENROUTER_API_KEY` (for LLM)
  - `VOYAGE_API_KEY` (optional, for embeddings)
  - `ADMIN_PASSWORD` (for bot administration)

### Main Entry Points

1. **Unified CLI** (`legale.py`):
   ```bash
   poetry run python legale.py profile create mybot --set-active
   poetry run python legale.py telegram dump "Chat Name" --limit 10000
   poetry run python legale.py ingest telegram_dump_Chat.json
   poetry run python legale.py chat -vv
   poetry run python legale.py bot run --port 8080
   ```

2. **Telegram Bot Webhook** (`src/bot/tgbot.py`):
   ```bash
   # Register webhook
   poetry run python src/bot/tgbot.py register url https://yourdomain.com/webhook
   
   # Run in foreground (testing)
   poetry run python src/bot/tgbot.py run --port 8080
   
   # Run as daemon (production)
   poetry run python src/bot/tgbot.py daemon --port 8000
   ```

3. **Interactive CLI Chat** (`src/bot/cli.py`):
   ```bash
   poetry run python src/bot/cli.py
   ```

## Repo Layout

```
legale-bot/
├── legale.py                    # Main CLI orchestrator (unified entry point)
├── models.txt                   # Available LLM models (one per line)
├── pyproject.toml              # Dependencies (Poetry)
├── .env                        # Environment variables (gitignored)
├── profiles/                   # Profile data (gitignored)
│   └── <profile_name>/
│       ├── legale_bot.db      # SQLite database
│       ├── chroma_db/          # ChromaDB vector store
│       ├── config.json         # Profile configuration
│       ├── admin.json          # Admin user info
│       └── telegram_session.session  # Telethon session
├── src/
│   ├── bot/                    # Telegram bot layer
│   │   ├── tgbot.py           # FastAPI webhook daemon
│   │   ├── core.py             # LegaleBot class (main bot logic)
│   │   ├── cli.py              # Interactive chat CLI
│   │   ├── admin.py            # AdminManager (access control)
│   │   ├── admin_router.py     # AdminCommandRouter (admin commands)
│   │   ├── admin_commands.py   # Admin command handlers
│   │   ├── command_parser.py   # Command argument parsing
│   │   ├── config.py           # BotConfig (profile config management)
│   │   └── utils/               # Utilities (access control, formatting, etc.)
│   ├── core/                   # Domain logic and use cases
│   │   ├── dispatcher.py       # CommandDispatcher (routing commands)
│   │   ├── commands.py          # Command handlers (start, help, reset, tokens, model, find)
│   │   ├── admin_commands.py   # Admin command handlers (admin_set, admin_get, admin)
│   │   ├── llm.py              # LLMClient (OpenRouter API client)
│   │   ├── embedding.py        # EmbeddingClient, LocalEmbeddingClient
│   │   ├── prompt.py           # PromptEngine (system prompt templates)
│   │   ├── search.py           # HybridSearch (use case)
│   │   ├── hybrid_retrieval.py # HybridRetrievalService (FTS5 + vector rerank)
│   │   ├── message_search.py   # High-level message search functions
│   │   ├── query_rewriter.py   # QueryRewriter (LLM-based query rephrasing)
│   │   ├── chunk_utils.py      # Chunk utilities
│   │   ├── distance_utils.py   # Distance/similarity utilities
│   │   ├── domain.py           # Domain models (Message, Chunk, SearchResult)
│   │   ├── interfaces.py       # Protocol interfaces (MessageStore, ChunkStore, VectorIndex, etc.)
│   │   └── ingest_use_cases/   # Ingestion use cases
│   │       ├── ingest_messages.py
│   │       ├── process_chunks.py
│   │       ├── generate_embeddings.py
│   │       ├── sync_to_vector_store.py
│   │       └── pipeline_orchestrator.py
│   ├── ingestion/              # Data ingestion pipeline
│   │   ├── parser.py           # ChatParser (JSON dump parsing)
│   │   ├── chunker.py          # MessageChunker (token-based chunking)
│   │   ├── pipeline.py         # IngestionPipeline (orchestrates ingestion)
│   │   └── telegram.py         # TelegramFetcher (Telethon client wrapper)
│   ├── storage/                # Data persistence layer
│   │   ├── db.py               # Database (SQLAlchemy models, SQLite)
│   │   ├── vector_store.py     # VectorStore (ChromaDB wrapper)
│   │   └── migrations/         # Database migrations
│   ├── adapters/               # Adapters (implement interfaces)
│   │   ├── persistence/       # SQLite adapters
│   │   │   ├── sqlite_message_store.py
│   │   │   ├── sqlite_chunk_store.py
│   │   │   └── sqlite_fts_index.py
│   │   ├── vector/             # ChromaDB adapters
│   │   │   └── chroma_vector_index.py
│   │   ├── llm/                # LLM adapters
│   │   │   └── llm_adapter.py
│   │   └── embedding/         # Embedding adapters
│   │       └── embedder_adapter.py
│   ├── app/                    # Application bootstrap
│   │   ├── bootstrap.py        # Dependency injection (create_hybrid_retrieval, etc.)
│   │   └── main_cli.py         # Command dispatcher setup
│   └── lib/                    # Shared utilities
│       ├── argparse2.py        # Custom argument parser
│       └── syslog2.py           # Structured logging
└── tests/                      # Test suite (87 tests, 41% coverage)
```

## Entry Points

### 1. Unified CLI (`legale.py`)
- **Purpose**: Single entry point for all operations (profile management, ingestion, chat, bot)
- **Commands**:
  - `profile` - Create/list/get/set/delete profiles
  - `telegram` - List chats, dump messages (via Telethon)
  - `ingest` - Ingest JSON dump into database
  - `chat` - Interactive chat CLI
  - `bot` - Bot webhook management (register/delete/run/daemon)
- **Profile Management**: Manages `profiles/<name>/` directories with separate databases
- **Environment**: Uses `.env` for `ACTIVE_PROFILE` and API keys

### 2. Telegram Bot Webhook (`src/bot/tgbot.py`)
- **Purpose**: FastAPI webhook server for Telegram Bot API
- **Modes**:
  - `register` - Register webhook URL with Telegram
  - `delete` - Delete webhook
  - `run` - Run server in foreground (for testing)
  - `daemon` - Run as background daemon (production)
- **Lifespan**: Initializes `LegaleBot`, `AdminManager`, `AdminCommandRouter` on startup
- **Endpoints**:
  - `POST /webhook` - Telegram webhook endpoint
  - `GET /health` - Health check

### 3. Interactive CLI Chat (`src/bot/cli.py`)
- **Purpose**: Interactive chat interface for testing bot responses
- **Usage**: Reads user input from stdin, calls `LegaleBot.chat()`, prints responses
- **Commands**: Supports same commands as Telegram bot (`/help`, `/reset`, `/tokens`, `/model`, `/find`)

## Components by File

### Bot Layer (`src/bot/`)

**`tgbot.py`** (1843 lines)
- **Purpose**: FastAPI webhook daemon for Telegram bot
- **Key Classes**: `MessageHandler`, FastAPI app with lifespan
- **Dependencies**: FastAPI, uvicorn, python-telegram-bot, LegaleBot, AdminManager
- **Inputs**: HTTP POST requests from Telegram (webhook updates)
- **Outputs**: HTTP responses, Telegram messages via Bot API
- **Lifecycle**: Global `bot_instance`, `admin_manager`, `admin_router` (singletons, initialized in lifespan)

**`core.py`** (636 lines)
- **Purpose**: Main bot logic orchestrating RAG pipeline
- **Key Class**: `LegaleBot`
- **Key Methods**:
  - `chat(user_input, n_results)` - Main chat method (retrieval + LLM)
  - `get_model()`, `set_model()` - Model switching
  - `reset_context()` - Clear chat history
  - `get_token_usage()` - Token counting
- **Dependencies**: Database, VectorStore, HybridRetrievalService, LLMClient, PromptEngine
- **State**: In-memory chat history, active context cache
- **Inputs**: User queries (strings)
- **Outputs**: LLM-generated responses (strings)

**`admin.py`** (AdminManager)
- **Purpose**: Access control and admin user management
- **Key Methods**: `is_admin()`, `set_admin()`, `get_admin()`
- **Storage**: `profiles/<profile>/admin.json` (user_id, username, name)
- **Dependencies**: BotConfig

**`admin_router.py`** (AdminCommandRouter)
- **Purpose**: Routes admin commands (`/admin <subcommand>`)
- **Subcommands**: `profile`, `ingest`, `stats`, `restart`, `allowed`, `chat`, `frequency`, `model`, `system_prompt`, `help`
- **Dependencies**: AdminManager, LegaleBot, TaskManager

**`config.py`** (BotConfig)
- **Purpose**: Profile configuration management
- **Storage**: `profiles/<profile>/config.json`
- **Key Settings**: `embedding_model`, `embedding_generator`, `current_model`, `system_prompt`, `allowed_chats`, `response_frequency`, chunking parameters, thresholds

### Core Domain (`src/core/`)

**`dispatcher.py`** (CommandDispatcher)
- **Purpose**: Routes commands to handlers
- **Key Methods**: `register()`, `dispatch()`, `dispatch_async()`
- **Handlers**: Sync (`CommandHandler`) and async (`AsyncCommandHandler`)

**`commands.py`** (Command Handlers)
- **Purpose**: Handlers for user commands (`/start`, `/help`, `/reset`, `/tokens`, `/model`, `/find`)
- **Key Classes**: `StartCommandHandler`, `HelpCommandHandler`, `ResetCommandHandler`, `TokensCommandHandler`, `ModelCommandHandler`, `FindCommandHandler`

**`hybrid_retrieval.py`** (HybridRetrievalService)
- **Purpose**: Hybrid search: FTS5 candidates → vector rerank → context packing
- **Key Methods**: `search()`, `retrieve()` (compatibility)
- **Flow**: FTS5 search → query rephrasing (optional) → vector reranking → dedup + token budget
- **Modes**: `hybrid`, `fts_only`, `vector_only`
- **Dependencies**: FTSIndex, VectorIndex, Embedder, ChunkStore, MessageStore, LLM (for rephrasing)

**`llm.py`** (LLMClient)
- **Purpose**: OpenRouter API client for LLM completions
- **Key Methods**: `complete(messages, temperature, max_tokens)`, `count_tokens()`
- **Dependencies**: OpenAI SDK, tiktoken
- **Inputs**: List of message dicts `[{"role": "user", "content": "..."}]`
- **Outputs**: Generated text string

**`embedding.py`** (EmbeddingClient, LocalEmbeddingClient)
- **Purpose**: Text embedding generation
- **EmbeddingClient**: Uses OpenAI-compatible API (OpenRouter/Voyage)
- **LocalEmbeddingClient**: Uses sentence-transformers (local models)
- **Key Methods**: `get_embeddings()`, `get_embedding()`, `get_embeddings_batched()`

**`prompt.py`** (PromptEngine)
- **Purpose**: System prompt template management
- **Key Methods**: `build_prompt()`, `get_system_prompt()`
- **Templates**: RAG prompt with context chunks and chat history

**`message_search.py`** (High-level search functions)
- **Purpose**: Search functions for message links and contents
- **Key Functions**: `search_message_links()`, `search_message_contents()`
- **Dependencies**: HybridRetrievalService, Database

### Ingestion (`src/ingestion/`)

**`pipeline.py`** (IngestionPipeline)
- **Purpose**: Orchestrates ingestion: parse → chunk → persist → embed → vector store
- **Key Methods**: `ingest()`, `clear_stage0()` through `clear_stage3()`
- **Stages**:
  1. Parse JSON dump → ChatMessage objects
  2. Chunk messages → EnhancedTextChunk objects
  3. Persist chunks to SQLite (with message references)
  4. Generate embeddings → store in SQLite (`embedding_json`)
  5. Sync to ChromaDB vector store
- **Dependencies**: ChatParser, MessageChunker, Database, VectorStore, EmbeddingClient

**`parser.py`** (ChatParser)
- **Purpose**: Parses JSON chat dumps into ChatMessage objects
- **Key Method**: `parse_file(file_path)`
- **Input**: JSON file with message array
- **Output**: List of `ChatMessage` (id, timestamp, sender, content)

**`chunker.py`** (MessageChunker)
- **Purpose**: Token-based chunking with overlap
- **Key Method**: `chunk_messages(messages)`
- **Parameters**: `chunk_token_min`, `chunk_token_max`, `chunk_overlap_ratio`
- **Output**: List of `EnhancedTextChunk` (text, metadata, original_messages)

**`telegram.py`** (TelegramFetcher)
- **Purpose**: Fetches messages from Telegram via Telethon
- **Key Methods**: `list_chats()`, `dump_chat()`
- **Dependencies**: Telethon
- **Output**: JSON dump file

### Storage (`src/storage/`)

**`db.py`** (Database)
- **Purpose**: SQLite database with SQLAlchemy ORM
- **Key Models**:
  - `MessageModel`: messages table (msg_id, chat_id, ts, from_id, text)
  - `ChunkModel`: chunks table (id, text, metadata_json, embedding_json, embedding_dim, msg_id_start, msg_id_end, ts_from, ts_to, chat_id)
  - `MessageMetaModel`: message_meta table (msg_id, meta_json)
- **FTS5 Tables**: `messages_fts`, `chunks_fts` (virtual tables with triggers)
- **Key Methods**: `add_message()`, `add_messages_batch()`, `add_chunk_with_messages()`, `get_messages_by_chunk()`, `fts_search()`

**`vector_store.py`** (VectorStore)
- **Purpose**: ChromaDB wrapper for vector storage
- **Key Methods**: `add_documents_with_embeddings()`, `query()`, `count()`, `clear()`
- **Collection**: `embed-chunks` (default)
- **Dependencies**: chromadb

### Adapters (`src/adapters/`)

**`persistence/sqlite_message_store.py`** (SqliteMessageStore)
- **Purpose**: Implements `MessageStore` interface using SQLite
- **Key Methods**: `save_batch()`, `get_by_chat()`, `get_context()`, `count()`

**`persistence/sqlite_chunk_store.py`** (SqliteChunkStore)
- **Purpose**: Implements `ChunkStore` interface using SQLite
- **Key Methods**: `save_batch()`, `get_by_ids()`, `update_topics()`, `clear()`

**`persistence/sqlite_fts_index.py`** (SqliteFTSIndex)
- **Purpose**: Implements `FTSIndex` interface using SQLite FTS5
- **Key Methods**: `search()`, `search_chunks()`, `normalize_text()`

**`vector/chroma_vector_index.py`** (ChromaVectorIndex)
- **Purpose**: Implements `VectorIndex` interface using ChromaDB
- **Key Methods**: `upsert()`, `query()`, `delete()`, `count()`

**`llm/llm_adapter.py`** (LLMAdapter)
- **Purpose**: Adapter wrapping LLMClient to implement `LLM` protocol
- **Key Method**: `complete(prompt, system, **kwargs)`

**`embedding/embedder_adapter.py`** (EmbedderAdapter)
- **Purpose**: Adapter wrapping EmbeddingClient to implement `Embedder` protocol
- **Key Methods**: `embed_documents()`, `embed_query()`

### Bootstrap (`src/app/bootstrap.py`)

**`bootstrap.py`** (Dependency Injection)
- **Purpose**: Creates and wires components together
- **Key Functions**: `create_hybrid_retrieval()`, `create_hybrid_search()`
- **Dependencies**: Creates Database, VectorStore, adapters, use cases

## Runtime Flows

### User Query Flow (Telegram Bot)

```
1. Telegram sends webhook POST → FastAPI /webhook endpoint
2. tgbot.py: _process_webhook_update()
   ├─ Parse Update object
   ├─ Check access control (AccessControlService)
   ├─ Check frequency limits (FrequencyController)
   └─ Route to MessageHandler.handle_message()
3. MessageHandler.handle_message()
   ├─ Check if command (starts with "/")
   │  └─ If command: route_command() → CommandDispatcher
   │     └─ Dispatch to handler (StartCommandHandler, FindCommandHandler, etc.)
   └─ If not command: treat as user query
      └─ bot_instance.chat(user_input, n_results=3)
4. LegaleBot.chat()
   ├─ _get_or_build_context(user_input, n_results)
   │  ├─ Check if context cache valid (_should_refresh_context)
   │  ├─ If refresh needed:
   │  │  ├─ retrieval_service.search(query, top_k=50, rerank_top_k=20)
   │  │  │  ├─ FTS5 search (SqliteFTSIndex.search_chunks())
   │  │  │  ├─ Query rephrasing (QueryRewriter.rephrase_for_embedding())
   │  │  │  ├─ Vector reranking (cosine similarity)
   │  │  │  └─ Context packing (dedup, neighbors, token budget)
   │  │  └─ Cache context chunks
   │  └─ Use cached context if valid
   ├─ _build_prompt_and_history(user_input, context_chunks)
   │  ├─ Build system prompt (PromptEngine.build_prompt())
   │  └─ Build chat history (last 5 messages)
   ├─ _call_llm_with_retry(messages, system_prompt)
   │  └─ llm_client.complete(messages, temperature, max_tokens)
   │     └─ OpenAI SDK → OpenRouter API
   └─ Return response string
5. Response sent back via Telegram Bot API
```

**Lifecycle**:
- **Singletons**: `bot_instance`, `admin_manager`, `admin_router` (created once in lifespan, reused)
- **Per-request**: `MessageHandler` (created per request), `CommandContext` (created per command)

### Ingestion Flow

```
1. User runs: legale.py ingest <file.json> [--clear]
2. legale.py: cmd_ingest()
   ├─ Create IngestionPipeline(db_url, vector_db_path, profile_dir)
   ├─ If --clear: pipeline._clear_data() (SQLite + ChromaDB)
   └─ pipeline.ingest(file_path)
3. IngestionPipeline.ingest()
   ├─ Stage 0: Parse file
   │  └─ parser.parse_file(file_path) → List[ChatMessage]
   ├─ Stage 1: Chunk messages
   │  └─ chunker.chunk_messages(messages) → List[EnhancedTextChunk]
   ├─ Stage 2: Persist to SQLite
   │  ├─ db.add_messages_batch(messages) → messages table
   │  └─ db.add_chunk_with_messages(chunks) → chunks table
   │     └─ FTS5 triggers update messages_fts, chunks_fts
   ├─ Stage 3: Generate embeddings
   │  ├─ embedding_client.get_embeddings_batched(chunk_texts)
   │  │  └─ LocalEmbeddingClient: sentence-transformers
   │  │  └─ EmbeddingClient: OpenAI API (OpenRouter/Voyage)
   │  └─ Update chunks.embedding_json, chunks.embedding_dim
   └─ Stage 4: Sync to ChromaDB
      └─ vector_store.add_documents_with_embeddings(ids, texts, embeddings, metadatas)
         └─ ChromaDB collection.add()
```

**Lifecycle**:
- **Per-ingestion**: `IngestionPipeline` instance (created per ingest command)
- **Reused**: `Database`, `VectorStore` instances (created in pipeline init)

## Data Model and Persistence

### Database Schema (SQLite)

**`messages` table**:
- `msg_id` (PK, String): Composite format `{chat_id}_{msg_id}` or just `{msg_id}`
- `chat_id` (String, indexed): Telegram chat ID
- `ts` (DateTime, indexed): Message timestamp
- `from_id` (String): Sender ID/username
- `text` (Text): Message content

**`chunks` table**:
- `id` (PK, String): Chunk UUID
- `text` (Text): Chunk text content
- `metadata_json` (Text): JSON metadata (chat_id, chat_username, etc.)
- `embedding_json` (Text): JSON array of floats (embedding vector)
- `embedding_dim` (Integer, indexed): Embedding dimension
- `chat_id` (String, indexed): Chat ID
- `msg_id_start` (FK → messages.msg_id): First message in chunk
- `msg_id_end` (FK → messages.msg_id): Last message in chunk
- `msg_id_start_raw`, `msg_id_end_raw` (String): Raw message IDs (without chat_id prefix)
- `ts_from`, `ts_to` (DateTime, indexed): Time range for chunk
- `created_at` (DateTime): Chunk creation timestamp

**`message_meta` table**:
- `msg_id` (PK, FK → messages.msg_id): Message ID
- `meta_json` (Text): Additional metadata JSON
- `created_at` (DateTime): Metadata creation timestamp

**FTS5 Virtual Tables**:
- `messages_fts`: Full-text search index on messages.text
- `chunks_fts`: Full-text search index on chunks.text
- Updated automatically via triggers

### Vector Store (ChromaDB)

**Collection**: `embed-chunks` (default)
- **Storage**: `profiles/<profile>/chroma_db/`
- **Format**: Persistent ChromaDB with cosine similarity
- **Documents**: Chunk texts
- **Embeddings**: Pre-computed vectors (not generated by ChromaDB)
- **Metadata**: Chunk metadata (chat_id, msg_id_start, msg_id_end, etc.)

### Profile Structure

```
profiles/<profile_name>/
├── legale_bot.db              # SQLite database
├── chroma_db/                 # ChromaDB directory
│   └── chroma.sqlite3         # ChromaDB internal storage
├── config.json                # Profile configuration (JSON)
├── admin.json                 # Admin user info (JSON, 600 permissions)
└── telegram_session.session   # Telethon session (optional)
```

**`config.json`**:
```json
{
  "admin_password": "",
  "allowed_chats": [],
  "response_frequency": 0,
  "system_prompt": "",
  "embedding_model": "paraphrase-multilingual-mpnet-base-v2",
  "embedding_generator": "local",
  "current_model": "openai/gpt-oss-20b:free",
  "chunk_token_min": 50,
  "chunk_token_max": 1024,
  "chunk_overlap_ratio": 0.30,
  "cosine_distance_thr": 4,
  "rag_ntop": 20,
  "fts5_score_thr": 0.2
}
```

### Configuration Files

**`.env`** (root):
- `ACTIVE_PROFILE`: Current active profile name
- `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`: Telethon credentials
- `TELEGRAM_BOT_TOKEN`: Bot token
- `OPENROUTER_API_KEY`: LLM API key
- `OPENROUTER_BASE_URL`: LLM API base URL
- `VOYAGE_API_KEY`: Embeddings API key (optional)
- `ADMIN_PASSWORD`: Admin password
- `MAX_CONTEXT_TOKENS`: Token limit (default: 14000)

**`models.txt`** (root):
- One model name per line (e.g., `openai/gpt-oss-20b:free`)
- Used for model cycling (`/model` command)

## External Dependencies and Integrations

### External Services

1. **Telegram Bot API** (python-telegram-bot)
   - **Usage**: Webhook updates, sending messages
   - **Integration**: `src/bot/tgbot.py` (FastAPI webhook endpoint)
   - **Risks**: Rate limits, network failures, webhook registration issues
   - **Retries**: Handled by python-telegram-bot library

2. **Telegram Client API** (Telethon)
   - **Usage**: Fetching chat history
   - **Integration**: `src/ingestion/telegram.py`
   - **Risks**: Session expiration, rate limits, network failures
   - **Retries**: Manual retry logic in TelegramFetcher

3. **OpenRouter API** (LLM)
   - **Usage**: LLM completions
   - **Integration**: `src/core/llm.py` (LLMClient)
   - **Risks**: API rate limits, token limits, network failures, cost
   - **Retries**: Token limit errors trigger context reset and retry
   - **Timeout**: Default HTTP timeout (may need explicit timeout)

4. **Voyage API** (Embeddings, optional)
   - **Usage**: Text embeddings (via EmbeddingClient)
   - **Integration**: `src/core/embedding.py`
   - **Risks**: API rate limits, network failures, cost
   - **Fallback**: LocalEmbeddingClient (sentence-transformers)

5. **ChromaDB** (Vector Store)
   - **Usage**: Vector similarity search
   - **Integration**: `src/storage/vector_store.py`, `src/adapters/vector/chroma_vector_index.py`
   - **Risks**: Disk I/O, corruption, version compatibility
   - **Storage**: Local filesystem (no network dependency)

6. **SQLite** (Relational Database)
   - **Usage**: Messages, chunks, metadata storage
   - **Integration**: `src/storage/db.py` (SQLAlchemy)
   - **Risks**: Disk I/O, corruption, concurrent access (single writer)
   - **FTS5**: Full-text search (virtual tables with triggers)

### Key Dependencies (Python Packages)

- **fastapi**, **uvicorn**: Web framework and ASGI server
- **python-telegram-bot**: Telegram Bot API client
- **telethon**: Telegram Client API
- **openai**: OpenAI SDK (used for OpenRouter/Voyage)
- **chromadb**: Vector database
- **sqlalchemy**: ORM for SQLite
- **sentence-transformers**: Local embedding models
- **tiktoken**: Token counting
- **python-dotenv**: Environment variable loading

## Extension Points

### Adding New Commands

1. **User Commands** (`src/core/commands.py`):
   - Create handler class inheriting `CommandHandler`
   - Implement `handle(context: CommandContext) -> CommandResult`
   - Register in `src/app/main_cli.py` → `create_dispatcher()`
   - Register in `src/bot/tgbot.py` → `MessageHandler.route_command()`

2. **Admin Commands** (`src/bot/admin_commands.py`):
   - Create command class (e.g., `MyCommand`)
   - Register in `src/bot/tgbot.py` → `_register_admin_commands()`
   - Use `AdminCommandRouter.register()` or `_register_command_group()`

### Adding New Transport Layer

1. **New Transport** (e.g., Discord, Slack):
   - Create adapter in `src/bot/` (e.g., `discord_bot.py`)
   - Use `LegaleBot` and `CommandDispatcher` (same as Telegram)
   - Implement transport-specific message parsing and sending
   - Register webhook/server entry point

### Changing Retrieval Strategy

1. **New Retrieval Service**:
   - Implement `HybridRetrievalService` interface (or create new)
   - Update `src/core/hybrid_retrieval.py` or create new file
   - Register in `src/app/bootstrap.py` → `create_hybrid_retrieval()`
   - Update `LegaleBot.__init__()` to use new service

2. **Retrieval Modes**:
   - Add new mode to `HybridRetrievalService.search()` (e.g., `output_mode="timeline"`)
   - Update `_pack_context()` to handle new mode

### Changing LLM Models

1. **Add Model**:
   - Add model name to `models.txt`
   - Model is automatically available via `/model` command

2. **Change Default Model**:
   - Set `OPENROUTER_MODEL` in `.env`
   - Or set `current_model` in profile `config.json`

### Changing Embedding Models

1. **Local Model**:
   - Set `embedding_generator: "local"` in `config.json`
   - Set `embedding_model: "<model-name>"` (sentence-transformers model)
   - Model downloaded automatically on first use

2. **API Model**:
   - Set `embedding_generator: "openrouter"` or `"openai"` in `config.json`
   - Set `embedding_model: "<model-name>"` (API model name)
   - Requires API key in `.env`

### Adding New Storage Backend

1. **New Vector Store**:
   - Implement `VectorIndex` protocol (`src/core/interfaces.py`)
   - Create adapter in `src/adapters/vector/`
   - Update `src/app/bootstrap.py` → `create_hybrid_retrieval()` to use new adapter

2. **New Message/Chunk Store**:
   - Implement `MessageStore` or `ChunkStore` protocol
   - Create adapter in `src/adapters/persistence/`
   - Update bootstrap to use new adapter

## Risks and TODO

### High Priority

1. **Error Handling**: Missing explicit timeouts for external API calls (OpenRouter, Voyage, Telegram)
   - **Risk**: Hanging requests, resource exhaustion
   - **Fix**: Add timeout parameters to HTTP clients

2. **Concurrent Access**: SQLite database may have issues with concurrent writes
   - **Risk**: Database locks, corruption
   - **Fix**: Ensure single writer or migrate to PostgreSQL

3. **Token Limit Handling**: Token limit errors trigger retry with context reset, but no exponential backoff
   - **Risk**: Infinite retry loops, API rate limiting
   - **Fix**: Add retry limits and exponential backoff

4. **Profile Isolation**: Profile switching requires restart (bot_instance is singleton)
   - **Risk**: Cannot switch profiles at runtime
   - **Fix**: Support profile switching without restart

5. **ChromaDB Version Compatibility**: ChromaDB API changes may break compatibility
   - **Risk**: Breaking changes in ChromaDB updates
   - **Fix**: Pin ChromaDB version, add compatibility tests

### Medium Priority

6. **Test Coverage**: Only 41% coverage (87 tests)
   - **Risk**: Regressions, bugs in untested code
   - **Fix**: Increase coverage, especially for core flows (retrieval, ingestion)

7. **FTS5 Availability**: FTS5 may not be available in all SQLite builds
   - **Risk**: FTS5 tables fail to create, fallback to basic search
   - **Fix**: Check FTS5 availability, provide fallback

8. **Embedding Dimension Mismatch**: No validation that embedding dimensions match across chunks
   - **Risk**: Vector search failures, inconsistent results
   - **Fix**: Validate embedding dimensions on ingestion

9. **Memory Usage**: Large chat histories may cause memory issues
   - **Risk**: OOM errors during ingestion
   - **Fix**: Stream processing, batch size limits

10. **Session Management**: Telegram session expiration not handled gracefully
    - **Risk**: Ingestion fails silently
    - **Fix**: Check session validity, prompt re-authentication

### Low Priority

11. **Logging**: Structured logging (syslog2) but no centralized log aggregation
    - **Risk**: Difficult to debug production issues
    - **Fix**: Add log aggregation (e.g., ELK stack)

12. **Metrics**: No performance metrics or monitoring
    - **Risk**: Cannot track performance degradation
    - **Fix**: Add metrics (Prometheus, custom metrics)

13. **Configuration Validation**: No validation of config.json values
    - **Risk**: Invalid config causes runtime errors
    - **Fix**: Add schema validation (JSON Schema, Pydantic)

14. **Migration System**: Auto-migration in `db.py._ensure_schema()` is fragile
    - **Risk**: Schema changes may fail silently
    - **Fix**: Use Alembic for proper migrations

15. **Chunking Strategy**: Token-based chunking may split related messages
    - **Risk**: Context loss, poor retrieval quality
    - **Fix**: Add semantic chunking (sentence boundaries, topic detection)

16. **Query Rewriting**: Query rephrasing may fail silently
    - **Risk**: Falls back to original query, no error notification
    - **Fix**: Add error handling and logging

17. **Admin Security**: Admin password stored in plaintext in config.json
    - **Risk**: Password exposure if config.json is leaked
    - **Fix**: Hash passwords, use secure storage

18. **Rate Limiting**: No rate limiting for LLM API calls
    - **Risk**: API rate limit exceeded, cost overruns
    - **Fix**: Add rate limiting middleware

19. **Vector Store Sync**: ChromaDB sync may fail silently
    - **Risk**: Inconsistent state between SQLite and ChromaDB
    - **Fix**: Add sync verification, retry logic

20. **Profile Backup**: No backup/restore mechanism for profiles
    - **Risk**: Data loss if profile directory is corrupted
    - **Fix**: Add backup/restore commands

---

**Generated**: 2024-12-19
**Repository**: legale-bot
**Analysis**: Static code analysis, no code modifications

