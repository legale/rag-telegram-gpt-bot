# Legale Bot - Architectural Overview

## 1. Quick Start & Purpose
**Legale Bot** is a RAG-based (Retrieval Augmented Generation) Telegram bot designed to answer questions based on chat history. It features a hybrid retrieval system (Full-Text Search + Vector Search) and a modular ingestion pipeline.

### Main Execution
The central entry point is **`legale.py`** in the root directory.

- **Check Version**: `./legale.py --version`
- **Ingest Data**: `./legale.py ingest all -file <dump.json>`
- **Run Chat (CLI)**: `./legale.py chat`
- **Run Bot (Telegram)**: `./legale.py bot run`

### Key Entities
- **Profile**: A named configuration environment (default: `default`). Contains its own DB, Vector DB, and Config.
- **Messages**: Raw Telegram messages imported from JSON dumps.
- **Chunks**: Split/aggregated messages used for embedding and retrieval.
- **Embeddings**: Vector representations of chunks (stored in SQLite + ChromaDB).

---

## 2. Project Structure
The source code is located in `src/`.

```text
.
├── legale.py               # Main CLI Entry Point (Orchestrator)
├── src/
│   ├── app/                # Application wiring & CLI Command Handlers
│   ├── bot/                # Telegram Bot logic (FastAPI + python-telegram-bot)
│   ├── core/               # Domain Logic (RAG, Embeddings, LLM)
│   ├── ingestion/          # Data Pipeline (Parse -> Chunk -> Embed)
│   ├── storage/            # Data Access Layer (SQLite, ChromaDB)
│   └── lib/                # Shared Utilities (Logging, ArgParse)
└── profiles/               # Data directory (DBs, Configs per profile)
```

### Component Roles
- **`src/app`**: Bridges the CLI (`legale.py`) with domain logic. Handles high-level commands like `cmd_chat`, `cmd_ingest`.
- **`src/bot`**: Implements the Telegram interface. Uses **FastAPI** for webhooks and **python-telegram-bot** for bot logic.
- **`src/core`**: The brain. Contains `HybridRetrievalService`, `LLMClient`, and `EmbeddingClient`.
- **`src/ingestion`**: ETL pipeline. Parses JSON dumps, chunks text, generates embeddings.
- **`src/storage`**: Abstractions for `Database` (SQLite) and `VectorStore` (ChromaDB).

---

## 3. Runtime Interactions

### A. Chat / RAG Flow ("Happy Path")
1. **User input** received via Telegram Webhook (`src/bot/tgbot.py`) or CLI (`src/bot/cli.py`).
2. **Dispatch**: Request routed to `LegaleBot.chat()`.
3. **Retrieval** (`src/core/hybrid_retrieval.py`):
    - **FTS5 Search**: SQLite finds keyword matches.
    - **Vector Reranking**: If configured, re-ranks FTS candidates using vector similarity.
    - **Context Packing**: Selects best chunks to fit token context.
4. **Generation** (`src/core/llm.py`):
    - Prompts constructed with System Prompt + Context + User query.
    - Sent to LLM (OpenAI/OpenRouter) via `LLMClient`.
5. **Response**: Answer returned to user.

### B. Ingestion Pipeline
Command: `legale ingest all -file dump.json`
1. **Stage 0 - Parsing** (`src/ingestion/parser.py`): Reads JSON, saves raw messages to SQLite (`messages` table).
2. **Stage 1 - Chunking** (`src/ingestion/chunker.py`): Groups messages into semantic chunks (token-based), saves to SQLite (`chunks` table).
3. **Stage 2 - Embedding** (`src/ingestion/pipeline.py`): Computes vectors for chunks using `sentence-transformers` (Local) or OpenAI API. Saves JSON embeddings to SQLite.
4. **Stage 3 - Indexing** (`src/ingestion/pipeline.py`): Syncs embeddings from SQLite to ChromaDB for fast similarity search.
5. **Stage 4 - Alias Discovery** (`src/ingestion/pipeline.py`): Identifies unique users and uses LLM to discover/save aliases for improved user profile matching.

---

## 4. Data & State

### Database (SQLite)
File: `profiles/<profile>/legale_bot.db`
- `messages`: Raw chat history.
- `chunks`: Useable text fragments with metadata (timeline, participating arrays).
- Stores configuration overrides if implemented in DB.

### Vector Schema (ChromaDB)
Directory: `profiles/<profile>/chroma_db`
- Collection: `embed-chunks`.
- Stores: `id`, `vector`, `metadata` (minimal).

### Configuration
1. **Environment**: `.env` (API Keys: `OPENROUTER_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_API_ID`).
2. **Profile Config**: `profiles/<profile>/config.json` (Model selection, Chunking params, System prompts).

---

## 5. Dependencies & Integrations

### External Services
- **LLM Provider**: OpenRouter or OpenAI (configured via env/config).
- **Telegram**: Telegram Bot API (via `python-telegram-bot`).
- **Telegram Data**: Telethon (used in `legale telegram` for dumping history).

### Key Libraries
- **FastAPI / Uvicorn**: Web server for Webhooks.
- **SQLAlchemy**: ORM for SQLite.
- **ChromaDB**: Vector search engine.
- **Tiktoken**: Token counting.
- **Sentence-Transformers**: Local embedding generation.

---

## 6. Risks & Fragile Points
- **Hybrid Retrieval Workarounds**: Code in `hybrid_retrieval.py` mentions compatibility hacks (e.g. `fts_index.search_chunks` vs `search`).
- **Environment Dependency**: Heavily relies on `.env` existing. Explicit checks present but critical for setup.
- **Poetry Shim**: `legale.py` attempts to re-exec itself with `poetry run` if not in venv. This can cause confusing process trees or signal handling issues.
- **Sync/Async Mix**: The bot uses `asyncio` (`python-telegram-bot`/`fastapi`) but some core logic (like `ingestion`) appears synchronous.

## 7. Extension Points
- **New Commands**: Register in `src/bot/admin_router.py` or `src/app/main_cli.py`.
- **New Backend**: Extend `VectorIndex` protocol in `src/core/interfaces.py` and implement in `src/storage`.
