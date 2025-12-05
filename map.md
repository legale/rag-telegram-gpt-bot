# Legale Bot - Union Lawyer Chatbot

## Project Goal
Create a chatbot that acts as an IT union lawyer, using a RAG (Retrieval-Augmented Generation) pipeline based on a chat dump history.

## Architecture

### Components
1.  **Data Ingestion Pipeline**:
    *   **Parser**: Reads the chat dump.
    *   **Chunker**: Splits text into logical blocks (e.g., by day or message count).
    *   **Embedder**: Converts text chunks into vector embeddings.
    *   **Storage**: Saves text chunks and vectors.

2.  **Runtime System**:
    *   **Query Handler**: Receives user input.
    *   **Retriever**: Searches the vector database for relevant context.
    *   **Prompt Engine**: Constructs the system prompt with the persona and retrieved context.
    *   **LLM Client**: Interfaces with the external model API.

### Tech Stack
*   **Language**: Python 3.10+
*   **Database**: SQLite (relational) - stores raw text chunks and metadata.
*   **Vector Store**: ChromaDB (local) - stores embeddings for semantic search.
*   **Testing**: Pytest
*   **Linting/Formatting**: Ruff / Black / MyPy

## Coding Standards
*   **Style**: Follow PEP 8. Use `black` for formatting.
*   **Typing**: Strict type hints in all function signatures.
*   **Documentation**: Google-style docstrings for all modules, classes, and functions.
*   **Error Handling**: Custom exception hierarchy. Fail gracefully.

## Testing Strategy
*   **Unit Tests**: Cover all core logic (chunking, prompt generation) with `pytest`.
*   **Mocking**: Mock all external API calls (LLM, Vector DB) during tests.
*   **Coverage**: Aim for high test coverage on business logic.

## Implementation Plan

### Phase 1: Foundation & Ingestion
- [x] **Project Setup**: Initialize git, poetry/pipenv, directory structure.
- [x] **Data Parser**: Implement `ChatParser` to read the dump file.
- [x] **Chunking Logic**: Implement `TextChunker` to split data by day/size.
- [x] **Database Setup**: Design schema for storing chunks (ID, text, metadata).
- [x] **Vector Store Setup**: Initialize vector DB client.
- [x] **Ingestion Script**: Orchestrate parsing -> chunking -> embedding -> saving.

### Phase 2: Core Logic (RAG)
- [x] **Embedding Service**:
    - Implement `EmbeddingClient` to interface with Openrouter (or compatible) API.
    - Add caching for embeddings to save costs/time.
    - Implement batch processing for large texts.
- [x] **Retrieval Service**:
    - Implement `Retriever` class.
    - Add logic to fetch full text chunks from SQLite based on VectorDB IDs.
    - Implement similarity threshold filtering.
- [x] **Prompt Engine**:
    - Create `PromptTemplate` class.
    - Design the "Union Lawyer" system prompt with placeholders for context and history.
    - Implement context truncation to fit context window.
- [x] **LLM Client**:
    - Implement `LLMClient` for chat completion (OpenAI API).
    - Add retry logic and error handling.
    - Support streaming responses (optional but good for UX).

### Phase 3: Application & Interface
- [x] **Bot Core**:
    - Implement `LegaleBot` class orchestrating the RAG flow.
    - Add conversation memory (session history).
- [/] **Interfaces**:
    - [x] **CLI**: Interactive command-line interface for testing.
    - [ ] **API**: Basic FastAPI wrapper for potential frontend integration.
- [ ] **Evaluation & Verification**:
    - Create a set of "golden" questions and answers.
    - Implement an evaluation script to check retrieval quality.

### Phase 4: Telegram Bot Integration (Webhook + Daemon)
- [ ] **Daemon Architecture**:
    - [ ] Design persistent memory architecture (keep DB/VectorStore loaded)
    - [ ] Choose web framework (FastAPI + uvicorn)
    - [ ] Design logging strategy (syslog for daemon, stdout for foreground)
    - [ ] Plan graceful shutdown handling
- [ ] **Telegram Bot Implementation**:
    - [ ] Implement `src/bot/tgbot.py` main module
    - [ ] Create FastAPI app with `/webhook` endpoint
    - [ ] Implement Telegram update parser
    - [ ] Add message handler (text messages → LegaleBot.chat())
    - [ ] Add `/start` and `/help` commands
    - [ ] Implement persistent LegaleBot instance (singleton pattern)
- [ ] **CLI Utilities**:
    - [ ] Implement `register` command (webhook registration)
    - [ ] Implement `delete` command (webhook deletion)
    - [ ] Implement `run` command (foreground mode with -v/-vv/-vvv)
    - [ ] Implement `daemon` command (background mode)
    - [ ] Add argument parsing with `argparse`
- [ ] **Daemonization**:
    - [ ] Implement daemon mode using `python-daemon` or systemd
    - [ ] Configure syslog logging for daemon mode
    - [ ] Implement PID file management
    - [ ] Add signal handlers (SIGTERM, SIGINT for graceful shutdown)
- [ ] **Nginx Integration**:
    - [ ] Create nginx config template (`nginx/telegram-bot.conf`)
    - [ ] Document SSL/TLS requirements
    - [ ] Document reverse proxy setup
    - [ ] Add health check endpoint (`/health`)
- [ ] **Systemd Service**:
    - [ ] Create systemd service file (`systemd/legale-bot.service`)
    - [ ] Document service installation steps
    - [ ] Document service management commands
- [ ] **Configuration & Documentation**:
    - [ ] Add `TELEGRAM_BOT_TOKEN` to `.env.example`
    - [ ] Update `README.md` with Telegram bot setup instructions
    - [ ] Update `pyproject.toml` with new dependencies
    - [ ] Document webhook registration process
- [ ] **Testing & Verification**:
    - [ ] Write unit tests for webhook handler
    - [ ] Write integration tests for daemon startup/shutdown
    - [ ] Test foreground mode with verbosity levels
    - [ ] Test daemon mode with syslog
    - [ ] Performance test (10 rapid messages)
    - [ ] Test graceful shutdown
    - [ ] Verify persistent memory (no DB reload on requests)
