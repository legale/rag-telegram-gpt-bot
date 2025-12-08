# Librarian bot

## Project Goal
RAG LLM chatbot with hierarchical topic clustering for improved retrieval

## Architecture
Poetry-based Python 3.11+ application

### Components
1. **Data Ingestion**: Parser → Chunker → Embedder → Storage (SQLite + ChromaDB)
2. **Runtime System**: Query Handler → Retriever (hybrid: vector + topic-based) → Prompt Engine → LLM Client
3. **Topic Clustering**: HDBSCAN hierarchical clustering (L1/L2) with LLM-based naming

### Tech Stack
- **Language**: Python 3.11+
- **Database**: SQLite (chunks, messages, topics)
- **Vector Store**: ChromaDB (embeddings)
- **Clustering**: HDBSCAN
- **Logging**: syslog2 (custom logging system)
- **CLI Parsing**: Custom `cli_parser.py` (instead of argparse)
- **Testing**: Pytest (80%+ coverage)
- **Linting**: Ruff / Black / MyPy

## Coding Standards
- PEP 8 style, strict type hints, Google-style docstrings
- Custom exception hierarchy, graceful error handling

## Implementation Status

### ✅ Completed Phases
- **Phase 1-2**: Foundation, ingestion pipeline, core RAG
- **Phase 3**: Bot core, CLI interface
- **Phase 4**: Telegram bot (webhook, daemon, systemd)
- **Phase 4.5**: Token management, context control
- **Phase 5**: Test coverage (80%+)
- **Phase 6**: Profile management, unified CLI
- **Phase 8**: Admin panel (profiles, ingestion, stats, control, settings)
- **Phase 10**: Code refactoring (utilities, BaseAdminCommand, all command classes)
- **Phase 14.1-14.6**: Topic clustering core (L1/L2 clustering, naming, RAG integration, CLI commands)

### 🔄 In Progress
- **Phase 10**: `handle_message()` simplification, utility tests
- **Phase 14.7**: Admin commands for topics (list, show, rebuild, stats)

### 📋 Planned Phases
- **Phase 15**: RAG quality (reranking, semantic chunking, query expansion, context compression)
- **Phase 16**: Data management (export, import, backup/restore)
- **Phase 17**: Analytics (usage, topics, users)
- **Phase 18**: Performance optimization (embeddings, DB, vector store)
- **Phase 19**: Advanced topics (merge/split, quality metrics, visualization)
- **Phase 20**: Enhanced RAG (multi-modal retrieval, contextual retrieval, response quality)

## Priority

### High
1. Phase 14.7 - Topic admin commands
2. Phase 15.1 - Reranking
3. Phase 16.1 - Data export
4. Phase 19.1 - Manual topic operations

### Medium
- Phase 15.2 (semantic chunking), 17.1 (analytics), 18.1 (performance), 20.1 (multi-modal)

### Low
- Phase 15.3-15.4, 19.3, 20.3

## Current Metrics
- **Test Coverage**: 80%+ ✅
- **Code Quality**: Good (refactoring mostly complete)
- **Features**: Core RAG + Topics + Admin Panel ✅
- **Documentation**: Complete ✅
