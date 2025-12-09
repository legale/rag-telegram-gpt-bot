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
- **No emojis**: Emojis are not used anywhere in the codebase, including this file
- **Status markers**: Use `[x]` for completed items, `[ ]` for incomplete items

## Implementation Status

### Completed Phases
- [x] **Phase 1-2**: Foundation, ingestion pipeline, core RAG
- [x] **Phase 3**: Bot core, CLI interface
- [x] **Phase 4**: Telegram bot (webhook, daemon, systemd)
- [x] **Phase 4.5**: Token management, context control
- [x] **Phase 5**: Test coverage (80%+)
- [x] **Phase 6**: Profile management, unified CLI
- [x] **Phase 8**: Admin panel (profiles, ingestion, stats, control, settings)
- [x] **Phase 10**: Code refactoring (utilities, BaseAdminCommand, all command classes)
- [x] **Phase 14.1-14.6**: Topic clustering core (L1/L2 clustering, naming, RAG integration, CLI commands)
- [x] **Phase 14.7 (CLI)**: Topic CLI commands (build, list, show, cluster-l1, cluster-l2, name)

### In Progress
- [ ] **Phase 14.7 (Admin)**: Admin panel commands for topics (list, show, rebuild, stats) - CLI commands ready, admin panel integration pending

### Planned Phases
- **Phase 15**: RAG quality (reranking, semantic chunking, query expansion, context compression)
- **Phase 16**: Data management (export, import, backup/restore)
- **Phase 17**: Analytics (usage, topics, users)
- **Phase 18**: Performance optimization (embeddings, DB, vector store)
- **Phase 19**: Advanced topics (merge/split, quality metrics, visualization)
- **Phase 20**: Enhanced RAG (multi-modal retrieval, contextual retrieval, response quality)

## Priority

### High
1. Phase 14.7 (Admin) - Topic admin commands in admin panel (CLI commands already implemented)
2. Phase 15.1 - Reranking
3. Phase 16.1 - Data export
4. Phase 19.1 - Manual topic operations

### Medium
- Phase 15.2 (semantic chunking), 17.1 (analytics), 18.1 (performance), 20.1 (multi-modal)

### Low
- Phase 15.3-15.4, 19.3, 20.3

## Current Metrics
- **Test Coverage**: 80%+ (40+ test files)
- **Code Quality**: Good (refactoring complete)
- **Features**: Core RAG + Topics (CLI) + Admin Panel
- **Documentation**: Complete

## Recent Updates
- **Topic CLI Commands**: Full CLI support for topic management (build, list, show, cluster-l1, cluster-l2, name)
- **Topic Integration**: Hierarchical topic-based retrieval integrated into RAG pipeline
- **Admin Panel**: Complete admin interface for profiles, ingestion, stats, control, and settings
