# Librarian bot

## Project Goal
RAG LLM chatbot with hierarchical topic clustering for improved retrieval

## Architecture
poetry based python program

### Components
1.  **Data Ingestion Pipeline**:
    *   **Parser**: Reads the chat dump.
    *   **Chunker**: Splits text into logical blocks (e.g., by day or message count).
    *   **Embedder**: Converts text chunks into vector embeddings.
    *   **Storage**: Saves text chunks and vectors.

2.  **Runtime System**:
    *   **Query Handler**: Receives user input.
    *   **Retriever**: Searches the vector database for relevant context (hybrid: vector + topic-based).
    *   **Prompt Engine**: Constructs the system prompt with the persona and retrieved context.
    *   **LLM Client**: Interfaces with the external model API.

3.  **Topic Clustering**:
    *   **TopicClusterer**: Hierarchical clustering (L1/L2) using HDBSCAN.
    *   **Topic Naming**: LLM-based topic title and description generation.
    *   **Topic-based Retrieval**: Enhanced RAG with topic-aware context selection.

### Tech Stack
*   **Language**: Python 3.10+
*   **Database**: SQLite (relational) - stores raw text chunks, messages, and hierarchical topics.
*   **Vector Store**: ChromaDB (local) - stores embeddings for semantic search.
*   **Clustering**: HDBSCAN for hierarchical topic clustering
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
*   **Coverage**: Current coverage: **80%+** (target achieved)

## Implementation Plan

### Phase 1: Foundation & Ingestion ✅
**Status: COMPLETE**

- [x] **Project Setup**: Initialize git, poetry, directory structure.
- [x] **Data Parser**: Implement `ChatParser` to read the dump file.
- [x] **Chunking Logic**: Implement `TextChunker` to split data by day/size.
- [x] **Database Setup**: Design schema for storing chunks (ID, text, metadata).
- [x] **Vector Store Setup**: Initialize vector DB client.
- [x] **Ingestion Script**: Orchestrate parsing -> chunking -> embedding -> saving.

### Phase 2: Core Logic (RAG) ✅
**Status: COMPLETE**

- [x] **Embedding Service**:
    - Implement `EmbeddingClient` to interface with Openrouter (or compatible) API.
    - Add caching for embeddings to save costs/time.
    - Implement batch processing for large texts.
    - Support local embeddings (sentence-transformers).
- [x] **Retrieval Service**:
    - Implement `Retriever` class.
    - Add logic to fetch full text chunks from SQLite based on VectorDB IDs.
    - Implement similarity threshold filtering.
    - Hybrid retrieval: vector search + topic-based search.
- [x] **Prompt Engine**:
    - Create `PromptTemplate` class.
    - Design the "Union Lawyer" system prompt with placeholders for context and history.
    - Implement context truncation to fit context window.
    - Topic-aware context formatting.
- [x] **LLM Client**:
    - Implement `LLMClient` for chat completion (OpenAI API).
    - Add retry logic and error handling.
    - Support streaming responses (optional but good for UX).

### Phase 3: Application & Interface ✅
**Status: COMPLETE**

- [x] **Bot Core**:
    - Implement `LegaleBot` class orchestrating the RAG flow.
    - Add conversation memory (session history).
- [x] **Interfaces**:
    - [x] **CLI**: Interactive command-line interface for testing.
    - [ ] **API**: Basic FastAPI wrapper for potential frontend integration (low priority).

### Phase 4: Telegram Bot Integration (Webhook + Daemon) ✅
**Status: COMPLETE**

- [x] **Daemon Architecture**: Persistent memory, FastAPI, logging, graceful shutdown
- [x] **Telegram Bot Implementation**: Webhook, message handling, commands (`/start`, `/help`, `/reset`, `/tokens`, `/model`)
- [x] **CLI Utilities**: Register/delete webhook, run/daemon commands
- [x] **Daemonization**: systemd service, PID management, signal handlers
- [x] **Nginx Integration**: Config templates, SSL/TLS docs, health check
- [x] **Systemd Service**: Service files, init scripts, documentation
- [x] **Configuration & Documentation**: Complete setup guides

### Phase 4.5: Token Management & Context Control ✅
**Status: COMPLETE**

- [x] User commands (`/reset`, `/tokens`)
- [x] Automatic context management (auto-reset on token limit)
- [x] Token optimization (tiktoken, reduced limits)
- [x] All tests passing

### Phase 5: Test Coverage Improvement ✅
**Status: COMPLETE** (Target 80% achieved)

- [x] Core modules: 98%+ coverage (`core.py`, `llm.py`)
- [x] Bot handlers: 80%+ coverage (`tgbot.py`)
- [x] Utilities: 100% coverage (access_control, frequency_controller)
- [x] Overall coverage: **80%+** (target achieved)
- [x] 200+ tests passing

### Phase 6: CLI Orchestrator & Profile Management ✅
**Status: COMPLETE**

- [x] **Profile System**: Multiple bot instances with separate databases
- [x] **Unified CLI (`legale.py`)**: Single entry point for all operations
- [x] **Documentation**: CLI guide, workflows, examples

**Features:**
- [x] Multiple bot instances with separate data
- [x] Easy switching between environments (dev/prod/test)
- [x] Profile-specific Telegram sessions
- [x] Automatic default profile creation

### Phase 8: Admin Bot Interface (Telegram Admin Panel) ✅
**Status: COMPLETE**

- [x] **Admin Command Architecture**: Command router, permission checks, interactive menus
- [x] **Profile Management via Bot**: Create, switch, delete, info profiles
- [x] **Data Ingestion via Bot**: File upload, progress updates, status
- [x] **Bot Statistics & Monitoring**: Stats, health checks, logs
- [x] **Bot Control & Management**: Restart, configuration
- [x] **Advanced Configuration**: Chat whitelist, response frequency

**Remaining (Optional):**
- [ ] User management commands (`/admin users list/info/block`)
- [ ] Database operations (`/admin db backup/restore/vacuum`)
- [ ] Model management (`/admin models list/add/remove/set`)
- [ ] Webhook management (`/admin webhook status/register/delete`)
- [ ] Notifications & alerts system
- [ ] Interactive wizards
- [ ] Security & audit logging

### Phase 10: Code Refactoring & Quality Improvement 🔄
**Status: IN PROGRESS** (Partially Complete)

**Completed:**
- [x] Utility modules created (6 modules: formatter, db_stats, validator, access_control, frequency_controller, health_checker)
- [x] BaseAdminCommand class created
- [x] StatsCommands refactored (complexity reduced 50-80%)
- [x] Code duplication reduced ~75%

**Remaining:**
- [ ] Refactor remaining admin command classes (ProfileCommands, IngestCommands, ControlCommands, SettingsCommands)
- [ ] Refactor `handle_message()` in `tgbot.py` (complexity: 25 → 8)
- [ ] Refactor `AdminCommandRouter.route()` (complexity: 10 → 5)
- [ ] Unit tests for utilities
- [ ] Update documentation

### Phase 14: RAG Enhancement - Hierarchical Topic Clustering 🔄
**Status: IN PROGRESS** (Core Complete, Admin Commands Pending)

#### 14.1-14.6 Core Implementation ✅
- [x] Database Schema & Models (MessageModel, TopicL1Model, TopicL2Model)
- [x] Message Storage & Chunking
- [x] L1 Topic Clustering (HDBSCAN)
- [x] L2 Super-Topic Clustering
- [x] Topic Naming with LLM
- [x] RAG Integration (hybrid retrieval)
- [x] CLI Commands (`legale topics build/list/show`)
- [x] Comprehensive testing (14 tests passing)
- [x] Parameter tuning (metric, method, epsilon)
- [x] Automatic cosine metric normalization

#### 14.7 Remaining Tasks
- [ ] **Incremental Topic Updates**:
    - [ ] Detect new chunks since last clustering
    - [ ] Re-cluster only new chunks and merge with existing topics
    - [ ] Update topic centroids incrementally
    - [ ] CLI: `legale topics update` command
    
- [ ] **Admin Commands for Topics**:
    - [ ] `/admin topics list` - List all topics (L1/L2)
    - [ ] `/admin topics show <id>` - Show topic details
    - [ ] `/admin topics rebuild` - Rebuild topics with current parameters
    - [ ] `/admin topics stats` - Topic statistics (count, coverage, etc.)

- [ ] **Topic Quality Improvements**:
    - [ ] Topic merging (merge similar topics)
    - [ ] Topic splitting (split large topics)
    - [ ] Manual topic editing (rename, merge, split via admin)
    - [ ] Topic quality metrics (coherence, coverage)

---

## New Features & Improvements

### Phase 15: RAG Quality Improvements
**Status: PLANNED**
**Goal**: Improve retrieval quality and response accuracy

#### 15.1 Reranking
- [ ] **Cross-Encoder Reranking**:
    - [ ] Implement reranking using cross-encoder models
    - [ ] Rerank top-K retrieved chunks (e.g., 20 → 5)
    - [ ] Improve relevance of final context
    - [ ] Configurable reranking (enable/disable, model selection)

#### 15.2 Advanced Chunking Strategies
- [ ] **Semantic Chunking**:
    - [ ] Implement semantic chunking (split by meaning, not just size)
    - [ ] Overlap-based chunking for better context continuity
    - [ ] Sentence-aware chunking
    - [ ] Configurable chunking strategies per profile

#### 15.3 Query Expansion & Reformulation
- [ ] **Query Enhancement**:
    - [ ] Query expansion (synonyms, related terms)
    - [ ] Query reformulation using LLM
    - [ ] Multi-query retrieval (generate multiple query variants)
    - [ ] Improve recall for complex queries

#### 15.4 Context Compression
- [ ] **Context Summarization**:
    - [ ] Summarize retrieved chunks before sending to LLM
    - [ ] Extract key information from chunks
    - [ ] Reduce token usage while preserving relevance
    - [ ] Configurable compression level

### Phase 16: Data Management & Export
**Status: PLANNED**

#### 16.1 Data Export
- [ ] **Export Functionality**:
    - [ ] Export chunks to JSON/CSV
    - [ ] Export topics with associated chunks
    - [ ] Export conversation history
    - [ ] CLI: `legale export chunks/topics/history`
    - [ ] Admin: `/admin export <type>`

#### 16.2 Data Import
- [ ] **Import Functionality**:
    - [ ] Import chunks from external sources
    - [ ] Merge data from multiple profiles
    - [ ] Validate imported data
    - [ ] CLI: `legale import <file>`

#### 16.3 Data Backup & Restore
- [ ] **Backup System**:
    - [ ] Full profile backup (DB + vector store)
    - [ ] Incremental backups
    - [ ] Backup compression
    - [ ] CLI: `legale backup create/restore/list`
    - [ ] Admin: `/admin backup create/restore`

### Phase 17: Analytics & Monitoring
**Status: PLANNED**

#### 17.1 Usage Analytics
- [ ] **Query Analytics**:
    - [ ] Track query patterns
    - [ ] Most common queries
    - [ ] Query success rate
    - [ ] Response time metrics
    - [ ] Admin: `/admin analytics queries`

#### 17.2 Topic Analytics
- [ ] **Topic Usage**:
    - [ ] Track which topics are retrieved most often
    - [ ] Topic coverage statistics
    - [ ] Topic quality metrics
    - [ ] Admin: `/admin analytics topics`

#### 17.3 User Analytics
- [ ] **User Behavior**:
    - [ ] Active users tracking
    - [ ] User query history
    - [ ] User engagement metrics
    - [ ] Admin: `/admin analytics users`

### Phase 18: Performance Optimization
**Status: PLANNED**

#### 18.1 Embedding Optimization
- [ ] **Caching & Batching**:
    - [ ] Improve embedding cache hit rate
    - [ ] Batch embedding requests
    - [ ] Async embedding generation
    - [ ] Reduce API costs

#### 18.2 Database Optimization
- [ ] **Query Optimization**:
    - [ ] Add database indexes for common queries
    - [ ] Optimize topic queries
    - [ ] Connection pooling
    - [ ] Query result caching

#### 18.3 Vector Store Optimization
- [ ] **ChromaDB Tuning**:
    - [ ] Optimize collection settings
    - [ ] Batch operations
    - [ ] Index optimization
    - [ ] Memory usage optimization

### Phase 19: Advanced Topic Features
**Status: PLANNED**

#### 19.1 Topic Management
- [ ] **Manual Topic Operations**:
    - [ ] Merge topics (combine similar topics)
    - [ ] Split topics (break large topics)
    - [ ] Rename topics manually
    - [ ] Delete topics
    - [ ] Admin: `/admin topics merge/split/rename/delete`

#### 19.2 Topic Quality Metrics
- [ ] **Quality Assessment**:
    - [ ] Topic coherence score
    - [ ] Topic coverage (percentage of chunks covered)
    - [ ] Topic size distribution
    - [ ] Orphan chunk detection
    - [ ] Admin: `/admin topics quality`

#### 19.3 Topic Visualization
- [ ] **Visualization Tools**:
    - [ ] Topic hierarchy visualization
    - [ ] Topic similarity graph
    - [ ] Export topic visualization (PNG/SVG)
    - [ ] CLI: `legale topics visualize`

### Phase 20: Enhanced RAG Features
**Status: PLANNED**

#### 20.1 Multi-Modal Retrieval
- [ ] **Advanced Retrieval**:
    - [ ] Keyword + semantic hybrid search
    - [ ] Date-based filtering
    - [ ] Author-based filtering
    - [ ] Topic-based filtering UI

#### 20.2 Contextual Retrieval
- [ ] **Context-Aware Retrieval**:
    - [ ] Use conversation history to improve retrieval
    - [ ] Follow-up question handling
    - [ ] Context window management
    - [ ] Dynamic context selection

#### 20.3 Response Quality
- [ ] **Response Enhancement**:
    - [ ] Citation/source tracking
    - [ ] Confidence scores
    - [ ] Response validation
    - [ ] Multi-turn conversation optimization

---

## Deprecated/Removed Phases

### ~~Phase 7: Multi-Bot Support~~ 
**Status: DEPRECATED** (Low Priority)
- Profile system already provides isolation
- Multi-bot daemon architecture is complex and not needed for current use case
- Can be revisited if there's actual need for multiple simultaneous bots

### ~~Phase 9: Testing & Stabilization~~
**Status: MERGED INTO Phase 5**
- Test coverage already at 80%+
- Remaining tasks are optional improvements

### ~~Phase 11: Test Coverage Improvement~~
**Status: MERGED INTO Phase 5**
- Duplicate of Phase 5
- Coverage target already achieved

### ~~Phase 12: Coverage Analysis~~
**Status: COMPLETE**
- 80% coverage achieved
- No further action needed

---

## Implementation Priority

### High Priority (Next Steps)
1. **Phase 14.7** - Admin commands for topics (complete topic management)
2. **Phase 15.1** - Reranking for better retrieval quality
3. **Phase 16.1** - Data export functionality
4. **Phase 19.1** - Manual topic operations (merge/split)

### Medium Priority
1. **Phase 15.2** - Advanced chunking strategies
2. **Phase 17.1** - Usage analytics
3. **Phase 18.1** - Performance optimization
4. **Phase 20.1** - Multi-modal retrieval

### Low Priority (Nice to Have)
1. **Phase 15.3** - Query expansion
2. **Phase 15.4** - Context compression
3. **Phase 19.3** - Topic visualization
4. **Phase 20.3** - Response quality enhancements

---

## Current Status Summary

**Completed Phases:** 1, 2, 3, 4, 4.5, 5, 6, 8 (core), 14 (core)
**In Progress:** Phase 10 (refactoring), Phase 14.7 (admin commands)
**Planned:** Phases 15-20 (new features)

**Key Metrics:**
- Test Coverage: **80%+** ✅
- Code Quality: Good (refactoring in progress)
- Features: Core RAG + Topics + Admin Panel ✅
- Documentation: Complete ✅
