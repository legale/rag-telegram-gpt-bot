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
- [x] **Daemon Architecture**:
    - [x] Design persistent memory architecture (keep DB/VectorStore loaded)
    - [x] Choose web framework (FastAPI + uvicorn)
    - [x] Design logging strategy (syslog for daemon, stdout for foreground)
    - [x] Plan graceful shutdown handling
- [x] **Telegram Bot Implementation**:
    - [x] Implement `src/bot/tgbot.py` main module
    - [x] Create FastAPI app with `/webhook` endpoint
    - [x] Implement Telegram update parser
    - [x] Add message handler (text messages → LegaleBot.chat())
    - [x] Add `/start` and `/help` commands
    - [x] Add `/reset` command (reset conversation context)
    - [x] Add `/tokens` command (show token usage statistics)
    - [x] Add `/model` command (cycle through available LLM models from models.txt)
    - [x] Implement persistent LegaleBot instance (singleton pattern)
- [x] **CLI Utilities**:
    - [x] Implement `register` command (webhook registration)
    - [x] Implement `delete` command (webhook deletion)
    - [x] Implement `run` command (foreground mode with -v/-vv/-vvv)
    - [x] Implement `daemon` command (background mode)
    - [x] Add argument parsing with `argparse`
- [x] **Daemonization**:
    - [x] Implement daemon mode using `python-daemon` or systemd
    - [x] Configure syslog logging for daemon mode
    - [x] Implement PID file management
    - [x] Add signal handlers (SIGTERM, SIGINT for graceful shutdown)
- [x] **Nginx Integration**:
    - [x] Create nginx config template (`nginx/telegram-bot.conf`)
    - [x] Document SSL/TLS requirements
    - [x] Document reverse proxy setup
    - [x] Add health check endpoint (`/health`)
- [x] **Systemd Service**:
    - [x] Create systemd service file (`systemd/legale-bot.service`)
    - [x] Create SysV init script (`init.d/legale-bot`)
    - [x] Document service installation steps
    - [x] Document service management commands
- [x] **Configuration & Documentation**:
    - [x] Add `TELEGRAM_BOT_TOKEN` to `.env.example`
    - [x] Update `README.md` with Telegram bot setup instructions
    - [x] Update `pyproject.toml` with new dependencies
    - [x] Document webhook registration process
    - [x] Add Quick Start guide with complete setup workflow
- [/] **Testing & Verification**:
    - [x] Write unit tests for webhook handler
    - [ ] Write integration tests for daemon startup/shutdown
    - [ ] Test foreground mode with verbosity levels
    - [ ] Test daemon mode with syslog
    - [ ] Performance test (10 rapid messages)
    - [ ] Test graceful shutdown
    - [ ] Verify persistent memory (no DB reload on requests)

### Phase 4.5: Token Management & Context Control ✅
**Status: COMPLETE**

- [x] **User Commands**:
    - [x] `/reset` command - clear conversation context
    - [x] `/tokens` command - show token usage statistics
    - [x] Update `/help` with new commands

- [x] **Automatic Context Management**:
    - [x] Auto-reset when reaching `MAX_CONTEXT_TOKENS` limit
    - [x] Configurable via `.env` (default: 14000 for gpt-3.5-turbo)
    - [x] Warning message when auto-reset occurs

- [x] **Token Optimization**:
    - [x] Add `tiktoken` library for accurate token counting
    - [x] Implement `count_tokens()` method in `LLMClient`
    - [x] Reduce `max_tokens`: 1000 → 500 (shorter responses)
    - [x] Reduce `n_results`: 5 → 3 (fewer context chunks)
    - [x] Add `max_context_chars`: 8000 limit

- [x] **Methods Added**:
    - [x] `LegaleBot.reset_context()` - clear chat history
    - [x] `LegaleBot.get_token_usage()` - return usage stats
    - [x] `LLMClient.count_tokens()` - count tokens in messages

- [x] **Testing**:
    - [x] Fix failing tests (test_prompt.py, test_retrieval.py)
    - [x] All 22 tests passing ✅

**Result**: Solved token limit error (17,336 > 16,385):
- Input tokens: ~12,000 (was 16,336)
- Output tokens: 500 (was 1,000)
- **Total: ~12,500 < 16,385** ✅

### Phase 5: Test Coverage Improvement
**Current Coverage: 41% (787 lines, 465 uncovered)** ⬆️ from 25%
**Target Coverage: 75%+**
**Tests: 87 (was 22)** ✅

**Progress:**
- ✅ `src/bot/core.py`: **98%** coverage (was 0%)
- ✅ `src/core/llm.py`: **100%** coverage (was 0%)
- ✅ **Milestone 1 COMPLETE**: Core modules to 50%+

#### 5.1 Core Bot Logic Tests (`src/bot/core.py` - 0% → 98% ✅)
- [x] **Model Management Tests** (`test_bot_core_models.py`):
    - [x] Test `_load_available_models()` with valid models.txt
    - [x] Test `_load_available_models()` with missing models.txt (fallback)
    - [x] Test `_load_available_models()` with empty models.txt
    - [x] Test `switch_model()` - single model in list
    - [x] Test `switch_model()` - multiple models cycling
    - [x] Test `switch_model()` - verify LLM client recreation
    - [x] Test `switch_model()` - cyclic behavior (wrap around)
    - [x] Test `get_current_model()` - correct model and position
    - [x] Test initial model index detection
- [x] **Context Management Tests** (`test_bot_core_context.py`):
    - [x] Test `reset_context()` - clears history
    - [x] Test `reset_context()` - returns confirmation message
    - [x] Test `get_token_usage()` - empty history
    - [x] Test `get_token_usage()` - with history
    - [x] Test `get_token_usage()` - percentage calculation
    - [x] Test token usage thresholds (0%, 50%, 80%, 100%)
- [x] **Chat Flow Tests** (`test_bot_core_chat.py`):
    - [x] Test `chat()` - basic query with mocked retrieval
    - [x] Test `chat()` - history accumulation
    - [x] Test `chat()` - auto-reset on token limit
    - [x] Test `chat()` - context retrieval integration
    - [x] Test `chat()` - prompt construction
    - [x] Test `chat()` - LLM response handling
    - [x] Test `chat()` - error handling (LLM failure)
    - [x] Test `chat()` - error handling (retrieval failure)

#### 5.2 LLM Client Tests (`src/core/llm.py` - 0% → 100% ✅)
- [x] **Initialization Tests** (`test_llm.py`):
    - [x] Test `__init__()` - with OPENROUTER_API_KEY
    - [x] Test `__init__()` - with OPENAI_API_KEY
    - [x] Test `__init__()` - missing API key (should raise)
    - [x] Test `__init__()` - custom base_url
    - [x] Test `__init__()` - tokenizer initialization (known model)
    - [x] Test `__init__()` - tokenizer fallback (unknown model)
- [x] **Token Counting Tests** (`test_llm.py`):
    - [x] Test `count_tokens()` - empty messages
    - [x] Test `count_tokens()` - single message
    - [x] Test `count_tokens()` - multiple messages
    - [x] Test `count_tokens()` - long messages
    - [x] Test `count_tokens()` - special characters
- [x] **Completion Tests** (`test_llm.py`):
    - [x] Test `complete()` - successful response (mocked)
    - [x] Test `complete()` - with custom temperature
    - [x] Test `complete()` - with custom max_tokens
    - [x] Test `complete()` - empty response handling
    - [x] Test `complete()` - API error handling
    - [x] Test `complete()` - network timeout handling
    - [x] Test `complete()` - verbosity levels (0, 1, 2, 3)
- [x] **Streaming Tests** (`test_llm.py`):
    - [x] Test `stream_complete()` - successful streaming (mocked)
    - [x] Test `stream_complete()` - chunk assembly
    - [x] Test `stream_complete()` - error during streaming

#### 5.3 Telegram Bot Handler Tests (`src/bot/tgbot.py` - 0% → 70%+)
- [ ] **Command Handler Tests** (`test_tgbot_commands.py`):
    - [ ] Test `/start` command - response format
    - [ ] Test `/help` command - includes all commands
    - [ ] Test `/reset` command - calls bot_instance.reset_context()
    - [ ] Test `/reset` command - error handling
    - [ ] Test `/tokens` command - displays usage correctly
    - [ ] Test `/tokens` command - warning at 80%+
    - [ ] Test `/tokens` command - info at 50%+
    - [ ] Test `/tokens` command - OK at <50%
    - [ ] Test `/tokens` command - error handling
    - [ ] Test `/model` command - switches model
    - [ ] Test `/model` command - displays new model info
    - [ ] Test `/model` command - error handling
- [ ] **Message Handler Tests** (`test_tgbot_messages.py`):
    - [ ] Test regular message - calls bot_instance.chat()
    - [ ] Test regular message - sends response
    - [ ] Test message with error - sends error message
    - [ ] Test message logging (different verbosity)
- [ ] **Webhook Tests** (`test_tgbot_webhook.py`):
    - [ ] Test webhook endpoint - valid update
    - [ ] Test webhook endpoint - malformed JSON
    - [ ] Test webhook endpoint - missing message
    - [ ] Test webhook endpoint - non-text message
- [ ] **Lifecycle Tests** (`test_tgbot_lifecycle.py`):
    - [ ] Test lifespan startup - bot_instance created
    - [ ] Test lifespan startup - telegram_app initialized
    - [ ] Test lifespan startup - missing token (should raise)
    - [ ] Test lifespan shutdown - graceful cleanup
    - [ ] Test health endpoint - bot loaded
    - [ ] Test health endpoint - bot not loaded

#### 5.4 CLI Tests (`src/bot/cli.py` - 0% → 60%+)
- [ ] **CLI Interaction Tests** (`test_cli.py`):
    - [ ] Test CLI startup - bot initialization
    - [ ] Test CLI - user query processing
    - [ ] Test CLI - exit command
    - [ ] Test CLI - verbosity levels (-v, -vv, -vvv)
    - [ ] Test CLI - custom chunks parameter
    - [ ] Test CLI - error handling

#### 5.5 Ingestion Pipeline Tests (60% → 85%+)
- [ ] **Pipeline Tests** (`test_pipeline_extended.py`):
    - [ ] Test pipeline with large dataset (1000+ messages)
    - [ ] Test pipeline with --clear flag
    - [ ] Test pipeline error handling (invalid JSON)
    - [ ] Test pipeline progress reporting
    - [ ] Test pipeline with different chunk sizes
- [ ] **Parser Tests** (`test_parser_extended.py`):
    - [ ] Test parse_file with malformed JSON
    - [ ] Test parse_file with missing fields
    - [ ] Test parse_file with different message types
    - [ ] Test parse_file with Unicode characters

#### 5.6 Database Tests (61% → 85%+)
- [ ] **Database Extended Tests** (`test_db_extended.py`):
    - [ ] Test add_chunk() - duplicate handling
    - [ ] Test get_chunk() - non-existent chunk
    - [ ] Test get_chunks() - multiple chunks
    - [ ] Test get_chunks() - empty database
    - [ ] Test database persistence
    - [ ] Test concurrent access

#### 5.7 Integration Tests
- [ ] **End-to-End Tests** (`test_e2e.py`):
    - [ ] Test full ingestion → query flow
    - [ ] Test model switching during conversation
    - [ ] Test context reset during conversation
    - [ ] Test token limit auto-reset
    - [ ] Test multiple concurrent users (if applicable)

#### 5.8 Test Infrastructure
- [ ] **Test Utilities** (`tests/utils.py`):
    - [ ] Create mock LLM client factory
    - [ ] Create mock Telegram update factory
    - [ ] Create test data fixtures (sample messages)
    - [ ] Create temporary database helper
    - [ ] Create temporary models.txt helper
- [ ] **CI/CD Integration**:
    - [ ] Add GitHub Actions workflow for tests
    - [ ] Add coverage reporting to CI
    - [ ] Add coverage badge to README
    - [ ] Set minimum coverage threshold (75%)

#### Coverage Milestones
- [x] **Milestone 1**: Core modules to 50%+ (src/bot/core.py 98%, src/core/llm.py 100%) ✅
- [ ] **Milestone 2**: Bot handlers to 70% (src/bot/tgbot.py)
- [ ] **Milestone 3**: Ingestion to 85% (src/ingestion/*)
- [ ] **Milestone 4**: Overall coverage to 75%+
- [ ] **Milestone 5**: Add integration and E2E tests

### Phase 6: CLI Orchestrator & Profile Management ✅
**Status: COMPLETE**

- [x] **Profile System**:
    - [x] Create `ProfileManager` class
    - [x] Implement profile directory structure (`profiles/<profile_name>/`)
    - [x] Store database, vector store, and session files per profile
    - [x] Track active profile in `.env` file (`ACTIVE_PROFILE`)
    - [x] Profile commands: create, list, switch, delete, info

- [x] **Unified CLI (`legale.py`)**:
    - [x] Single entry point for all bot operations
    - [x] Profile management commands
    - [x] Data ingestion with profile support
    - [x] Telegram fetching with profile-specific sessions
    - [x] Interactive chat with profile selection
    - [x] Bot webhook management with profiles
    - [x] Comprehensive help and examples

- [x] **Documentation**:
    - [x] Create `docs/CLI_GUIDE.md` with quick reference
    - [x] Update `.env.example` with `ACTIVE_PROFILE`
    - [x] Update `.gitignore` for profiles directory
    - [x] Add common workflows and examples

**Features:**
- ✅ Multiple bot instances with separate data
- ✅ Easy switching between environments (dev/prod/test)
- ✅ Profile-specific Telegram sessions
- ✅ Automatic default profile creation
- ✅ Self-documenting CLI with built-in help
- ✅ Consistent interface across all operations

**Usage:**
```bash
# Create and use a profile
legale profile create mybot --set-active
legale telegram dump "Chat" --limit 10000
legale ingest telegram_dump.json
legale chat -vv

# Switch profiles
legale profile switch production
legale bot run
```

### Phase 7: Multi-Bot Support (Multiple Telegram Bots)
**Status: PLANNED**
**Goal**: Support running multiple Telegram bots simultaneously, each with their own profile and configuration.

#### 7.1 Profile-Specific Bot Configuration
- [ ] **Bot Token Storage**:
    - [ ] Move `TELEGRAM_BOT_TOKEN` from global `.env` to profile-specific config
    - [ ] Create `profiles/<profile>/config.env` for profile-specific settings
    - [ ] Keep global `.env` for shared settings (API keys, etc.)
    - [ ] Add `bot_token` field to profile metadata
    - [ ] Update `ProfileManager.get_profile_paths()` to include config file

- [ ] **Profile Config Structure**:
    ```
    profiles/
    └── <profile_name>/
        ├── config.env              # Profile-specific config (NEW)
        │   ├── TELEGRAM_BOT_TOKEN
        │   ├── BOT_NAME
        │   ├── BOT_DESCRIPTION
        │   └── WEBHOOK_URL (optional)
        ├── legale_bot.db
        ├── chroma_db/
        └── telegram_session.session
    ```

- [ ] **Config Management**:
    - [ ] `ProfileManager.create_profile()` - create default config.env
    - [ ] `ProfileManager.get_bot_config()` - load profile config
    - [ ] `ProfileManager.set_bot_config()` - update profile config
    - [ ] Merge global .env with profile config.env (profile overrides global)

#### 7.2 Bot Management Commands
- [ ] **`legale bot` Subcommands**:
    - [ ] `legale bot add <profile>` - register new bot with profile
        - [ ] Prompt for bot token (or --token flag)
        - [ ] Optionally set webhook URL
        - [ ] Store in profile config.env
        - [ ] Validate token with Telegram API
        - [ ] Show bot info (username, name)
    
    - [ ] `legale bot remove <profile>` - unregister bot
        - [ ] Delete webhook from Telegram
        - [ ] Remove bot token from config
        - [ ] Optionally delete entire profile (--delete-profile)
    
    - [ ] `legale bot list` - list all configured bots
        - [ ] Show profile name, bot username, status (active/inactive)
        - [ ] Show webhook URL if configured
        - [ ] Indicate which bots are currently running
    
    - [ ] `legale bot info <profile>` - show bot details
        - [ ] Bot username, name, description
        - [ ] Webhook status
        - [ ] Database stats (chunks count)
        - [ ] Last activity timestamp

#### 7.3 Multi-Bot Daemon Architecture
- [ ] **Process Management**:
    - [ ] Design multi-process architecture (one process per bot)
    - [ ] Create `BotManager` class to manage multiple bot processes
    - [ ] Implement process spawning/monitoring
    - [ ] Add graceful shutdown for all bots
    - [ ] Implement health checks for each bot process

- [ ] **Port Allocation**:
    - [ ] Auto-assign ports for each bot (8000, 8001, 8002, ...)
    - [ ] Store port mapping in profile config
    - [ ] Update nginx config generation for multiple bots
    - [ ] Add port conflict detection

- [ ] **Daemon Commands**:
    - [ ] `legale daemon start [profile]` - start bot(s)
        - [ ] Start specific bot if profile specified
        - [ ] Start all configured bots if no profile
        - [ ] Check for port conflicts
        - [ ] Create PID files per bot
    
    - [ ] `legale daemon stop [profile]` - stop bot(s)
        - [ ] Stop specific bot if profile specified
        - [ ] Stop all bots if no profile
        - [ ] Graceful shutdown with timeout
    
    - [ ] `legale daemon restart [profile]` - restart bot(s)
    
    - [ ] `legale daemon status` - show status of all bots
        - [ ] PID, uptime, memory usage
        - [ ] Request count, error count
        - [ ] Last activity timestamp
    
    - [ ] `legale daemon logs <profile>` - show bot logs
        - [ ] Tail logs for specific bot
        - [ ] Filter by log level

#### 7.4 Nginx Configuration Generator
- [ ] **Multi-Bot Nginx Config**:
    - [ ] Generate nginx config for all bots
    - [ ] Each bot gets unique location: `/webhook/<profile>`
    - [ ] Proxy to correct port based on profile
    - [ ] SSL/TLS configuration per domain (optional)

- [ ] **Config Commands**:
    - [ ] `legale nginx generate` - generate nginx config
        - [ ] Create config for all active bots
        - [ ] Output to stdout or file
        - [ ] Include SSL setup instructions
    
    - [ ] `legale nginx install` - install config to nginx
        - [ ] Copy to /etc/nginx/sites-available/
        - [ ] Create symlink to sites-enabled/
        - [ ] Test nginx config
        - [ ] Reload nginx

#### 7.5 Webhook Management
- [ ] **Profile-Specific Webhooks**:
    - [ ] Update `register_webhook()` to use profile config
    - [ ] Auto-generate webhook URL: `https://domain.com/webhook/<profile>`
    - [ ] Store webhook URL in profile config
    - [ ] Support custom webhook URLs per profile

- [ ] **Webhook Commands**:
    - [ ] `legale webhook register <profile>` - register webhook for bot
        - [ ] Use profile's bot token
        - [ ] Use profile's webhook URL from config
        - [ ] Auto-generate URL if not configured
    
    - [ ] `legale webhook delete <profile>` - delete webhook
    
    - [ ] `legale webhook list` - list all webhooks
        - [ ] Show profile, bot, webhook URL, status
    
    - [ ] `legale webhook info <profile>` - show webhook details
        - [ ] Pending updates count
        - [ ] Last error time/message
        - [ ] Max connections

#### 7.6 Bot Isolation & Security
- [ ] **Process Isolation**:
    - [ ] Each bot runs in separate process
    - [ ] Separate memory space per bot
    - [ ] Independent crash recovery
    - [ ] Resource limits per bot (optional)

- [ ] **Data Isolation**:
    - [ ] Verify profile data isolation
    - [ ] No cross-profile data access
    - [ ] Separate log files per bot
    - [ ] Separate PID files per bot

- [ ] **Security**:
    - [ ] Validate bot tokens before storing
    - [ ] Encrypt tokens in config.env (optional)
    - [ ] Restrict file permissions on config files
    - [ ] Add webhook secret tokens (optional)

#### 7.7 Monitoring & Logging
- [ ] **Per-Bot Logging**:
    - [ ] Separate log file per bot: `profiles/<profile>/bot.log`
    - [ ] Structured logging with profile/bot context
    - [ ] Log rotation per bot
    - [ ] Centralized log aggregation (optional)

- [ ] **Metrics**:
    - [ ] Request count per bot
    - [ ] Response time per bot
    - [ ] Error rate per bot
    - [ ] Active users per bot
    - [ ] Token usage per bot

- [ ] **Health Checks**:
    - [ ] `/health/<profile>` endpoint per bot
    - [ ] Webhook connectivity check
    - [ ] Database connectivity check
    - [ ] LLM API connectivity check



- [ ] **Migration Tools**:
    - [ ] `legale migrate` - migrate old setup to multi-bot
        - [ ] Move global bot token to default profile
        - [ ] Create config.env for default profile
        - [ ] Update .env to remove bot token
        - [ ] Preserve existing data

#### 7.9 Documentation & Examples
- [ ] **Update Documentation**:
    - [ ] Update README.md with multi-bot examples
    - [ ] Add multi-bot workflow examples
    - [ ] Document config.env format
    - [ ] Add troubleshooting section for multi-bot

- [ ] **Example Workflows**:
    ```bash
    # Setup multiple bots
    legale profile create support-bot --set-active
    legale bot add support-bot --token <TOKEN1>
    legale ingest support_data.json
    
    legale profile create sales-bot
    legale bot add sales-bot --token <TOKEN2>
    legale ingest sales_data.json --profile sales-bot
    
    # Start all bots
    legale daemon start
    
    # Or start specific bot
    legale daemon start support-bot
    
    # Monitor
    legale daemon status
    legale daemon logs support-bot
    ```

#### 7.10 Testing
- [ ] **Multi-Bot Tests**:
    - [ ] Test multiple bots running simultaneously
    - [ ] Test port allocation and conflicts
    - [ ] Test profile isolation
    - [ ] Test graceful shutdown of all bots
    - [ ] Test bot restart without affecting others
    - [ ] Test webhook routing to correct bot
    - [ ] Load testing with multiple bots

- [ ] **Integration Tests**:
    - [ ] Test full workflow: add bot → ingest → start → webhook
    - [ ] Test migration from single-bot to multi-bot
    - [ ] Test config.env loading and merging
    - [ ] Test nginx config generation

#### Implementation Priority
1. **Phase 7.1** - Profile-specific config (foundation)
2. **Phase 7.2** - Bot management commands (user interface)
3. **Phase 7.5** - Webhook management (core functionality)
4. **Phase 7.3** - Multi-bot daemon (advanced)
5. **Phase 7.4** - Nginx config generator (deployment)
6. **Phase 7.6-7.9** - Security, monitoring, docs (polish)

#### Success Criteria
- [ ] Can run 3+ bots simultaneously without conflicts
- [ ] Each bot has isolated data and config
- [ ] Easy to add/remove bots via CLI
- [ ] Automatic webhook URL generation
- [ ] Graceful handling of bot failures
- [ ] Clear monitoring and logging per bot

### Phase 8: Admin Bot Interface (Telegram Admin Panel)
**Status: COMPLETED** ✅
**Goal**: Provide full orchestrator functionality through Telegram bot interface for administrators.

#### 8.1 Admin Command Architecture ✅
- [x] **Command Router**:
    - [x] Create `AdminCommandHandler` class
    - [x] Implement command parsing and routing
    - [x] Add permission checks (admin-only)
    - [x] Support interactive multi-step commands
    - [x] Add command state management (for confirmations)

- [x] **Command Structure**:
    ```
    /admin                          # Main admin menu
    /admin profile list             # List profiles
    /admin profile create <name>    # Create profile
    /admin profile switch <name>    # Switch profile
    /admin profile delete <name>    # Delete profile
    /admin profile info [name]      # Show profile info
    /admin ingest <file_id>         # Ingest from uploaded file
    /admin ingest clear             # Clear current profile data
    /admin stats                    # Show bot statistics
    /admin logs [lines]             # Show recent logs
    /admin restart                  # Restart bot
    ```

- [x] **Interactive Menus**:
    - [x] Use Telegram inline keyboards for navigation
    - [x] Implement callback query handlers
    - [x] Add confirmation dialogs for destructive actions

#### 8.2 Profile Management via Bot ✅
- [x] **Profile Commands**:
    - [x] `/admin profile list` - List all profiles
        - [x] Show profile name, status, DB size
        - [x] Indicate active profile
        - [x] Show chunk count per profile
        - [x] Add inline buttons for quick actions
    
    - [x] `/admin profile create <name>` - Create new profile
        - [x] Validate profile name
        - [x] Create profile structure
        - [x] Set as active (optional)
        - [x] Show confirmation with profile path
    
    - [x] `/admin profile switch <name>` - Switch active profile
        - [x] Validate profile exists
        - [x] Update active profile
        - [x] Reload bot with new profile
        - [x] Show confirmation
    
    - [x] `/admin profile delete <name>` - Delete profile
        - [x] Show profile info before deletion
        - [x] Require confirmation (inline button)
        - [x] Prevent deletion of active profile
        - [x] Show deletion summary
    
    - [x] `/admin profile info [name]` - Show profile details
        - [x] Database size and chunk count
        - [x] Vector store size
        - [x] Last activity timestamp
        - [x] Bot configuration (if configured)

#### 8.3 Data Ingestion via Bot ✅
- [x] **File Upload Handling**:
    - [x] Accept JSON file uploads
    - [x] Validate file format (JSON)
    - [x] Store uploaded file temporarily
    - [x] Process ingestion in background
    - [x] Show progress updates
    - [x] Clean up temp files after processing

- [x] **Ingestion Commands**:
    - [x] `/admin ingest` - Start ingestion wizard
        - [x] Prompt to upload JSON file
        - [x] Show file info (size, message count)
        - [x] Ask for confirmation
        - [x] Option to clear existing data
        - [x] Show ingestion progress
    
    - [x] `/admin ingest clear` - Clear current profile data
        - [x] Show current data stats
        - [x] Require confirmation
        - [x] Clear SQL and vector DB
        - [x] Show cleared data summary
    
    - [x] `/admin ingest status` - Show ingestion status
        - [x] Current/last ingestion progress
        - [x] Messages processed
        - [x] Chunks created
        - [x] Errors (if any)

#### 8.4 Bot Statistics & Monitoring ✅
- [x] **Statistics Commands**:
    - [x] `/admin stats` - Show comprehensive stats
        - [x] Active profile info
        - [x] Total chunks in DB
        - [x] Vector store size
        - [x] Request count (today/week/total)
        - [x] Average response time
        - [x] Error rate
        - [x] Token usage statistics
        - [x] Most active users
    
    - [x] `/admin stats profile <name>` - Profile-specific stats
        - [x] Database metrics
        - [x] Usage patterns
        - [x] Last activity
    
    - [x] `/admin stats users` - User statistics
        - [x] Active users count
        - [x] Top users by request count
        - [x] User activity timeline

- [x] **Monitoring**:
    - [x] `/admin health` - System health check
        - [x] Database connectivity
        - [x] Vector store status
        - [x] LLM API status
        - [x] Memory usage
        - [x] Disk space
    
    - [x] `/admin logs [lines]` - Show recent logs
        - [x] Tail bot logs (default: 50 lines)
        - [x] Filter by log level
        - [x] Search in logs
        - [x] Download logs as file

#### 8.5 Bot Control & Management ✅
- [x] **Control Commands**:
    - [x] `/admin restart` - Restart bot
        - [x] Graceful shutdown
        - [x] Reload configuration
        - [x] Restart process
        - [ ] Allow admin access only
        - [ ] Log maintenance periods

#### 8.6 Advanced Configuration ✅
- [x] **Config Management**:
    - [x] `config.json` per profile
    - [x] Store admin password in config (migrated from env)

- [x] **Chat Whitelist**:
    - [x] `/admin chat list` - Show allowed chats
    - [x] `/admin chat add <id>` - Allow chat
    - [x] `/admin chat remove <id>` - Disallow chat
    - [x] `/admin allowed list/add` - Aliases added
    - [x] DM Logic: Admins always allowed; Non-admins only commands allowed (unless whitelisted)
    - [x] `/id` command to show chat ID

- [x] **Response Frequency**:
    - [x] `/admin frequency <n>` - Set response frequency
        - [x] 1 = Respond to every message
        - [x] n > 1 = Respond to every Nth message
    - [x] Silent processing (add to context without response)

### Phase 9: Testing & Stabilization

#### 9.1 Test Suite Improvements ✅
- [x] Fix existing test failures
- [x] Improve mocking for admin commands
- [x] Add tests for new admin features (profiles, config, ingestion)
- [x] Ensure all tests pass

#### 9.2 Code Coverage Increase 🔄
- [x] Cover admin router (85%)
- [x] Cover admin tasks (88%)
- [x] Cover CLI (62%)
- [ ] Increase coverage for `tgbot.py` (>50%)
- [ ] Increase coverage for `ingestion/telegram.py` (>50%)
- [ ] Overall coverage > 60%
    
    - [ ] `/admin config set <key> <value>` - Update config
        - [ ] Update environment variables
        - [ ] Validate values
        - [ ] Require confirmation
        - [ ] Apply changes

#### 8.6 User Management
- [ ] **User Commands**:
    - [ ] `/admin users list` - List all users
        - [ ] User ID, name, username
        - [ ] Last activity
        - [ ] Request count
        - [ ] Pagination support
    
    - [ ] `/admin users info <user_id>` - User details
        - [ ] Full user information
        - [ ] Activity history
        - [ ] Request statistics
        - [ ] Recent queries
    
    - [ ] `/admin users block <user_id>` - Block user
        - [ ] Add to blocklist
        - [ ] Prevent bot access
        - [ ] Log blocking action
    
    - [ ] `/admin users unblock <user_id>` - Unblock user
        - [ ] Remove from blocklist
        - [ ] Restore access
        - [ ] Log unblocking action

#### 8.7 Database Operations
- [ ] **Database Commands**:
    - [ ] `/admin db info` - Database information
        - [ ] Database size
        - [ ] Table statistics
        - [ ] Index information
        - [ ] Chunk count
    
    - [ ] `/admin db vacuum` - Optimize database
        - [ ] Run VACUUM on SQLite
        - [ ] Reclaim disk space
        - [ ] Show space saved
    
    - [ ] `/admin db backup` - Create backup
        - [ ] Backup current profile DB
        - [ ] Compress backup
        - [ ] Upload backup file to admin
        - [ ] Store backup metadata
    
    - [ ] `/admin db restore <file_id>` - Restore from backup
        - [ ] Download backup file
        - [ ] Validate backup
        - [ ] Require confirmation
        - [ ] Restore database
        - [ ] Rebuild vector store

#### 8.8 Model Management
- [ ] **Model Commands**:
    - [ ] `/admin models list` - List available models
        - [ ] Show models from models.txt
        - [ ] Indicate current model
        - [ ] Show model capabilities
    
    - [ ] `/admin models add <model_name>` - Add model
        - [ ] Append to models.txt
        - [ ] Validate model name
        - [ ] Test model availability
    
    - [ ] `/admin models remove <model_name>` - Remove model
        - [ ] Remove from models.txt
        - [ ] Prevent removal of current model
        - [ ] Require confirmation
    
    - [ ] `/admin models set <model_name>` - Set default model
        - [ ] Update current model
        - [ ] Reinitialize LLM client
        - [ ] Show confirmation

#### 8.9 Webhook Management (Multi-Bot)
- [ ] **Webhook Commands**:
    - [ ] `/admin webhook status` - Show webhook status
        - [ ] Current webhook URL
        - [ ] Pending updates count
        - [ ] Last error (if any)
        - [ ] Max connections
    
    - [ ] `/admin webhook register <url>` - Register webhook
        - [ ] Set webhook URL
        - [ ] Validate URL format
        - [ ] Register with Telegram
        - [ ] Show confirmation
    
    - [ ] `/admin webhook delete` - Delete webhook
        - [ ] Remove webhook from Telegram
        - [ ] Show confirmation
    
    - [ ] `/admin webhook test` - Test webhook
        - [ ] Send test update
        - [ ] Verify reception
        - [ ] Show latency

#### 8.10 Notifications & Alerts
- [ ] **Notification System**:
    - [ ] Error notifications to admin
        - [ ] Critical errors
        - [ ] API failures
        - [ ] Database errors
    
    - [ ] Activity notifications
        - [ ] New user alerts
        - [ ] High usage alerts
        - [ ] Unusual activity patterns
    
    - [ ] System notifications
        - [ ] Low disk space
        - [ ] High memory usage
        - [ ] Token limit warnings

- [ ] **Alert Configuration**:
    - [ ] `/admin alerts on` - Enable alerts
    - [ ] `/admin alerts off` - Disable alerts
    - [ ] `/admin alerts config` - Configure alert thresholds

#### 8.11 Interactive Wizards
- [ ] **Setup Wizard**:
    - [ ] `/admin setup` - Initial setup wizard
        - [ ] Create first profile
        - [ ] Upload initial data
        - [ ] Configure bot settings
        - [ ] Test bot functionality
        - [ ] Step-by-step guidance

- [ ] **Ingestion Wizard**:
    - [ ] Interactive data ingestion
        - [ ] Choose profile
        - [ ] Upload file
        - [ ] Preview data
        - [ ] Confirm ingestion
        - [ ] Monitor progress

#### 8.12 Help & Documentation
- [ ] **Help System**:
    - [ ] `/admin help` - Show admin help
        - [ ] List all admin commands
        - [ ] Command categories
        - [ ] Quick examples
    
    - [ ] `/admin help <command>` - Command-specific help
        - [ ] Detailed usage
        - [ ] Parameters
        - [ ] Examples
        - [ ] Related commands

#### 8.13 Security & Audit
- [ ] **Security Features**:
    - [ ] Command logging
        - [ ] Log all admin commands
        - [ ] Store in audit log
        - [ ] Include timestamp and user
    
    - [ ] Session management
        - [ ] Admin session timeout
        - [ ] Re-authentication for sensitive ops
        - [ ] Session activity tracking
    
    - [ ] Rate limiting
        - [ ] Limit admin command frequency
        - [ ] Prevent command flooding
        - [ ] Configurable limits

- [ ] **Audit Commands**:
    - [ ] `/admin audit log` - Show audit log
        - [ ] Recent admin actions
        - [ ] Filter by action type
        - [ ] Export audit log
    
    - [ ] `/admin audit export` - Export audit data
        - [ ] Generate audit report
        - [ ] Send as file to admin

#### 8.14 Implementation Details
- [ ] **Technical Requirements**:
    - [ ] Async command handlers
    - [ ] State management for multi-step commands
    - [ ] Error handling and user feedback
    - [ ] Progress indicators for long operations
    - [ ] Inline keyboard navigation
    - [ ] Callback query handling
    - [ ] File upload/download handling
    - [ ] Background task processing

- [ ] **Code Structure**:
    ```python
    src/bot/
    ├── admin.py              # Existing admin auth
    ├── admin_commands.py     # NEW: Admin command handlers
    ├── admin_router.py       # NEW: Command routing
    ├── admin_keyboards.py    # NEW: Inline keyboards
    ├── admin_state.py        # NEW: State management
    └── admin_tasks.py        # NEW: Background tasks
    ```

#### 8.15 Testing
- [ ] **Admin Interface Tests**:
    - [ ] Test all admin commands
    - [ ] Test permission checks
    - [ ] Test interactive flows
    - [ ] Test error handling
    - [ ] Test state management
    - [ ] Test file uploads
    - [ ] Test background tasks

- [ ] **Integration Tests**:
    - [ ] Test full admin workflows
    - [ ] Test profile switching
    - [ ] Test data ingestion
    - [ ] Test bot restart
    - [ ] Test multi-step wizards

#### Implementation Priority
1. **Phase 8.1-8.2** - Basic admin commands & profile management
2. **Phase 8.3** - Data ingestion via bot
3. **Phase 8.4** - Statistics & monitoring
4. **Phase 8.5** - Bot control
5. **Phase 8.6-8.8** - User, DB, model management
6. **Phase 8.9-8.13** - Advanced features

#### Success Criteria
- [ ] All orchestrator functions accessible via Telegram
- [ ] Intuitive inline keyboard navigation
- [ ] Secure admin-only access
- [ ] Comprehensive error handling
- [ ] Real-time progress updates
- [ ] Full audit trail of admin actions
- [ ] No need to use CLI for common operations
