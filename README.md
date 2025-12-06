# Legale Bot - Union Lawyer Chatbot

A RAG-based chatbot designed to act as an IT union lawyer. It ingests Telegram chat history, creates vector embeddings, and uses an LLM to answer questions based on the chat context.

## Features

- **Unified CLI**: Single command interface for all operations
- **Profile Management**: Multiple bot instances with separate databases
- **Data Ingestion**: Fetch from Telegram or import JSON dumps
- **RAG Pipeline**: ChromaDB vector store + context retrieval
- **Telegram Bot**: Webhook-based bot with `/model`, `/reset`, `/tokens` commands
- **Interactive Chat**: CLI interface for testing
- **Privacy**: All data stored locally

## Quick Start

### 1. Installation

```bash
git clone git@github.com:legale/rag-telegram-gpt-bot.git
cd rag-telegram-gpt-bot
poetry install
```

### 2. Configuration

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required keys:
- `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` from [my.telegram.org](https://my.telegram.org)
- `OPENROUTER_API_KEY` from [OpenRouter](https://openrouter.ai)
- `TELEGRAM_BOT_TOKEN` from [@BotFather](https://t.me/BotFather)
- `VOYAGE_API_KEY` for embeddings

### 3. Create Your First Bot

```bash
# Create a profile
poetry run python legale.py profile create mybot --set-active

# Fetch Telegram data
poetry run python legale.py telegram dump "My Chat" --limit 10000

# Ingest into database
poetry run python legale.py ingest telegram_dump_My_Chat.json

# Test with interactive chat
poetry run python legale.py chat -vv
```

## CLI Reference

The `legale.py` script provides a unified interface for all bot operations.

### Profile Management

Profiles allow you to manage multiple bot instances with separate databases.

```bash
# Create a new profile
legale profile create <name>
legale profile create <name> --set-active

# List all profiles
legale profile list

# Switch active profile
legale profile switch <name>

# Show profile info
legale profile info [name]

# Delete a profile
legale profile delete <name>
```

**Profile Structure:**
```
profiles/
└── mybot/
    ├── legale_bot.db              # SQLite database
    ├── chroma_db/                 # Vector embeddings
    └── telegram_session.session   # Telegram session
```

### Telegram Data Fetching

```bash
# List available chats
legale telegram list

# List chat members
legale telegram members "Chat Name"

# Dump chat messages
legale telegram dump "Chat Name" --limit 10000
legale telegram dump "Chat Name" --limit 10000 --output custom.json
legale telegram dump "Chat Name" --profile mybot
```

### Data Ingestion

```bash
# Ingest data into current profile
legale ingest <file.json>

# Clear existing data and re-ingest
legale ingest <file.json> --clear

# Ingest into specific profile
legale ingest <file.json> --profile mybot
```

### Interactive Chat

```bash
# Start chat session
legale chat

# With verbosity levels
legale chat -v      # Basic info
legale chat -vv     # Retrieval details
legale chat -vvv    # Full LLM logs

# Custom context chunks
legale chat --chunks 10

# Use specific profile
legale chat --profile mybot
```

### Telegram Bot

```bash
# Register webhook
legale bot register --url https://yourdomain.com/webhook

# Delete webhook
legale bot delete

# Run in foreground (testing)
legale bot run
legale bot run -vv --port 8080

# Run as daemon (production)
legale bot daemon
```

**Bot Commands:**
- `/start` - Welcome message
- `/help` - Show available commands
- `/reset` - Clear conversation context
- `/tokens` - Show token usage statistics
- `/model` - Switch to next LLM model (cycles through `models.txt`)
- `/admin_set <password>` - Set yourself as bot administrator
- `/admin_get` - Show admin information (admin only)

## Common Workflows

### Setup Development and Production Environments

```bash
# Development
legale profile create dev --set-active
legale telegram dump "Dev Chat" --limit 1000
legale ingest telegram_dump_Dev_Chat.json
legale chat -vv

# Production
legale profile create prod
legale telegram dump "Prod Chat" --limit 10000 --profile prod
legale ingest telegram_dump_Prod_Chat.json --profile prod

# Switch between them
legale profile switch dev
legale profile switch prod
```

### Update Bot Data

```bash
# Fetch new messages
legale telegram dump "Chat" --limit 10000

# Clear old data and re-ingest
legale ingest telegram_dump_Chat.json --clear
```

### Deploy Telegram Bot

```bash
# 1. Setup nginx with SSL
sudo apt install nginx certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com

# 2. Copy nginx config
sudo cp nginx/telegram-bot.conf /etc/nginx/sites-available/legale-bot
# Edit and enable
sudo ln -s /etc/nginx/sites-available/legale-bot /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 3. Register webhook
legale bot register --url https://yourdomain.com/webhook

# 4. Run bot
legale bot run -vv  # Test first
legale bot daemon   # Then run as daemon
```

## Model Management

The bot supports switching between multiple LLM models at runtime.

### Configure Models

Edit `models.txt`:
```
openai/gpt-oss-20b:free
nvidia/nemotron-nano-9b-v2:free
cognitivecomputations/dolphin-mistral-24b-venice-edition:free
google/gemma-3-27b-it:free
```

### Switch Models

In Telegram bot, use `/model` command to cycle through available models.

In CLI, models are configured via environment variables in `.env`:
```bash
OPENROUTER_MODEL=openai/gpt-3.5-turbo
```

## Configuration

### Environment Variables (`.env`)

```bash
# Profile (managed automatically)
ACTIVE_PROFILE=default

# LLM API
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-3.5-turbo

# Token Management
MAX_CONTEXT_TOKENS=14000

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_token_here

# Telegram API (for data fetching)
TELEGRAM_API_ID=your_id_here
TELEGRAM_API_HASH=your_hash_here

# Embeddings
VOYAGE_API_KEY=your_key_here

# Bot Administration
ADMIN_PASSWORD=your_secure_admin_password_here
```

### Bot Administration

The bot supports a simple admin system for managing access.

**Setup:**
1. Set `ADMIN_PASSWORD` in your `.env` file
2. In Telegram, send `/admin_set <your_password>` to become admin
3. Use `/admin_get` to view current admin info

**Admin File:**
- Stored in `profiles/<profile>/admin.json`
- Contains admin user ID, username, and name
- File permissions set to 600 (owner read/write only)

**Security Notes:**
- Only one admin per profile
- Password is checked against `ADMIN_PASSWORD` in `.env`
- Failed attempts are logged
- Admin info is profile-specific

### Token Limits

Adjust `MAX_CONTEXT_TOKENS` based on your model:
- GPT-3.5-turbo (16k total): `14000`
- GPT-4-turbo (128k total): `120000`
- Claude-3-sonnet (200k total): `190000`

## Project Structure

```
legale-bot/
├── legale.py              # Unified CLI orchestrator
├── models.txt             # Available LLM models
├── profiles/              # Profile data (gitignored)
│   └── default/
│       ├── legale_bot.db
│       ├── chroma_db/
│       └── telegram_session.session
├── src/
│   ├── bot/
│   │   ├── cli.py        # Interactive chat
│   │   ├── core.py       # Bot logic
│   │   └── tgbot.py      # Telegram webhook
│   ├── core/
│   │   ├── embedding.py  # Embeddings
│   │   ├── llm.py        # LLM client
│   │   ├── prompt.py     # Prompt engineering
│   │   └── retrieval.py  # RAG retrieval
│   ├── ingestion/
│   │   ├── chunker.py    # Text chunking
│   │   ├── parser.py     # JSON parsing
│   │   ├── pipeline.py   # Ingestion pipeline
│   │   └── telegram.py   # Telegram fetcher
│   └── storage/
│       ├── db.py         # SQLite database
│       └── vector_store.py # ChromaDB
└── tests/                # Test suite (87 tests, 41% coverage)
```

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# With coverage
poetry run pytest --cov=src --cov-report=term

# Specific test file
poetry run pytest tests/test_bot_core_models.py -v
```

### Test Coverage

Current coverage: **41%** (87 tests)
- `src/bot/core.py`: **98%** ✅
- `src/core/llm.py`: **100%** ✅
- `src/core/embedding.py`: **100%** ✅

See `map.md` for detailed test coverage improvement plan.

## Troubleshooting

### Check Profile Status

```bash
legale profile info
ls -lh profiles/default/
```

### Database Issues

```bash
# Clear and re-ingest
legale ingest data.json --clear

# Check database
sqlite3 profiles/default/legale_bot.db "SELECT COUNT(*) FROM chunks;"
```

### Telegram Session Issues

```bash
# Delete session and re-authenticate
rm profiles/default/telegram_session.session
legale telegram list  # Will prompt for login
```

### Bot Not Responding

```bash
# Check webhook status
curl https://api.telegram.org/bot<TOKEN>/getWebhookInfo

# Re-register webhook
legale bot delete
legale bot register --url https://yourdomain.com/webhook

# Check logs
legale bot run -vvv
```

## License

MIT

## Contributing

See `map.md` for project roadmap and development guidelines.
