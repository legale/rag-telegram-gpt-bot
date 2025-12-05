# Legale Bot - Union Lawyer Chatbot

A RAG-based chatbot designed to act as an IT union lawyer. It ingests Telegram chat history, creates vector embeddings, and uses an LLM to answer questions based on the chat context.

## Features

- **Data Ingestion**: Parses Telegram JSON dumps or fetches directly from Telegram.
- **RAG Pipeline**: Chunks text, creates embeddings (ChromaDB), and retrieves relevant context.
- **Bot Interface**: CLI for interacting with the bot using OpenRouter/OpenAI models.
- **Privacy**: Sensitive data (API keys, sessions) is kept local.

## Setup

1.  **Install Dependencies**:
    ```bash
    poetry install
    ```

2.  **Configuration**:
    Copy `.env.example` to `.env` and fill in your credentials:
    ```bash
    cp .env.example .env
    ```
    *   `TELEGRAM_API_ID` / `TELEGRAM_API_HASH`: From my.telegram.org
    *   `OPENROUTER_API_KEY`: Your LLM API key.

3.  **Models**:
    Create a `models.txt` file with your preferred model name (e.g., `openai/gpt-3.5-turbo` or `anthropic/claude-3-opus`).

## Quick Start

### Complete Setup: From Zero to Telegram Bot

This guide walks you through the entire process of setting up the bot, creating the database from Telegram chat history, and deploying it as a Telegram bot.

#### Step 1: Initial Setup

1. **Clone and Install**:
   ```bash
   git clone git@github.com:legale/rag-telegram-gpt-bot.git
   cd rag-telegram-gpt-bot
   poetry install
   ```

2. **Configure Credentials**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add:
   - `TELEGRAM_API_ID` and `TELEGRAM_API_HASH` from [my.telegram.org](https://my.telegram.org)
   - `OPENROUTER_API_KEY` from [OpenRouter](https://openrouter.ai)
   - `TELEGRAM_BOT_TOKEN` from [@BotFather](https://t.me/BotFather)

3. **Configure Model**:
   ```bash
   echo "openai/gpt-3.5-turbo" > models.txt
   ```

#### Step 2: Create Database from Telegram Chat

1. **Dump Chat History**:
   ```bash
   # List available chats
   poetry run python src/ingestion/telegram.py list_chan
   
   # Dump messages from your chat (replace with actual chat name)
   poetry run python src/ingestion/telegram.py dump_chan "Your Chat Name" \
     --limit 10000 \
     --output telegram_dump.json
   ```

2. **Ingest Data into Database**:
   ```bash
   # Create database and vector embeddings
   poetry run python src/ingestion/pipeline.py telegram_dump.json --clear
   
   # This will:
   # - Parse 10,000 messages
   # - Create chunks with 20% overlap
   # - Generate vector embeddings (ChromaDB)
   # - Store in SQLite database
   ```

3. **Verify Database**:
   ```bash
   # Test with CLI
   poetry run ./src/bot/cli.py -vv
   
   # Ask a question to verify retrieval works
   # You: Что обсуждали в чате?
   ```

#### Step 3: Deploy Telegram Bot

1. **Create Bot with BotFather**:
   - Open Telegram and talk to [@BotFather](https://t.me/BotFather)
   - Send `/newbot` and follow instructions
   - Save the bot token to `.env` as `TELEGRAM_BOT_TOKEN`

2. **Setup Server (requires domain with SSL)**:
   ```bash
   # Install nginx if not already installed
   sudo apt install nginx certbot python3-certbot-nginx
   
   # Get SSL certificate
   sudo certbot --nginx -d yourdomain.com
   
   # Copy nginx config
   sudo cp nginx/telegram-bot.conf /etc/nginx/sites-available/legale-bot
   
   # Edit config and change:
   # - server_name to your domain
   # - SSL certificate paths (certbot creates them automatically)
   sudo nano /etc/nginx/sites-available/legale-bot
   
   # Enable site
   sudo ln -s /etc/nginx/sites-available/legale-bot /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

3. **Register Webhook**:
   ```bash
   poetry run python src/bot/tgbot.py register \
     --url https://yourdomain.com/webhook \
     --token YOUR_BOT_TOKEN
   ```

4. **Start Bot Service**:
   ```bash
   # Option A: Foreground (for testing)
   poetry run python src/bot/tgbot.py run -vv
   
   # Option B: Systemd service (production)
   sudo cp systemd/legale-bot.service /etc/systemd/system/
   # Edit service file and update paths/user
   sudo nano /etc/systemd/system/legale-bot.service
   sudo systemctl daemon-reload
   sudo systemctl start legale-bot
   sudo systemctl enable legale-bot
   ```

5. **Add Bot to Chat**:
   - Add your bot to the Telegram chat
   - Send a message: "Привет!"
   - Bot should respond with context from chat history

#### Step 4: Verify Everything Works

```bash
# Check bot service status
sudo systemctl status legale-bot

# View logs
sudo journalctl -u legale-bot -f

# Test webhook health
curl https://yourdomain.com/health

# Test in Telegram
# Send: "Что обсуждали в чате на прошлой неделе?"
```

---

## Usage

### 1. Dump Chat History
To dump messages from a Telegram chat:
```bash
poetry run python src/ingestion/telegram.py dump_chan "Chat Name" --limit 10000 --output telegram_dump.json
```

### 2. Ingest Data
Process the dump, create chunks, and store embeddings:
```bash
# First run or to append data
poetry run python src/ingestion/pipeline.py telegram_dump.json

# To clear existing database and re-ingest
poetry run python src/ingestion/pipeline.py telegram_dump.json --clear
```

### 3. Run the Bot
Start the interactive CLI:
```bash
poetry run ./src/bot/cli.py

# Verbose modes for debugging:
poetry run ./src/bot/cli.py -v   # Basic info
poetry run ./src/bot/cli.py -vv  # Retrieval details
poetry run ./src/bot/cli.py -vvv # Full LLM logs

# Adjust number of context chunks:
poetry run ./src/bot/cli.py --chunks 10
```

### 4. Telegram Bot (Webhook)

#### Prerequisites
- Domain with SSL certificate (use Let's Encrypt/certbot)
- Nginx installed
- Bot token from [@BotFather](https://t.me/BotFather)

#### Setup

1. **Create Bot**:
   ```bash
   # Talk to @BotFather on Telegram
   # Use /newbot command
   # Save the token
   ```

2. **Configure Environment**:
   Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   ```

3. **Configure Nginx**:
   ```bash
   # Copy config template
   sudo cp nginx/telegram-bot.conf /etc/nginx/sites-available/legale-bot
   
   # Edit the file and change:
   # - server_name to your domain
   # - SSL certificate paths
   
   # Enable site
   sudo ln -s /etc/nginx/sites-available/legale-bot /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

4. **Register Webhook**:
   ```bash
   poetry run python src/bot/tgbot.py register \
     --url https://yourdomain.com/webhook \
     --token YOUR_BOT_TOKEN
   ```

#### Running

**Foreground (for testing)**:
```bash
# Basic
poetry run python src/bot/tgbot.py run

# With verbosity
poetry run python src/bot/tgbot.py run -vv

# Custom port
poetry run python src/bot/tgbot.py run --port 8080
```

**As Systemd Service (production)**:
```bash
# Install service
sudo cp systemd/legale-bot.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start service
sudo systemctl start legale-bot
sudo systemctl enable legale-bot

# Check status
sudo systemctl status legale-bot

# View logs
sudo journalctl -u legale-bot -f
```

#### Management

```bash
# Delete webhook
poetry run python src/bot/tgbot.py delete

# Check health
curl https://yourdomain.com/health
```

