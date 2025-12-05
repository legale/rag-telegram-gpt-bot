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

