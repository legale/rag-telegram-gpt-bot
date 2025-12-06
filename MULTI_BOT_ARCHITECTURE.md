# Multi-Bot Architecture Design

## Overview

Support for running multiple Telegram bots simultaneously, each with isolated data and configuration.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Legale CLI (legale.py)                  │
│  Commands: bot add/remove/list, daemon start/stop/status   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    BotManager (NEW)                         │
│  - Manages multiple bot processes                           │
│  - Port allocation (8000, 8001, 8002...)                   │
│  - Process monitoring & health checks                       │
└────────────┬────────────────┬────────────────┬──────────────┘
             │                │                │
             ▼                ▼                ▼
    ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
    │  Bot Process 1 │ │  Bot Process 2 │ │  Bot Process N │
    │  Port: 8000    │ │  Port: 8001    │ │  Port: 800N    │
    │  Profile: bot1 │ │  Profile: bot2 │ │  Profile: botN │
    └────────┬───────┘ └────────┬───────┘ └────────┬───────┘
             │                  │                  │
             ▼                  ▼                  ▼
    ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
    │  Profile: bot1 │ │  Profile: bot2 │ │  Profile: botN │
    ├────────────────┤ ├────────────────┤ ├────────────────┤
    │ config.env     │ │ config.env     │ │ config.env     │
    │  - BOT_TOKEN   │ │  - BOT_TOKEN   │ │  - BOT_TOKEN   │
    │  - WEBHOOK_URL │ │  - WEBHOOK_URL │ │  - WEBHOOK_URL │
    │  - BOT_NAME    │ │  - BOT_NAME    │ │  - BOT_NAME    │
    ├────────────────┤ ├────────────────┤ ├────────────────┤
    │ legale_bot.db  │ │ legale_bot.db  │ │ legale_bot.db  │
    │ chroma_db/     │ │ chroma_db/     │ │ chroma_db/     │
    │ bot.log        │ │ bot.log        │ │ bot.log        │
    │ bot.pid        │ │ bot.pid        │ │ bot.pid        │
    └────────────────┘ └────────────────┘ └────────────────┘
```

## Nginx Routing

```
                    ┌──────────────────┐
                    │   Nginx (SSL)    │
                    │  yourdomain.com  │
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
/webhook/bot1       /webhook/bot2       /webhook/botN
         │                   │                   │
         ▼                   ▼                   ▼
  localhost:8000      localhost:8001      localhost:800N
```

## File Structure

```
legale-bot/
├── legale.py                    # CLI orchestrator
├── .env                         # Global config (shared API keys)
│   ├── OPENROUTER_API_KEY
│   ├── VOYAGE_API_KEY
│   ├── TELEGRAM_API_ID
│   └── TELEGRAM_API_HASH
│
└── profiles/
    ├── support-bot/
    │   ├── config.env           # Bot-specific config
    │   │   ├── TELEGRAM_BOT_TOKEN=<TOKEN1>
    │   │   ├── BOT_NAME="Support Bot"
    │   │   ├── WEBHOOK_URL=https://domain.com/webhook/support-bot
    │   │   └── PORT=8000
    │   ├── legale_bot.db
    │   ├── chroma_db/
    │   ├── bot.log
    │   ├── bot.pid
    │   └── telegram_session.session
    │
    ├── sales-bot/
    │   ├── config.env
    │   │   ├── TELEGRAM_BOT_TOKEN=<TOKEN2>
    │   │   ├── BOT_NAME="Sales Bot"
    │   │   ├── WEBHOOK_URL=https://domain.com/webhook/sales-bot
    │   │   └── PORT=8001
    │   ├── legale_bot.db
    │   ├── chroma_db/
    │   ├── bot.log
    │   ├── bot.pid
    │   └── telegram_session.session
    │
    └── hr-bot/
        ├── config.env
        │   ├── TELEGRAM_BOT_TOKEN=<TOKEN3>
        │   ├── BOT_NAME="HR Bot"
        │   ├── WEBHOOK_URL=https://domain.com/webhook/hr-bot
        │   └── PORT=8002
        ├── legale_bot.db
        ├── chroma_db/
        ├── bot.log
        ├── bot.pid
        └── telegram_session.session
```

## Workflow Examples

### Setup Multiple Bots

```bash
# 1. Create profiles and add bots
legale profile create support-bot --set-active
legale bot add support-bot --token <SUPPORT_TOKEN>
legale ingest support_chat.json

legale profile create sales-bot
legale bot add sales-bot --token <SALES_TOKEN>
legale ingest sales_chat.json --profile sales-bot

legale profile create hr-bot
legale bot add hr-bot --token <HR_TOKEN>
legale ingest hr_chat.json --profile hr-bot

# 2. List all bots
legale bot list
# Output:
# Profile       Bot Username    Status    Webhook
# support-bot   @SupportBot     active    https://domain.com/webhook/support-bot
# sales-bot     @SalesBot       active    https://domain.com/webhook/sales-bot
# hr-bot        @HRBot          active    https://domain.com/webhook/hr-bot

# 3. Start all bots
legale daemon start

# 4. Check status
legale daemon status
# Output:
# Profile       PID     Port    Uptime    Requests    Errors
# support-bot   1234    8000    2h 15m    1,234       0
# sales-bot     1235    8001    2h 15m    567         1
# hr-bot        1236    8002    2h 15m    890         0
```

### Manage Individual Bots

```bash
# Start specific bot
legale daemon start support-bot

# Stop specific bot
legale daemon stop sales-bot

# Restart specific bot
legale daemon restart hr-bot

# View logs
legale daemon logs support-bot --tail 100
```

### Webhook Management

```bash
# Register webhooks for all bots
legale webhook register support-bot
legale webhook register sales-bot
legale webhook register hr-bot

# Or auto-register when starting daemon
legale daemon start --register-webhooks

# List all webhooks
legale webhook list

# Check webhook status
legale webhook info support-bot
```

## Key Design Decisions

### 1. Config Hierarchy
- **Global `.env`**: Shared API keys (OpenRouter, Voyage, Telegram API)
- **Profile `config.env`**: Bot-specific settings (bot token, webhook URL)
- **Merge Strategy**: Profile config overrides global config

### 2. Process Model
- **One process per bot**: Isolation, independent crashes
- **BotManager**: Parent process that spawns/monitors bot processes
- **Health checks**: Each bot reports health to manager
- **Graceful shutdown**: SIGTERM → cleanup → exit

### 3. Port Allocation
- **Auto-assign**: 8000, 8001, 8002, ... (sequential)
- **Stored in config**: PORT field in profile config.env
- **Conflict detection**: Check if port already in use before starting

### 4. Webhook Routing
- **Pattern**: `/webhook/<profile_name>`
- **Nginx**: Routes to correct localhost port based on URL path
- **Auto-generation**: CLI generates nginx config for all bots

### 5. Data Isolation
- **Separate databases**: Each profile has own SQLite DB
- **Separate vector stores**: Each profile has own ChromaDB
- **Separate logs**: Each bot writes to own log file
- **No cross-profile access**: Enforced by file system permissions

## Migration Path

### From Single-Bot to Multi-Bot

```bash
# Before: Single bot with global TELEGRAM_BOT_TOKEN
.env:
  TELEGRAM_BOT_TOKEN=<TOKEN>

# After: Multi-bot with profile configs
.env:
  # No TELEGRAM_BOT_TOKEN here

profiles/default/config.env:
  TELEGRAM_BOT_TOKEN=<TOKEN>

# Migration command
legale migrate
# → Moves TELEGRAM_BOT_TOKEN from .env to profiles/default/config.env
# → Creates config.env for default profile
# → Updates .env to remove bot token
```

## Implementation Phases

### Phase 1: Foundation (7.1)
- Profile-specific config.env
- Config loading/merging logic
- Update ProfileManager

### Phase 2: Bot Management (7.2)
- `legale bot add/remove/list/info` commands
- Token validation
- Bot info retrieval from Telegram API

### Phase 3: Webhook Management (7.5)
- Profile-specific webhook registration
- Auto-generate webhook URLs
- Webhook status checking

### Phase 4: Multi-Process Daemon (7.3)
- BotManager class
- Process spawning/monitoring
- Port allocation
- Health checks

### Phase 5: Deployment Tools (7.4)
- Nginx config generator
- Config installation
- SSL setup automation

### Phase 6: Polish (7.6-7.9)
- Security hardening
- Monitoring & metrics
- Migration tools
- Documentation

## Success Metrics

- ✅ Can run 5+ bots simultaneously
- ✅ Zero downtime when restarting individual bots
- ✅ < 1 minute to add new bot
- ✅ Automatic recovery from bot crashes
- ✅ Clear separation of bot data and logs
- ✅ Easy migration from single-bot setup
