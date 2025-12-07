# Token Management Configuration

## Problem
The bot was hitting the OpenRouter/OpenAI token limit (16,385 tokens for gpt-3.5-turbo) because:
- Input prompt (context + history + task): ~16,336 tokens
- Output tokens requested: 1,000 tokens
- **Total: 17,336 tokens > 16,385 limit**

## Solutions Implemented

### 1. Reduced Output Token Limit
**File**: `src/core/llm.py`
- Changed `max_tokens` from 1000 to **500** in `LLMClient.complete()`
- This gives more room for input context

### 2. Reduced Default Context Chunks
**File**: `src/bot/core.py`
- Changed default `n_results` from 5 to **3** in `LegaleBot.chat()`
- Retrieves fewer context chunks by default

### 3. Added Context Size Limiting
**File**: `src/core/prompt.py`
- Added `max_context_chars` parameter (default: 8000 characters)
- Automatically truncates context chunks if they exceed the limit
- Prevents token overflow while preserving most relevant information

### 4. Token Counting and Monitoring
**File**: `src/core/llm.py`
- Added `tiktoken` library for accurate token counting
- New method `count_tokens()` to track token usage

### 5. Automatic Context Reset
**File**: `src/bot/core.py`
- Automatically resets chat history when token limit is reached
- Configurable via `MAX_CONTEXT_TOKENS` environment variable (default: 14000)
- Shows warning message when auto-reset occurs

### 6. User Commands
**File**: `src/bot/tgbot.py`
- **`/reset`** - Manually reset chat context
- **`/tokens`** - View current token usage and percentage

## Configuration Options

### Environment Variables

Add to your `.env` file:

```bash
# Maximum context tokens before auto-reset (default: 14000)
MAX_CONTEXT_TOKENS=14000

# Model selection affects token limits
OPENROUTER_MODEL=openai/gpt-oss-20b:free
```

### Using Telegram Commands

#### `/tokens` - Check Token Usage
Shows current token usage:
```
📊 Использование токенов:

Текущее: 3,245
Максимум: 14,000
Использовано: 23.18%

✅ Достаточно места для разговора.
```

#### `/reset` - Reset Context
Manually clear chat history:
```
✅ Контекст сброшен!
```

### Option 1: Use a Larger Context Model (Recommended)
Switch to a model with a larger context window by setting the environment variable:

```bash
# In your .env file
OPENROUTER_MODEL=openai/gpt-4-turbo  # 128k tokens
MAX_CONTEXT_TOKENS=120000

# or
OPENROUTER_MODEL=anthropic/claude-3-sonnet  # 200k tokens
MAX_CONTEXT_TOKENS=190000
```

### Option 2: Adjust Context Limits
You can tune these parameters when calling the bot:

```python
# Retrieve fewer chunks
response = bot.chat(user_input, n_results=2)

# Or modify max_context_chars in construct_prompt call
system_prompt = self.prompt_engine.construct_prompt(
    context_chunks=context_chunks,
    chat_history=history_for_prompt,
    user_task=user_input,
    max_context_chars=5000  # Reduce from default 8000
)
```

### Option 3: Adjust Output Tokens
If you need longer responses, you can pass `max_tokens` explicitly:

```python
response = self.llm_client.complete(messages, max_tokens=300)  # Even shorter
```

## Monitoring Token Usage

To see token usage in real-time, run the bot with verbose logging:

```bash
poetry run python src/bot/tgbot.py run -vvv
```

## Recommended Settings by Model

| Model | Context Window | MAX_CONTEXT_TOKENS | n_results | max_context_chars | max_tokens |
|-------|---------------|-------------------|-----------|-------------------|------------|
| gpt-3.5-turbo | 16k | 14000 | 3 | 8000 | 500 |
| gpt-4-turbo | 128k | 120000 | 5-10 | 20000 | 1000 |
| claude-3-sonnet | 200k | 190000 | 5-10 | 30000 | 1000 |

## Quick Fix for Current Error

The changes already made should fix your current error. Just restart the bot:

```bash
# Stop the current bot (Ctrl+C)
# Restart it
poetry run python src/bot/tgbot.py run -vv
```

The bot will now:
- Use only 3 context chunks instead of 5
- Limit context to 8000 characters
- Request only 500 output tokens instead of 1000
- Automatically reset when reaching token limit
- Allow manual reset via `/reset` command
- Show token usage via `/tokens` command
- **Total should be well under 16,385 tokens**
