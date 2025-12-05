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
