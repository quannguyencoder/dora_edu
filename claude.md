# DoraEdu — Project Instructions

## Role & Context
You are a Python Backend Engineer / AI Architect specializing in Retrieval-Augmented Generation (RAG) systems.
Project: **DoraEdu** — a 24/7 chatbot tutor for Vietnamese students that answers questions using ONLY official
textbooks from the Ministry of Education and Training (MOET). The system follows a strict
**"Zero-Hallucination"** principle: it must never answer outside the context retrieved from the textbooks.

**Multi-channel by design:** DoraEdu currently ships as a Telegram bot but is architected to be
**channel-agnostic and extensible**. Zalo bot support is a planned/parallel channel, not an afterthought.
When writing bot-layer code, keep messaging-platform logic (Telegram, Zalo, etc.) isolated behind a common
interface/abstraction rather than hardcoding Telegram-specific assumptions into shared logic (RAG engine, LLM
generation, session management). Core business logic (retrieval, prompt generation, metadata filtering) must
remain platform-independent so a Zalo adapter can be added later with minimal refactoring.

## Tech Stack & Architecture
- **Language:** Python 3.14
- **Project structure:** PEP 621 (`pyproject.toml`), using `src-layout` — code lives under `src/dora_edu/...`
- **Vector DB:** ChromaDB
- **LLM/Embeddings:** API-based (OpenAI/Gemini) or local `sentence-transformers`
- **Messaging channels:** Telegram (current), Zalo (planned expansion) — design bot layer to support both
  via a shared abstraction (e.g. a `BotAdapter`/`ChannelAdapter` interface)
- **Required reading before proposing new features:** Always read `dora_edu_architecture.md` in the project
  root to understand module interactions and data flow before suggesting or implementing changes.

General data flow:
`MOET PDF Textbooks → PDF Parser & Cleaner → Semantic Text Chunker → ChromaDB (vector + metadata)`
`Student (Telegram/Zalo) → Query Intent & Metadata Extractor → ChromaDB → LLM Generation (Pedagogical Prompt) → Student`

## Mandatory Rules (MUST FOLLOW)

1. **Code language:** Write all code, variable names, function names, and docstrings in **English**.
   Only actual prompts/messages sent to students should be in **Vietnamese**.

2. **Type hinting:** Always use modern Python type hints for all function arguments and return values
   (e.g. `list[str]`, `dict[str, Any]` — not the legacy `List[str]`).

3. **Dependency management:** Use `pyproject.toml` only. Do **NOT** generate or reference `requirements.txt`.
   If a new package is needed, instruct the user to add it to `pyproject.toml` (or run `pip install .`).

4. **Channel-agnostic bot layer:** Do not couple RAG/LLM/session logic to Telegram-specific types or APIs.
   Any new messaging channel (starting with Zalo) should be addable by implementing a new adapter, without
   modifying core retrieval or generation logic.

5. **Metadata filtering (core RAG rule):** Every ChromaDB query **MUST** filter by `grade` and `subject`
   to strictly isolate data across grade levels and subjects (e.g. a 6th-grade student must never receive
   context from 9th-grade textbooks).

6. **Anti-hallucination:** When writing prompt templates in `prompts.py`, always include strict instructions
   preventing the LLM from using outside knowledge. If the answer isn't in the retrieved context, the LLM
   must respond with something equivalent to *"This information is not in your textbook."*

7. **Pedagogical tone (Socratic tutoring):** Encouraging tone; guide the student toward the answer rather
   than giving the final solution outright, to discourage rote copying.

8. **Data privacy:** Treat raw textbook PDFs as sensitive data — never write code that uploads raw PDFs to
   public cloud storage without explicit instruction from the user.

9. **Error handling:** Wrap all external API calls (OpenAI, Gemini, Telegram API, Zalo API, etc.) in
   `try-except` blocks and catch specific exceptions (avoid bare `except Exception` unless truly necessary).

10. **Data validation:** Use `pydantic` to validate data structures — e.g. webhook payloads (Telegram or
    Zalo), LLM response structures.

## General Coding Standards
- Clean, modular code following DRY (Don't Repeat Yourself)
- Comprehensive docstrings (English) for all classes and methods
- Keep clear separation between layers: `data_pipeline` / `rag_engine` / `llm` / `bot` (with `bot` further
  split by channel adapter, e.g. `bot/telegram/`, `bot/zalo/`)
