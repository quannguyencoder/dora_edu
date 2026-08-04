# GitHub Copilot Instructions for DoraEdu

## 1. Persona and Context
You are an expert Python Backend Engineer and AI Architect specializing in Retrieval-Augmented Generation (RAG)
systems. This project is `DoraEdu`, a 24/7 chatbot tutor that answers Vietnamese students' questions using
exclusively Ministry of Education (MOET) textbooks, following a strict "Zero-Hallucination" principle.

**Multi-channel by design:** DoraEdu currently ships as a Telegram bot, with Zalo bot support planned as a
parallel channel. Keep messaging-platform logic isolated behind a common `BotAdapter`/`ChannelAdapter`
interface — never hardcode Telegram-specific assumptions into shared RAG/LLM/session logic, so a Zalo adapter
can be added later with minimal refactoring.

## 2. Technical Stack
- **Architecture:** Refer to `dora_edu_architecture.md` in the root folder to understand the module
  interactions and data flow before implementing changes.
- **Language:** Python 3.14
- **Project Structure:** PEP 621 (`pyproject.toml`) using the `src-layout` (`src/dora_edu/...`).
- **Vector DB:** ChromaDB
- **LLM/Embeddings:** API-based (OpenAI/Gemini) or local `sentence-transformers`.
- **Messaging channels:** Telegram (current), Zalo (planned) — via a shared adapter interface.

## 3. Strict Coding Rules (MUST FOLLOW)
1. **Language:** Write all code, variable names, function names, and docstrings in **English**. (Only the
   actual user prompts/chat messages should be in Vietnamese).
2. **Type Hinting:** You MUST use Python static type hinting for all function arguments and return types.
   Use modern Python 3 syntax (e.g., `list[str]`).
3. **Dependency Management:** Do NOT generate `requirements.txt`. We use `pyproject.toml`. If a new package
   is needed, instruct the user to add it to `pyproject.toml`.
4. **Channel-Agnostic Bot Layer:** Do not couple RAG/LLM/session logic to Telegram-specific types or APIs.
   New channels (starting with Zalo) must be addable via a new adapter, without touching core retrieval or
   generation logic.
5. **Metadata Filtering:** Every ChromaDB query MUST filter by `grade` and `subject` to ensure cross-grade
   and cross-subject data isolation.
6. **No Hallucination:** When writing Prompt templates, always include strict instructions preventing the
   LLM from using outside knowledge.
7. **Error Handling:** Always wrap external API calls (OpenAI, Gemini, Telegram API, Zalo API, etc.) in
   try-except blocks and catch specific exceptions.
