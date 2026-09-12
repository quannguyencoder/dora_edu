Project Architecture: DoraEdu (The Magic Textbook Pouch)

An offline-first, Zero-Hallucination Retrieval-Augmented Generation (RAG) system for Vietnamese students, based strictly on textbooks from the Ministry of Education and Training (MOET).

1. Directory Structure (Modern src layout)

```
dora-edu/
│
├── .github/
│   └── copilot-instructions.md # Core rules for GitHub Copilot
│
├── .claude.md                  # Project instructions for Claude (Claude Projects)
│
├── data/                       # (Excluded from Git via .gitignore)
│   ├── raw_pdfs/               # Original MOET textbook PDFs
│   └── processed/              # Cleaned text data ready for chunking
│
├── vector_db/                  # (Excluded from Git via .gitignore)
│   └── ...                     # ChromaDB persistence storage
│
├── src/
│   └── dora_edu/               # Main application package
│       ├── __init__.py
│       ├── config.py           # Environment variables and system settings (pydantic-settings)
│       ├── models.py           # Shared domain models: StudentProfile, TextbookMetadata,
│       │                       #   ParsedPage, TextChunk, RetrievedChunk + subject/grade
│       │                       #   normalisation. Platform- and vendor-neutral.
│       ├── data_pipeline/
│       │   ├── __init__.py
│       │   ├── pdf_parser.py   # PDF extraction + local Tesseract OCR fallback for scans
│       │   ├── text_chunker.py # Sentence-aligned overlap chunking
│       │   ├── ingest.py       # `dora-ingest` CLI: parse -> chunk -> index (one book)
│       │   ├── catalog.py      # Infers grade/subject/title from real MOET filenames
│       │   └── bulk_ingest.py  # `dora-bulk-ingest` CLI: ingests data/raw_pdfs/<grade>/*
│       ├── rag_engine/
│       │   ├── __init__.py
│       │   ├── store.py        # Shared ChromaDB client + embedding function
│       │   ├── indexer.py      # Embedding and ChromaDB insertion (idempotent upsert)
│       │   └── retriever.py    # Query logic with metadata filtering (Subject/Grade)
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── generator.py    # LLM API communication (AnswerGenerator interface + OpenAI)
│       │   └── prompts.py      # System prompts and strict RAG guardrails (Vietnamese)
│       └── bot/
│           ├── __init__.py
│           ├── adapter.py      # ChannelAdapter interface + IncomingMessage/OutgoingMessage
│           ├── core.py         # TutorService: channel-independent tutoring brain
│           ├── session.py      # Sliding window chat history + student profile
│           ├── discord/
│           │   ├── __init__.py
│           │   ├── adapter.py  # Discord-specific adapter (gateway/WebSocket)
│           │   └── app.py      # `dora-run-discord-bot` entry point -- primary channel
│           ├── telegram/
│           │   ├── __init__.py
│           │   ├── adapter.py  # Telegram-specific adapter (long polling)
│           │   └── app.py      # `dora-run-bot` entry point
│           └── zalo/
│               ├── __init__.py
│               ├── webhook.py  # Webhook payload models + HMAC signature check
│               ├── adapter.py  # Zalo-specific adapter (webhook server + Send API)
│               └── app.py      # `dora-run-zalo-bot` entry point
│
├── tests/                      # Unit tests (pytest)
│   ├── conftest.py             # Fake ChromaDB collection and fake generator
│   ├── test_models.py
│   ├── test_pdf_parser.py
│   ├── test_text_chunker.py
│   ├── test_retriever.py       # Grade/subject isolation rule
│   ├── test_prompts.py         # Anti-hallucination + Socratic guardrails
│   ├── test_generator.py
│   ├── test_session.py
│   ├── test_core.py
│   ├── test_adapter.py         # Proves a new channel needs no core changes
│   ├── test_catalog.py         # Filename -> grade/subject/title inference
│   ├── test_indexer.py         # Upsert idempotency + bulk-ingest resume check
│   ├── test_bulk_ingest.py     # Resume/--force orchestration
│   ├── test_discord_adapter.py # DiscordAdapter against fake gateway objects
│   ├── test_zalo_webhook.py    # Signature verification + payload validation
│   └── test_zalo_adapter.py    # ZaloAdapter against a fake httpx client
│
├── .env.example                # Example environment variables
├── .gitignore
├── pyproject.toml              # Modern Python dependency & project manager (PEP 621)
├── dora_edu_architecture.md    # Project blueprint and module explanations
└── README.md
```


2. Core Data Flow

Ingestion (Admin): the real MOET textbook corpus was collected out of band into `data/raw_pdfs/<grade>/`
(gitignored). Most of those PDFs are scans with no text layer, so `pdf_parser.py` extracts embedded text
where it exists and falls back to local Tesseract OCR (Vietnamese + English) on the page image otherwise --
nothing ever leaves the machine. `dora-bulk-ingest` walks that whole tree, using `catalog.py` to infer each
book's grade/subject/title from its filename, and drives the same per-book path as the single-book
`dora-ingest` CLI: `text_chunker.py` builds sentence-aligned chunks with overlap -> `indexer.py` embeds them
and upserts into `vector_db/`, stamping every row with `grade` and `subject`.

Retrieval (Student): the channel adapter (`bot/discord/adapter.py`, `bot/telegram/adapter.py`, or
`bot/zalo/adapter.py`) normalises the platform message into an `IncomingMessage` -> `bot/core.py` resolves
the student's session and profile -> `retriever.py` queries ChromaDB **always filtered by that student's
`grade` and `subject`** -> `generator.py` builds the pedagogical prompt from `prompts.py` around the
retrieved context -> the reply travels back out through the same adapter (a Discord channel message, a
Telegram message, or a Zalo OA Send API call).


3. Layering Rules

- `models.py`, `rag_engine/`, `llm/` and `bot/core.py` never import a messaging library. `bot/discord/` and
  `bot/zalo/` were each added as a sibling of `bot/telegram/`, written against `ChannelAdapter` alone, with
  zero changes to any of those layers -- `test_adapter.py`, `test_discord_adapter.py` and `test_zalo_adapter.py`
  all exercise that same unchanged `TutorService`.
- `store.py` is the single place that decides the embedding model and distance metric, so the indexer and
  the retriever cannot drift apart.
- `Retriever.retrieve()` only accepts a `StudentProfile`, whose `grade` and `subject` are both mandatory and
  validated. That is what makes the isolation rule structural rather than a convention.
- `generator.py` returns the fixed "not in your textbook" answer without calling the LLM at all when
  retrieval comes back empty.
- `pdf_parser.py`'s OCR fallback only ever reads the PDF's own embedded page image and calls the local
  `tesseract` binary -- no page is ever sent to a network service, per the data-privacy rule.
- `bulk_ingest.py` never guesses: a filename `catalog.py` cannot parse, or whose filename-encoded grade
  disagrees with its folder, is skipped and reported rather than indexed under a wrong grade/subject.
