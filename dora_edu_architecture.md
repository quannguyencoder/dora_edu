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
│       │   ├── pdf_parser.py   # PDF extraction, cleaning, running-header removal
│       │   ├── text_chunker.py # Sentence-aligned overlap chunking
│       │   └── ingest.py       # `dora-ingest` CLI: parse -> chunk -> index
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
│           ├── telegram/
│           │   ├── __init__.py
│           │   ├── adapter.py  # Telegram-specific adapter (long polling)
│           │   └── app.py      # `dora-run-bot` entry point
│           └── zalo/           # (Planned) Zalo adapter - a sibling of telegram/
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
│   └── test_adapter.py         # Proves a new channel needs no core changes
│
├── .env.example                # Example environment variables
├── .gitignore
├── pyproject.toml              # Modern Python dependency & project manager (PEP 621)
├── dora_edu_architecture.md    # Project blueprint and module explanations
└── README.md
```


2. Core Data Flow

Ingestion (Admin): `ingest.py` walks the PDFs -> `pdf_parser.py` extracts and cleans each page ->
`text_chunker.py` builds sentence-aligned chunks with overlap -> `indexer.py` embeds them and upserts
into `vector_db/`, stamping every row with `grade` and `subject`.

Retrieval (Student): the channel adapter (`bot/telegram/adapter.py`) normalises the platform message into
an `IncomingMessage` -> `bot/core.py` resolves the student's session and profile -> `retriever.py` queries
ChromaDB **always filtered by that student's `grade` and `subject`** -> `generator.py` builds the
pedagogical prompt from `prompts.py` around the retrieved context -> the reply travels back out through the
same adapter.


3. Layering Rules

- `models.py`, `rag_engine/`, `llm/` and `bot/core.py` never import a messaging library. Adding Zalo means
  writing `bot/zalo/adapter.py` against `ChannelAdapter` and nothing else.
- `store.py` is the single place that decides the embedding model and distance metric, so the indexer and
  the retriever cannot drift apart.
- `Retriever.retrieve()` only accepts a `StudentProfile`, whose `grade` and `subject` are both mandatory and
  validated. That is what makes the isolation rule structural rather than a convention.
- `generator.py` returns the fixed "not in your textbook" answer without calling the LLM at all when
  retrieval comes back empty.
