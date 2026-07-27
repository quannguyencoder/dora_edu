Project Architecture: DoraEdu (The Magic Textbook Pouch)

An offline-first, Zero-Hallucination Retrieval-Augmented Generation (RAG) system for Vietnamese students, based strictly on textbooks from the Ministry of Education and Training (MOET).

1. Directory Structure (Modern src layout)

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
│       ├── config.py           # Environment variables and system settings
│       ├── data_pipeline/
│       │   ├── __init__.py
│       │   ├── pdf_parser.py   # PDF extraction logic
│       │   └── text_chunker.py # Overlap chunking logic
│       ├── rag_engine/
│       │   ├── __init__.py
│       │   ├── indexer.py      # Embedding and ChromaDB insertion
│       │   └── retriever.py    # Query logic with metadata filtering (Subject/Grade)
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── generator.py    # LLM API communication (OpenAI/Gemini/Claude)
│       │   └── prompts.py      # System prompts and strict RAG guardrails
│       └── bot/
│           ├── __init__.py
│           ├── session.py      # Sliding window chat history management
│           └── telegram_app.py # Telegram bot webhook/polling entry point
│
├── tests/                      # Unit tests (pytest)
│   └── ...
│
├── .env.example                # Example environment variables
├── .gitignore
├── pyproject.toml              # Modern Python dependency & project manager (PEP 621)
├── dora_edu_architecture.md    # Project blueprint and module explanations
└── README.md



2. Core Data Flow

Ingestion (Admin): pdf_parser.py -> text_chunker.py -> indexer.py (Saves to vector_db/ with grade and subject metadata).

Retrieval (Student): telegram_app.py receives query -> retriever.py searches ChromaDB matching the student's grade -> generator.py formats prompt with Context -> LLM returns the exact answer.