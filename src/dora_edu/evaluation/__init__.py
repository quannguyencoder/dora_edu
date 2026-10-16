"""Offline quality measurement for the RAG pipeline.

Separate from ``tests/``: the test suite proves the code behaves as written,
using fakes and no network. This package measures how well the *real* index
and the *real* LLM answer real textbook questions -- which can only be done
by asking them, and which therefore costs API calls and takes minutes.
"""
