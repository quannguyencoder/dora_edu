"""Shared ChromaDB client and embedding function.

The indexer and the retriever must agree on the exact same embedding model and
collection settings, otherwise queries silently return nothing useful. Both go
through this module so that agreement is structural rather than accidental.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaClientSettings
from chromadb.utils import embedding_functions

from dora_edu.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Cosine distance keeps scores in a stable 0..2 range across embedding models,
#: which is what makes ``max_retrieval_distance`` meaningful as a cut-off.
_DISTANCE_SPACE = "cosine"


@lru_cache(maxsize=4)
def _build_embedding_function(model_name: str):
    """Load the local sentence-transformers embedding model.

    Cached because loading the model costs seconds and hundreds of megabytes.

    Raises:
        RuntimeError: If the model cannot be loaded (e.g. no local copy and no
            network access on first run).
    """
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    except (OSError, ValueError, ImportError) as exc:
        raise RuntimeError(
            f"Cannot load embedding model {model_name!r}. "
            "Check EMBEDDING_MODEL, or pre-download the model for offline use."
        ) from exc


def get_collection(settings: Settings | None = None, *, create: bool = True) -> Collection:
    """Open the textbook collection, creating it when missing.

    Args:
        settings: Application settings; the process singleton when omitted.
        create: Create the collection if it does not exist yet. Pass ``False``
            on the read path so that querying an un-ingested database fails
            loudly instead of returning an empty collection.

    Returns:
        The ChromaDB collection holding every indexed textbook chunk.

    Raises:
        RuntimeError: If the persistent client or the collection cannot be opened.
    """
    settings = settings or get_settings()
    embedding_function = _build_embedding_function(settings.embedding_model)

    try:
        settings.chroma_db_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(settings.chroma_db_path),
            settings=ChromaClientSettings(anonymized_telemetry=False, allow_reset=False),
        )
        if create:
            return client.get_or_create_collection(
                name=settings.chroma_collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": _DISTANCE_SPACE},
            )
        return client.get_collection(
            name=settings.chroma_collection_name,
            embedding_function=embedding_function,
        )
    except (OSError, ValueError, chromadb.errors.ChromaError) as exc:
        raise RuntimeError(
            f"Cannot open ChromaDB collection {settings.chroma_collection_name!r} "
            f"at {settings.chroma_db_path}: {exc}"
        ) from exc
