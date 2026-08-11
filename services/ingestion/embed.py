"""Embedding generation. Defaults to a local sentence-transformers model
(BAAI/bge-base-en-v1.5) so ingestion and CI need no API key — see ADR-0003."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from services.config import get_settings


@lru_cache
def _get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    if settings.embedding_provider != "local":
        raise NotImplementedError(
            f"embedding_provider={settings.embedding_provider!r} not implemented; "
            "only 'local' (bge-base-en-v1.5) is wired up. See ADR-0003."
        )
    model = _get_model(settings.embedding_model)
    # bge models recommend this instruction prefix for retrieval-style passages.
    prefixed = [f"passage: {t}" for t in texts]
    vectors: np.ndarray = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    settings = get_settings()
    model = _get_model(settings.embedding_model)
    vector: np.ndarray = model.encode([f"query: {query}"], normalize_embeddings=True, show_progress_bar=False)
    return vector[0].tolist()
