from __future__ import annotations

from .config import Settings
from .pipeline import SecureRAGPipeline
from .rag import Embedder, VectorStore


def create_pipeline(settings: Settings | None = None) -> SecureRAGPipeline:
    settings = settings or Settings.from_env()
    embedder = Embedder(settings.ollama_url, settings.ollama_embed_model)
    store = VectorStore(settings.vector_db_path, embedder)
    return SecureRAGPipeline(settings, store)

