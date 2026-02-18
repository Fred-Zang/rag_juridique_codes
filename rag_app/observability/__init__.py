"""
Observabilité et monitoring pour le RAG juridique.

Ce module fournit l'intégration avec Langfuse (self-hosted)
pour le suivi des requêtes, latences, coûts et qualité.

Usage:
    from rag_app.observability import get_langfuse_handler, get_langfuse_context

    handler = get_langfuse_handler()

    with get_langfuse_context(
        session_id="user-123",
        user_id="fred",
        tags=["api", "endpoint:/rag/invoke"],
        metadata={"request_id": "abc"},
    ):
        result = chain.invoke(
            {"question": "..."},
            config={"callbacks": [handler]},
        )
"""

from rag_app.observability.langfuse_setup import get_langfuse_context, get_langfuse_handler
from rag_app.observability.run_manager import RAGRunManager

__all__ = [
    "get_langfuse_handler",
    "get_langfuse_context",
    "RAGRunManager",
]
