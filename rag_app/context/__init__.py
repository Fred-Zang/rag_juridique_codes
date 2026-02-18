"""
Construction du contexte LLM pour le RAG juridique.

Ce module gère la préparation du contexte envoyé au LLM :
- Formatage des documents
- Déduplication de chunks similaires
- Limitation du contexte (tokens)
- Citations et références

Usage:
    from rag_app.context import ContextBuilder, format_docs_for_llm

    builder = ContextBuilder(max_tokens=4000, include_citations=True)
    context = builder.build(documents, query)
"""

from rag_app.context.builder import ContextBuilder, format_docs_for_llm
from rag_app.context.dedup import ChunkDeduplicator

__all__ = [
    "ContextBuilder",
    "format_docs_for_llm",
    "ChunkDeduplicator",
]
