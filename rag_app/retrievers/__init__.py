"""
Retrievers LangChain wrappant nos modules rag_bench existants.

Ces wrappers ne dupliquent PAS la logique - ils appellent nos modules existants
et convertissent les résultats au format LangChain (Document).

Usage:
    from rag_app.retrievers import LegalBM25Retriever, LegalDenseRetriever, LegalHybridRetriever

    retriever = LegalHybridRetriever(
        corpus_path="/path/to/gold/chunks",
        k=10,
        filters={"corpus_juridique": "LEGITEXT000006072050"}
    )
    docs = retriever.invoke("durée période essai CDI")
"""

from rag_app.retrievers.bm25_retriever import LegalBM25Retriever
from rag_app.retrievers.dense_retriever import LegalDenseRetriever
from rag_app.retrievers.hybrid_retriever import LegalHybridRetriever

__all__ = [
    "LegalBM25Retriever",
    "LegalDenseRetriever",
    "LegalHybridRetriever",
]
