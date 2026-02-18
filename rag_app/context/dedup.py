"""
Déduplication de chunks pour le RAG juridique.

Élimine les doublons et chunks trop similaires pour optimiser
le contexte envoyé au LLM.

Usage:
    from rag_app.context import ChunkDeduplicator

    dedup = ChunkDeduplicator(similarity_threshold=0.9)
    unique_docs = dedup.deduplicate(documents)
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Set

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from rag_bench.logging_utils import setup_logging

logger = setup_logging()


class ChunkDeduplicator:
    """
    Déduplique les chunks similaires ou identiques.

    Stratégies de déduplication :
    - exact: Hash MD5 du contenu (chunks identiques)
    - chunk_id: Basé sur chunk_id (même source)
    - fuzzy: Similarité de contenu (non implémenté, nécessite embeddings)

    Attributes:
        strategy: Stratégie de déduplication (exact, chunk_id)
        keep_first: Garder le premier ou le dernier doublon

    Example:
        >>> dedup = ChunkDeduplicator(strategy="exact")
        >>> unique = dedup.deduplicate(documents)
        >>> print(f"Dédupliqués: {len(documents)} -> {len(unique)}")
    """

    def __init__(
        self,
        strategy: str = "exact",
        keep_first: bool = True,
    ):
        """
        Initialise le déduplicateur.

        Args:
            strategy: "exact" (hash contenu) ou "chunk_id" (même ID)
            keep_first: Garder premier (True) ou dernier (False) doublon
        """
        if strategy not in ("exact", "chunk_id"):
            raise ValueError(f"Stratégie inconnue: {strategy}. Utilisez 'exact' ou 'chunk_id'")

        self.strategy = strategy
        self.keep_first = keep_first

    def _content_hash(self, text: str) -> str:
        """Calcule un hash MD5 du contenu normalisé."""
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _get_key(self, doc: Document) -> str:
        """Extrait la clé de déduplication selon la stratégie."""
        if self.strategy == "chunk_id":
            return doc.metadata.get("chunk_id", self._content_hash(doc.page_content))
        else:  # exact
            return self._content_hash(doc.page_content)

    def deduplicate(self, documents: List[Document]) -> List[Document]:
        """
        Déduplique une liste de documents.

        Args:
            documents: Liste de Documents LangChain

        Returns:
            Liste sans doublons (ordre préservé)
        """
        if not documents:
            return []

        seen: Set[str] = set()
        unique: List[Document] = []

        # Si keep_first=False, on inverse puis re-inverse
        docs_to_process = documents if self.keep_first else list(reversed(documents))

        for doc in docs_to_process:
            key = self._get_key(doc)
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        if not self.keep_first:
            unique = list(reversed(unique))

        removed = len(documents) - len(unique)
        if removed > 0:
            logger.debug(
                "Déduplication (%s): %d -> %d (-%d)",
                self.strategy,
                len(documents),
                len(unique),
                removed,
            )

        return unique

    def deduplicate_by_article(self, documents: List[Document]) -> List[Document]:
        """
        Déduplique en gardant un seul chunk par article.

        Utile quand plusieurs chunks du même article sont récupérés,
        on garde celui avec le meilleur score.

        Args:
            documents: Liste de Documents (triés par score décroissant)

        Returns:
            Un document par article unique
        """
        seen_articles: Set[str] = set()
        unique: List[Document] = []

        for doc in documents:
            article = doc.metadata.get("article_num") or doc.metadata.get("doc_key")
            if article and article not in seen_articles:
                seen_articles.add(article)
                unique.append(doc)
            elif not article:
                # Pas d'article identifié, garder
                unique.append(doc)

        return unique

    def as_runnable(self) -> RunnableLambda:
        """Convertit en Runnable LangChain pour LCEL."""
        return RunnableLambda(self.deduplicate)

    def __call__(self, documents: List[Document]) -> List[Document]:
        """Raccourci pour deduplicate()."""
        return self.deduplicate(documents)

    def __repr__(self) -> str:
        return f"ChunkDeduplicator(strategy='{self.strategy}', keep_first={self.keep_first})"
