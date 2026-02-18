"""
Wrapper LangChain pour notre BM25 existant.

RÉUTILISE rag_bench/bm25.py sans duplication de logique.

Usage:
    from rag_app.retrievers import LegalBM25Retriever

    retriever = LegalBM25Retriever(
        corpus_path="/path/to/gold/chunks",
        k=10,
        filters={"status_in": ["VIGUEUR"], "corpus_juridique": "LEGITEXT000006072050"}
    )

    # Utilisation standalone
    docs = retriever.invoke("durée période essai")

    # Utilisation dans une chain LCEL
    chain = retriever | format_docs | prompt | llm
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr

# ═══════════════════════════════════════════════════════════════════════════
# IMPORT DE NOS MODULES EXISTANTS (pas de duplication !)
# ═══════════════════════════════════════════════════════════════════════════
from rag_bench.bm25 import build_bm25_index, BM25Index
from rag_bench.io_parquet import read_chunks_from_parquet
from rag_bench.filtering import filter_chunks


logger = logging.getLogger(__name__)


class LegalBM25Retriever(BaseRetriever):
    """
    Retriever BM25 pour corpus juridique français.

    Wrap notre implémentation rank-bm25 existante (rag_bench/bm25.py)
    avec l'interface LangChain BaseRetriever pour intégration LCEL.

    Attributes:
        corpus_path: Chemin vers gold/chunks (Parquet)
        k: Nombre de résultats à retourner
        text_field: Champ texte à utiliser pour BM25
        filters: Filtres métier à appliquer (doc_types, status_in, corpus_juridique, as_of)

    Example:
        >>> retriever = LegalBM25Retriever(
        ...     corpus_path="/path/to/gold/chunks",
        ...     k=10,
        ...     filters={"status_in": ["VIGUEUR"]}
        ... )
        >>> docs = retriever.invoke("durée période essai")
        >>> print(docs[0].page_content[:100])
    """

    # Champs publics (configurables)
    corpus_path: str = Field(description="Chemin vers gold/chunks (Parquet)")
    k: int = Field(default=10, ge=1, le=100, description="Nombre de résultats")
    text_field: str = Field(default="text", description="Champ texte pour BM25")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filtres métier")

    # Attributs privés (non sérialisés par Pydantic)
    _chunks: List[Dict] = PrivateAttr(default_factory=list)
    _filtered_chunks: List[Dict] = PrivateAttr(default_factory=list)
    _index: Any = PrivateAttr(default=None)  # BM25Index
    _loaded: bool = PrivateAttr(default=False)

    def _load_corpus(self) -> None:
        """Charge le corpus et construit l'index BM25 (lazy loading)."""
        if self._loaded:
            return

        logger.info("Chargement corpus BM25: %s", self.corpus_path)

        # 1. Charger corpus via notre module existant
        self._chunks = read_chunks_from_parquet(
            parquet_path=self.corpus_path,
            use_pandas=True,
        )
        logger.info("Corpus chargé: %d chunks", len(self._chunks))

        # 2. Appliquer filtres métier via notre module existant
        if self.filters:
            filter_stats = {}
            self._filtered_chunks = filter_chunks(
                self._chunks,
                doc_types=self.filters.get("doc_types"),
                corpus_juridique=self.filters.get("corpus_juridique"),
                status_in=self.filters.get("status_in"),
                as_of=self.filters.get("as_of"),
                strict_temporal=self.filters.get("strict_temporal", False),
                stats=filter_stats,
            )
            logger.info("Après filtrage: %d chunks | stats=%s", len(self._filtered_chunks), filter_stats)
        else:
            self._filtered_chunks = self._chunks

        # 3. Construire index BM25 via notre module existant
        self._index = build_bm25_index(
            chunks=self._filtered_chunks,
            text_field=self.text_field,
        )
        logger.info("Index BM25 construit")

        self._loaded = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Recherche BM25 et conversion en Documents LangChain.

        Méthode requise par BaseRetriever.
        """
        self._load_corpus()

        # Recherche via notre module existant (BM25Index.search)
        results = self._index.search(query=query, k=self.k)

        # Conversion en Documents LangChain
        # results est une liste de tuples (chunk_dict, score)
        docs = []
        for rank, (hit, score) in enumerate(results, start=1):
            meta = hit.get("meta", {})

            doc = Document(
                page_content=hit.get(self.text_field) or hit.get("text", ""),
                metadata={
                    # Identifiants
                    "doc_key": meta.get("doc_key"),
                    "chunk_id": hit.get("chunk_id"),
                    "article_num": meta.get("article_num"),

                    # Temporel
                    "valid_from": meta.get("valid_from"),
                    "valid_to": meta.get("valid_to"),
                    "status": meta.get("status"),

                    # Structurel
                    "corpus_juridique": meta.get("corpus_juridique"),
                    "code_titre": meta.get("code_titre"),
                    "doc_type": meta.get("doc_type"),

                    # Retrieval info
                    "score": score,
                    "rank": rank,
                    "retriever": "bm25",

                    # Audit
                    "source_path": meta.get("source_path"),
                },
            )
            docs.append(doc)

        return docs
