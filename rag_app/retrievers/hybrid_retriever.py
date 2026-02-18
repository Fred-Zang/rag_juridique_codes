"""
Wrapper LangChain pour notre Hybrid retriever (BM25 + Dense + RRF).

RÉUTILISE les modules rag_bench existants sans duplication.

Usage:
    from rag_app.retrievers import LegalHybridRetriever

    retriever = LegalHybridRetriever(
        corpus_path="/path/to/gold/chunks",
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        k=10,
        bm25_weight=0.5,
        dense_weight=0.5,
        rrf_k=60,
        filters={"status_in": ["VIGUEUR"]},
    )

    docs = retriever.invoke("durée période essai CDI")
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr

# ═══════════════════════════════════════════════════════════════════════════
# IMPORT DE NOS MODULES EXISTANTS (pas de duplication !)
# ═══════════════════════════════════════════════════════════════════════════
from rag_bench.io_parquet import read_chunks_from_parquet
from rag_bench.filtering import filter_chunks
from rag_bench.bm25 import build_bm25_index, BM25Index
from rag_bench.core.dense import build_dense_embeddings
from rag_bench.paths import get_project_paths


logger = logging.getLogger(__name__)

def _stable_hash(payload: Dict[str, Any]) -> str:
    """Empreinte stable (sert de tag pour le cache embeddings)."""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _resolve_cache_dir(cache_dir: Optional[str]) -> str:
    """
    Résout le dossier de cache embeddings de manière portable.

    Règles:
    - si cache_dir n'est pas fourni (ou vaut '.dense_cache'), on utilise le cache central du projet
      (result_tests/cache_dense) via get_project_paths().
    - sinon, on résout le chemin fourni en absolu (pour éviter une dépendance au CWD).
    """
    if (cache_dir is None) or (str(cache_dir).strip() in {"", ".dense_cache"}):
        return str(get_project_paths().cache_dense_dir)

    return str(Path(str(cache_dir)).expanduser().resolve())


def reciprocal_rank_fusion(
    rankings: List[List[str]],
    weights: Optional[List[float]] = None,
    k: int = 60,
) -> List[tuple[str, float]]:
    """
    Fusionne plusieurs rankings via Reciprocal Rank Fusion (RRF).

    Args:
        rankings: Liste de rankings (chaque ranking = liste d'IDs ordonnés)
        weights: Poids optionnels pour chaque ranking
        k: Paramètre RRF (typiquement 60)

    Returns:
        Liste de tuples (doc_id, rrf_score) triés par score décroissant
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    rrf_scores: Dict[str, float] = {}

    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += weight / (k + rank)

    # Trier par score décroissant
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


class LegalHybridRetriever(BaseRetriever):
    """
    Retriever Hybride (BM25 + Dense + RRF) pour corpus juridique français.

    Combine recherche lexicale (BM25) et sémantique (Dense) via
    Reciprocal Rank Fusion pour de meilleurs résultats.

    Attributes:
        corpus_path: Chemin vers gold/chunks (Parquet)
        embedding_model: Nom du modèle sentence-transformers
        k: Nombre de résultats finaux
        bm25_k: Nombre de candidats BM25 (avant fusion)
        dense_k: Nombre de candidats Dense (avant fusion)
        bm25_weight: Poids BM25 dans la fusion RRF
        dense_weight: Poids Dense dans la fusion RRF
        rrf_k: Paramètre k de RRF (typiquement 60)
        text_field: Champ texte à utiliser
        cache_dir: Dossier pour cache embeddings
        device: Device pour encoding (cpu/cuda)
        filters: Filtres métier

    Example:
        >>> retriever = LegalHybridRetriever(
        ...     corpus_path="/path/to/gold/chunks",
        ...     k=10,
        ...     bm25_weight=0.5,
        ...     dense_weight=0.5,
        ... )
        >>> docs = retriever.invoke("obligations employeur")
    """

    # Champs publics
    corpus_path: str = Field(description="Chemin vers gold/chunks")
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Modèle sentence-transformers"
    )
    k: int = Field(default=10, ge=1, le=100, description="Nombre de résultats finaux")
    bm25_k: int = Field(default=100, ge=1, description="Candidats BM25 avant fusion")
    dense_k: int = Field(default=100, ge=1, description="Candidats Dense avant fusion")
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Poids BM25")
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Poids Dense")
    rrf_k: int = Field(default=60, ge=1, description="Paramètre RRF k")
    text_field: str = Field(default="text", description="Champ texte")
    cache_dir: Optional[str] = Field(default=".dense_cache", description="Cache embeddings")
    device: str = Field(default="cpu", description="Device (cpu/cuda)")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filtres métier")

    # Attributs privés
    _chunks: List[Dict] = PrivateAttr(default_factory=list)
    _filtered_chunks: List[Dict] = PrivateAttr(default_factory=list)
    _chunk_id_to_idx: Dict[str, int] = PrivateAttr(default_factory=dict)
    _bm25_index: Any = PrivateAttr(default=None)
    _dense_model: Any = PrivateAttr(default=None)
    _doc_embeddings: Any = PrivateAttr(default=None)
    _loaded: bool = PrivateAttr(default=False)

    def _load_corpus_and_indexes(self) -> None:
        """Charge corpus et construit les deux index (lazy loading)."""
        if self._loaded:
            return

        logger.info("Chargement corpus Hybrid: %s", self.corpus_path)

        # 1. Charger corpus
        self._chunks = read_chunks_from_parquet(
            parquet_path=str(Path(self.corpus_path).expanduser().resolve()),
            use_pandas=True,
        )
        logger.info("Corpus chargé: %d chunks", len(self._chunks))

        # 2. Appliquer filtres
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
            logger.info("Après filtrage: %d chunks", len(self._filtered_chunks))
        else:
            self._filtered_chunks = self._chunks

        # 3. Construire mapping chunk_id → index
        self._chunk_id_to_idx = {
            chunk.get("chunk_id"): idx
            for idx, chunk in enumerate(self._filtered_chunks)
        }

        # 4. Construire index BM25
        self._bm25_index = build_bm25_index(
            chunks=self._filtered_chunks,
            text_field=self.text_field,
        )
        logger.info("Index BM25 construit")

        # 5. Construire embeddings Dense
        resolved_corpus_path = str(Path(self.corpus_path).expanduser().resolve())

        # Tag stable: évite les noms de fichiers contenant des dicts ({...})
        # et garantit la réutilisation du cache si (corpus, filtres, modèle, text_field, device) sont identiques.
        cache_tag = _stable_hash({
            "retriever": "hybrid",
            "corpus_parquet": resolved_corpus_path,
            "filters": self.filters,
            "text_field": self.text_field,
            "dense_model": self.embedding_model,
            "dense_device": self.device,
        })
        self._dense_model, self._doc_embeddings = build_dense_embeddings(
            documents=self._filtered_chunks,
            model_name=self.embedding_model,
            cache_dir=_resolve_cache_dir(self.cache_dir),
            cache_tag=cache_tag,
            text_field=self.text_field,
            device=self.device,
            logger=logger,
        )
        logger.info("Embeddings Dense prêts: shape=%s", self._doc_embeddings.shape)

        self._loaded = True

    def _bm25_search(self, query: str, k: int) -> List[str]:
        """Recherche BM25 et retourne liste d'IDs ordonnés."""
        # BM25Index.search retourne List[Tuple[chunk_dict, score]]
        results = self._bm25_index.search(query=query, k=k)
        return [hit.get("chunk_id") for hit, score in results]

    def _dense_search(self, query: str, k: int) -> List[str]:
        """Recherche Dense et retourne liste d'IDs ordonnés."""
        # Encoder la query
        query_embedding = self._dense_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        # Calcul similarité cosine
        scores = np.dot(self._doc_embeddings, query_embedding.T).flatten()

        # Top-k indices
        top_indices = np.argsort(scores)[::-1][:k]

        return [self._filtered_chunks[idx].get("chunk_id") for idx in top_indices]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Recherche hybride (BM25 + Dense + RRF) et conversion en Documents.
        """
        self._load_corpus_and_indexes()

        # 1. Recherche BM25
        bm25_ranking = self._bm25_search(query, self.bm25_k)

        # 2. Recherche Dense
        dense_ranking = self._dense_search(query, self.dense_k)

        # 3. Fusion RRF
        fused_results = reciprocal_rank_fusion(
            rankings=[bm25_ranking, dense_ranking],
            weights=[self.bm25_weight, self.dense_weight],
            k=self.rrf_k,
        )

        # 4. Prendre top-k
        top_results = fused_results[:self.k]

        # 5. Conversion en Documents
        docs = []
        for rank, (chunk_id, rrf_score) in enumerate(top_results, start=1):
            idx = self._chunk_id_to_idx.get(chunk_id)
            if idx is None:
                continue

            chunk = self._filtered_chunks[idx]
            meta = chunk.get("meta", {})

            # Déterminer la source (BM25, Dense, ou les deux)
            in_bm25 = chunk_id in bm25_ranking[:self.k]
            in_dense = chunk_id in dense_ranking[:self.k]
            if in_bm25 and in_dense:
                source = "hybrid_both"
            elif in_bm25:
                source = "hybrid_bm25"
            else:
                source = "hybrid_dense"

            doc = Document(
                page_content=chunk.get(self.text_field) or chunk.get("text", ""),
                metadata={
                    # Identifiants
                    "doc_key": meta.get("doc_key"),
                    "chunk_id": chunk_id,
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
                    "score": rrf_score,
                    "rank": rank,
                    "retriever": "hybrid",
                    "source": source,  # hybrid_both, hybrid_bm25, hybrid_dense

                    # Audit
                    "source_path": meta.get("source_path"),
                },
            )
            docs.append(doc)

        return docs
