"""
Wrapper LangChain pour notre Dense retriever existant.

RÉUTILISE rag_bench/dense.py et rag_bench/core/dense.py sans duplication.

Usage:
    from rag_app.retrievers import LegalDenseRetriever

    retriever = LegalDenseRetriever(
        corpus_path="/path/to/gold/chunks",
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        k=10,
        cache_dir=".dense_cache",
        device="cuda",
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




class LegalDenseRetriever(BaseRetriever):
    """
    Retriever Dense (vectoriel) pour corpus juridique français.

    Wrap notre implémentation sentence-transformers existante
    avec l'interface LangChain BaseRetriever.

    Attributes:
        corpus_path: Chemin vers gold/chunks (Parquet)
        embedding_model: Nom du modèle sentence-transformers
        k: Nombre de résultats à retourner
        text_field: Champ texte à encoder
        cache_dir: Dossier pour cache embeddings
        device: Device pour encoding (cpu/cuda)
        filters: Filtres métier

    Example:
        >>> retriever = LegalDenseRetriever(
        ...     corpus_path="/path/to/gold/chunks",
        ...     embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ...     k=10,
        ... )
        >>> docs = retriever.invoke("obligations employeur")
    """

    # Champs publics
    corpus_path: str = Field(description="Chemin vers gold/chunks")
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Modèle sentence-transformers"
    )
    k: int = Field(default=10, ge=1, le=100, description="Nombre de résultats")
    text_field: str = Field(default="text", description="Champ texte")
    cache_dir: Optional[str] = Field(default=".dense_cache", description="Cache embeddings")
    # Ajout : permet de réutiliser un cache existant créé par rag_bench
    cache_tag: Optional[str] = Field(
        default=None,
        description="Tag de cache embeddings (si fourni, force la réutilisation du cache)")

    device: str = Field(default="cpu", description="Device (cpu/cuda)")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filtres métier")

    # Attributs privés
    _chunks: List[Dict] = PrivateAttr(default_factory=list)
    _filtered_chunks: List[Dict] = PrivateAttr(default_factory=list)
    _model: Any = PrivateAttr(default=None)
    _doc_embeddings: Any = PrivateAttr(default=None)  # np.ndarray
    _loaded: bool = PrivateAttr(default=False)

    def _load_corpus_and_embeddings(self) -> None:
        """Charge corpus et embeddings (lazy loading avec cache)."""
        if self._loaded:
            return

        logger.info("Chargement corpus Dense: %s", self.corpus_path)

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

        # 3. Construire embeddings via notre module existant (avec cache)
        # Si cache_tag est fourni, on réutilise EXACTEMENT le cache correspondant (ex: 30020d...)
        # Sinon, on retombe sur l'ancien schéma de tag basé sur le corpus + nb chunks filtrés + filtres
        resolved_corpus_path = str(Path(self.corpus_path).expanduser().resolve())

        # Tag stable: évite les noms de fichiers contenant des dicts ({...})
        # et garantit la réutilisation du cache si (corpus, filtres, modèle, text_field, device) sont identiques.
        cache_tag = self.cache_tag or _stable_hash({
            "retriever": "dense",
            "corpus_parquet": resolved_corpus_path,
            "filters": self.filters,
            "text_field": self.text_field,
            "dense_model": self.embedding_model,
            "dense_device": self.device,
        })

        self._model, self._doc_embeddings = build_dense_embeddings(
            documents=self._filtered_chunks,
            model_name=self.embedding_model,
            cache_dir=_resolve_cache_dir(self.cache_dir),
            cache_tag=cache_tag,
            text_field=self.text_field,
            device=self.device,
            logger=logger,
        )

        logger.info("Embeddings prêts: shape=%s", self._doc_embeddings.shape)

        self._loaded = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Recherche dense (cosine similarity) et conversion en Documents.
        """
        self._load_corpus_and_embeddings()

        # Encoder la query
        query_embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        # Calcul similarité cosine
        scores = np.dot(self._doc_embeddings, query_embedding.T).flatten()

        # Top-k
        top_indices = np.argsort(scores)[::-1][:self.k]

        # Conversion en Documents
        docs = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self._filtered_chunks[idx]
            meta = chunk.get("meta", {})
            score = float(scores[idx])

            doc = Document(
                page_content=chunk.get(self.text_field) or chunk.get("text", ""),
                metadata={
                    # Identifiants
                    "doc_key": meta.get("doc_key"),
                    "chunk_id": chunk.get("chunk_id"),
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
                    "retriever": "dense",

                    # Audit
                    "source_path": meta.get("source_path"),
                },
            )
            docs.append(doc)

        return docs
