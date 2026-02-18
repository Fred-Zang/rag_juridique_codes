#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/dense.py

Dense retrieval unifié (CPU + GPU).
Fusion de l'ancien dense.py (CPU) et dense_gpu.py (GPU optimisé).

Le paramètre 'device' contrôle l'utilisation CPU ou GPU.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Import depuis common/cache_utils
from rag_bench.common.cache_utils import (
    compute_corpus_fingerprint,
    validate_cache,
    resolve_cache_dir,
    save_cache_metadata,
    load_cache_embeddings,
    save_cache_embeddings
)


def format_time(seconds: float) -> str:
    """
    Formate une durée en secondes en format lisible.
    
    Args:
        seconds: Durée en secondes
    
    Returns:
        String formatée (ex: "2m 34s", "1h 23m 45s")
    
    Example:
        >>> print(format_time(154.5))
        '2m 34s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def compute_or_load_embeddings(
    chunks: List[Dict[str, Any]],
    text_field: str,
    cache_dir: Path,
    cache_tag: str,
    model_name: str,
    device: str = "cpu",
    batch_size: int = 32,
    logger: logging.Logger | None = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Calcule ou charge les embeddings du corpus.
    
    Stratégie :
    1. Vérifie si cache existe et est valide
    2. Si oui → charge depuis cache (rapide)
    3. Si non → calcule embeddings + sauvegarde cache
    
    Cette fonction est UNIFIÉE CPU+GPU. Le paramètre 'device' contrôle le mode.
    
    Args:
        chunks: Liste de chunks du corpus
        text_field: Nom du champ texte
        cache_dir: Dossier du cache
        cache_tag: Tag du cache
        model_name: Nom du modèle SentenceTransformers
        device: "cpu" ou "cuda" (défaut: "cpu")
        batch_size: Taille des batchs (GPU: 32-256, CPU: 8-16)
        logger: Logger optionnel
    
    Returns:
        Tuple (embeddings, timings_dict)
        - embeddings: np.ndarray (N, D)
        - timings_dict: {"load_cache": ..., "load_model": ..., "encode": ...}
    
    Example:
        >>> embeddings, timings = compute_or_load_embeddings(
        ...     chunks=chunks,
        ...     text_field="text",
        ...     cache_dir=Path(".dense_cache"),
        ...     cache_tag="abc123",
        ...     model_name="intfloat/multilingual-e5-large",
        ...     device="cuda",
        ...     batch_size=64
        ... )
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    timings = {}
    
    # Calculer empreinte corpus
    corpus_fingerprint = compute_corpus_fingerprint(chunks, text_field)
    
    # Vérifier cache existant
    t_start_cache = time.time()
    cache_valid = validate_cache(
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        expected_count=len(chunks),
        corpus_fingerprint=corpus_fingerprint,
        logger=logger
    )
    
    if cache_valid:
        # Charger depuis cache
        logger.info("📦 Chargement embeddings depuis cache...")
        embeddings = load_cache_embeddings(cache_dir, cache_tag)
        
        if embeddings is not None:
            t_cache = time.time() - t_start_cache
            timings["load_cache"] = t_cache
            
            logger.info("✅ Cache chargé en %.2fs", t_cache)
            logger.info("   Embeddings: %s (%.1f Mo)", 
                       embeddings.shape,
                       embeddings.nbytes / (1024**2))
            
            return embeddings, timings
    
    # Cache invalide ou absent → calcul embeddings
    logger.info("🔄 Calcul des embeddings (cache invalide/absent)...")
    
    # Charger modèle
    t_start_model = time.time()
    logger.info("📥 Chargement modèle : %s", model_name)
    logger.info("   Device : %s", device)
    
    model = SentenceTransformer(model_name, device=device)
    
    t_model = time.time() - t_start_model
    timings["load_model"] = t_model
    
    logger.info("✅ Modèle chargé en %.2fs", t_model)
    
    # Extraire textes
    texts = [c.get(text_field, "") for c in chunks]
    
    # Encoder (avec barre de progression)
    t_start_encode = time.time()
    
    logger.info("🔄 Encodage de %d chunks...", len(texts))
    logger.info("   Batch size : %d", batch_size)
    
    # Déterminer si on utilise GPU
    use_gpu = (device == "cuda" and torch.cuda.is_available())
    
    if use_gpu:
        # Mode GPU : conversion tensors + optimisations
        logger.info("   Mode : GPU (optimisé)")
        
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,  # ← GPU : tensors
            normalize_embeddings=True  # Normalisation pour cosine
        )
        
        # Convertir tensor → numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
    
    else:
        # Mode CPU : numpy direct
        logger.info("   Mode : CPU")
        
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,  # ← CPU : numpy direct
            normalize_embeddings=True
        )
    
    t_encode = time.time() - t_start_encode
    timings["encode"] = t_encode
    
    # Stats encodage
    chunks_per_sec = len(texts) / t_encode if t_encode > 0 else 0
    
    logger.info("✅ Encodage terminé en %s", format_time(t_encode))
    logger.info("   Vitesse : %.1f chunks/s", chunks_per_sec)
    logger.info("   Embeddings : %s (%.1f Mo)",
               embeddings.shape,
               embeddings.nbytes / (1024**2))
    
    # Sauvegarder cache
    logger.info("💾 Sauvegarde cache...")
    
    save_cache_embeddings(cache_dir, cache_tag, embeddings)
    
    save_cache_metadata(
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        chunk_count=len(chunks),
        corpus_fingerprint=corpus_fingerprint,
        model_name=model_name,
        embedding_dim=embeddings.shape[1],
        device=device
    )
    
    logger.info("✅ Cache sauvegardé : %s/%s.*", cache_dir, cache_tag)
    
    return embeddings, timings


def dense_search(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int = 10,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recherche top-k par similarité cosinus.
    
    Version unifiée CPU+GPU. Le paramètre 'device' contrôle le mode.
    
    Args:
        query_embedding: Embedding query (D,)
        corpus_embeddings: Embeddings corpus (N, D)
        k: Nombre de résultats
        device: "cpu" ou "cuda"
    
    Returns:
        Tuple (indices, scores)
        - indices: np.ndarray (k,) - Indices des top-k
        - scores: np.ndarray (k,) - Scores de similarité
    
    Example:
        >>> indices, scores = dense_search(
        ...     query_embedding=q_emb,
        ...     corpus_embeddings=corpus_emb,
        ...     k=10,
        ...     device="cuda"
        ... )
    """
    use_gpu = (device == "cuda" and torch.cuda.is_available())
    
    if use_gpu:
        # Mode GPU : calcul sur tensors
        q_tensor = torch.from_numpy(query_embedding).to(device)
        c_tensor = torch.from_numpy(corpus_embeddings).to(device)
        
        # Similarité cosinus (dot product si embeddings normalisés)
        scores = torch.matmul(c_tensor, q_tensor)
        
        # Top-k
        top_scores, top_indices = torch.topk(scores, k=min(k, len(scores)))
        
        # Convertir en numpy
        indices = top_indices.cpu().numpy()
        scores_array = top_scores.cpu().numpy()
    
    else:
        # Mode CPU : calcul numpy
        scores = np.dot(corpus_embeddings, query_embedding)
        
        # Top-k (argsort décroissant)
        top_indices = np.argsort(scores)[::-1][:k]
        indices = top_indices
        scores_array = scores[top_indices]
    
    return indices, scores_array


def benchmark_dense(
    chunks: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
    text_field: str,
    cache_dir: Path,
    cache_tag: str,
    model_name: str,
    device: str = "cpu",
    batch_size: int = 32,
    k: int = 10,
    logger: logging.Logger | None = None
) -> List[Dict[str, Any]]:
    """
    Benchmark dense retrieval complet.
    
    Version unifiée CPU+GPU.
    
    Args:
        chunks: Corpus (liste de dicts)
        queries: Queries (liste de dicts avec 'id' et 'text')
        text_field: Nom du champ texte dans chunks
        cache_dir: Dossier cache
        cache_tag: Tag cache
        model_name: Modèle embedding
        device: "cpu" ou "cuda"
        batch_size: Taille batch
        k: Top-k résultats par query
        logger: Logger optionnel
    
    Returns:
        Liste de résultats (dicts) avec :
        - query_id, doc_key, score, rank, query_text, chunk_text, ...
    
    Example:
        >>> results = benchmark_dense(
        ...     chunks=chunks,
        ...     queries=queries,
        ...     text_field="text",
        ...     cache_dir=Path(".dense_cache"),
        ...     cache_tag="abc123",
        ...     model_name="intfloat/multilingual-e5-large",
        ...     device="cuda",
        ...     k=10
        ... )
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 1. Charger/calculer embeddings corpus
    logger.info("=" * 80)
    logger.info("DENSE RETRIEVAL - CORPUS EMBEDDINGS")
    logger.info("=" * 80)
    
    corpus_embeddings, timings_corpus = compute_or_load_embeddings(
        chunks=chunks,
        text_field=text_field,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        logger=logger
    )
    
    # 2. Charger modèle pour queries
    logger.info("=" * 80)
    logger.info("DENSE RETRIEVAL - QUERY EMBEDDINGS")
    logger.info("=" * 80)
    
    logger.info("📥 Chargement modèle pour queries...")
    model = SentenceTransformer(model_name, device=device)
    
    # 3. Encoder queries
    query_texts = [q["text"] for q in queries]
    
    logger.info("🔄 Encodage de %d queries...", len(query_texts))
    
    use_gpu = (device == "cuda" and torch.cuda.is_available())
    
    if use_gpu:
        query_embeddings = model.encode(
            query_texts,
            batch_size=8,  # Queries : petit batch
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.cpu().numpy()
    else:
        query_embeddings = model.encode(
            query_texts,
            batch_size=8,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    
    logger.info("✅ Queries encodées : %s", query_embeddings.shape)
    
    # 4. Recherche top-k pour chaque query
    logger.info("=" * 80)
    logger.info("DENSE RETRIEVAL - RECHERCHE TOP-K")
    logger.info("=" * 80)
    
    all_results = []
    
    t_start_search = time.time()
    
    for idx, (query, q_emb) in enumerate(zip(queries, query_embeddings)):
        query_id = query["id"]
        query_text = query["text"]
        
        logger.info("Query %d/%d : %s (id=%s)", 
                   idx+1, len(queries), query_text[:80], query_id)
        
        # Recherche top-k
        top_indices, top_scores = dense_search(
            query_embedding=q_emb,
            corpus_embeddings=corpus_embeddings,
            k=k,
            device=device
        )
        
        # Construire résultats
        for rank, (chunk_idx, score) in enumerate(zip(top_indices, top_scores), start=1):
            chunk = chunks[chunk_idx]
            
            result = {
                "query_id": query_id,
                "query_text": query_text,
                "doc_key": chunk.get("doc_key", ""),
                "score": float(score),
                "rank": rank,
                "chunk_text": chunk.get(text_field, "")[:500],  # Preview
                # Métadonnées additionnelles si présentes
                "source_path": chunk.get("source_path", ""),
                "doc_type": chunk.get("doc_type", ""),
                "status": chunk.get("status", ""),
            }
            
            all_results.append(result)
        
        logger.info("  → Top-1 : %s (score=%.4f)", 
                   all_results[-k]["doc_key"], all_results[-k]["score"])
    
    t_search = time.time() - t_start_search
    
    logger.info("=" * 80)
    logger.info("DENSE RETRIEVAL - TERMINÉ")
    logger.info("=" * 80)
    logger.info("✅ Recherche terminée en %s", format_time(t_search))
    logger.info("   Vitesse : %.2f queries/s", len(queries) / t_search if t_search > 0 else 0)
    logger.info("   Total résultats : %d", len(all_results))
    
    return all_results

# ============================================================================
# WRAPPERS DE COMPATIBILITÉ (pour benchmark_hybrid_rrf.py et autres)
# ============================================================================

def build_dense_embeddings(
    documents: List[Dict[str, Any]],
    model_name: str,
    cache_dir: str | Path,
    cache_tag: str,
    text_field: str = "text",
    device: str = "cpu",
    batch_size: int = 32,
    logger: logging.Logger | None = None
) -> Tuple[SentenceTransformer, np.ndarray]:
    """
    Wrapper de compatibilité pour benchmark_hybrid_rrf.py.
    
    Construit ou charge les embeddings + retourne le modèle.
    
    Returns:
        Tuple (model, embeddings)
        - model: SentenceTransformer (pour encoder queries)
        - embeddings: np.ndarray (N, D)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Convertir cache_dir en Path si nécessaire
    cache_dir_path = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
    
    # Calculer ou charger embeddings
    embeddings, _ = compute_or_load_embeddings(
        chunks=documents,
        text_field=text_field,
        cache_dir=cache_dir_path,
        cache_tag=cache_tag,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        logger=logger
    )
    
    # Charger modèle pour queries
    model = SentenceTransformer(model_name, device=device)
    
    return model, embeddings


def dense_topk(
    query: str,
    model: SentenceTransformer,
    doc_embeddings: np.ndarray,
    k: int = 10,
    device: str = "cpu"
) -> List[Tuple[int, float]]:
    """
    Wrapper de compatibilité pour benchmark_hybrid_rrf.py.
    
    Encode une query et retourne top-k documents.
    
    Args:
        query: Texte de la query
        model: Modèle SentenceTransformer (déjà chargé)
        doc_embeddings: Embeddings corpus (N, D)
        k: Nombre de résultats
        device: "cpu" ou "cuda"
    
    Returns:
        Liste de tuples (index, score) triée par score décroissant
    """
    # Encoder query
    use_gpu = (device == "cuda" and torch.cuda.is_available())
    
    if use_gpu:
        query_emb = model.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True
        )[0]
        if isinstance(query_emb, torch.Tensor):
            query_emb = query_emb.cpu().numpy()
    else:
        query_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
    
    # Recherche top-k
    indices, scores = dense_search(
        query_embedding=query_emb,
        corpus_embeddings=doc_embeddings,
        k=k,
        device=device
    )
    
    # Retourner liste de tuples (index, score)
    return list(zip(indices.tolist(), scores.tolist()))


def dense_retrieval(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    k: int = 10,
    text_field: str = "text",
    device: str = "cpu"
) -> List[Tuple[int, float]]:
    """
    Wrapper de compatibilité pour benchmark_dense.py.
    
    Encode une query et retourne top-k.
    
    Returns:
        Liste de tuples (chunk_index, score)
    """
    return dense_topk(
        query=query,
        model=model,
        doc_embeddings=embeddings,
        k=k,
        device=device
    )