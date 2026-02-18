# -*- coding: utf-8 -*-
"""
Benchmark dense multi-queries.

Rôle :
- calculer/recharger les embeddings du corpus filtré
- exécuter plusieurs requêtes
- sauvegarder un top-k par requête au format JSONL comparable à BM25

Format de sortie :
- dense_results.jsonl avec au minimum : query_id, rank, score, doc_key (ou doc_id)

Benchmark dense retrieval (embeddings + cosine similarity).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from rag_bench.core.dense import compute_or_load_embeddings, dense_retrieval  # ✅ Ajouté
from rag_bench.common.cache_utils import resolve_cache_dir
from rag_bench.ids import extract_doc_key

#from rag_bench.filtering import filter_temporal


def run_dense_benchmark(
    *,
    chunks: List[Dict[str, Any]],
    queries: List[Dict[str, str]],
    run_dir: str,
    k: int,
    text_field: str,
    embedding_model: str,
    cache_dir: Path,
    cache_tag: str,
    device: str,
    batch_size: int = 32,   # pourquoi ici 32 alors que 256 dans le yaml ???
    show_verbatim: bool = False,
    preview_chars: int = 450,
    logger: logging.Logger,
    as_of: Optional[str] = None,
) -> None:
    """
    Exécute un benchmark dense multi-requêtes.
    
    Args:
        chunks: Corpus filtré
        queries: Liste de requêtes
        run_dir: Dossier de sortie
        k: Nombre de résultats
        text_field: Nom du champ texte
        embedding_model: Modèle d'embeddings
        cache_dir: Dossier du cache
        cache_tag: Tag du cache
        device: cpu ou cuda
        show_verbatim: Afficher texte complet
        preview_chars: Nombre de caractères à afficher
        logger: Logger
        as_of: Date de référence (optionnel)
    """
    logger.info("=" * 80)
    logger.info("BENCHMARK DENSE")
    logger.info("=" * 80)
    logger.info("Modèle: %s", embedding_model)
    logger.info("Device: %s", device)
    logger.info("Cache: %s", cache_dir)
    logger.info("Cache tag: %s", cache_tag)
    logger.info("Nombre de requêtes: %d", len(queries))
    logger.info("Top-k: %d", k)
    
    # Calculer ou charger les embeddings
    logger.info("Calcul/chargement des embeddings...")
    embeddings, timings = compute_or_load_embeddings(
        chunks=chunks,
        text_field=text_field,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        model_name=embedding_model,
        device=device,
        batch_size=batch_size,
        logger=logger
    )
        
    logger.info("✅ Embeddings prêts: shape=%s", embeddings.shape)
    
    # Charger le modèle (nécessaire pour encoder les requêtes)
    logger.info("Chargement du modèle pour requêtes...")
    model = SentenceTransformer(embedding_model, device=device)
    logger.info("✅ Modèle chargé")
    
    # Exécuter les requêtes
    results_file = os.path.join(run_dir, "dense_results.jsonl")
    logger.info("Fichier de sortie: %s", results_file)
    
    with open(results_file, "w", encoding="utf-8") as f_out:
        for idx, q in enumerate(queries, 1):
            query_id = q.get("id", f"q{idx}")
            query_text = q.get("text", "")
            
            if not query_text:
                logger.warning("⚠️ Requête %s vide, ignorée", query_id)
                continue
            
            logger.info("Requête %d/%d: %s (id=%s)", idx, len(queries), query_text[:80], query_id)
            
            # Retrieval dense
            topk_results = dense_retrieval(
                query=query_text,
                chunks=chunks,
                embeddings=embeddings,
                model=model,
                k=k,
                text_field=text_field,
                device=device,
            )
            
            # Construire les résultats
            first_result = None
            # Écrire UN RECORD PAR HIT
            for rank, (chunk_idx, score) in enumerate(topk_results, 1):
                chunk = chunks[chunk_idx]
                doc_key = extract_doc_key(chunk)
                meta = chunk.get("meta") or {}
                
                record = {
                    "retriever": "dense",
                    "query_id": query_id,
                    "query_text": query_text,
                    "rank": rank,
                    "score": float(score),
                    "doc_key": doc_key,
                    "doc_id": chunk.get("doc_id"),
                    "doc_type": chunk.get("doc_type"),
                    "titre": meta.get("titre"),
                    "chunk_index": chunk_idx,
                    "text_preview": (chunk.get(text_field) or "")[:preview_chars],
                    "as_of": as_of,
                    "date_debut": meta.get("date_debut"),
                    "date_fin": meta.get("date_fin"),
                    "etat": meta.get("etat"),
                    "cache_tag": cache_tag,
                    "timestamp": datetime.now().isoformat(),
                }
                
                if show_verbatim:
                    record["text"] = chunk.get(text_field, "")
                
                if rank == 1:
                    first_result = record  # ✅ Sauvegarder le premier
                
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Afficher Top-1
            if first_result:
                logger.info("  → Top-1: doc_key=%s, score=%.4f", 
                        first_result["doc_key"], first_result["score"])
    
    logger.info("=" * 80)
    logger.info("✅ Benchmark dense terminé")
    logger.info("Résultats: %s", results_file)
    logger.info("=" * 80)