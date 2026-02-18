# -*- coding: utf-8 -*-
"""
Benchmark hybride BM25 + Dense via RRF (Reciprocal Rank Fusion).

Rôle :
- construire l'index BM25 (lexical)
- construire/recharger les embeddings (dense)
- fusionner deux rankings via RRF
- sauvegarder un top-k par requête au format JSONL comparable à BM25

Pourquoi RRF ?
- RRF fusionne des rangs (pas des scores bruts), donc il est stable quand
  les échelles de scores diffèrent fortement (BM25 vs cos-sim).
"""

from __future__ import annotations
from typing import Dict, List, Optional

import json
import os
from typing import Dict, List

# Imports BM25
from rag_bench.bm25 import build_bm25_index

# Imports Dense (wrappers rétro-compatibles utilisés par l'hybride RRF)
# build_dense_embeddings : calcule/charge les embeddings + retourne aussi le modèle pour encoder les requêtes
# dense_topk            : renvoie un top-k (idx, score) sur les embeddings déjà prêts
from rag_bench.core.dense import build_dense_embeddings, dense_topk

from rag_bench.ids import extract_doc_key



def _rrf_fuse(
    *,
    bm25_ranked: List[int],
    dense_ranked: List[int],
    rrf_k: int,
) -> List[tuple[int, float]]:
    """
    Fusionne deux rankings (listes d'indices) via RRF.

    Score RRF standard :
        score(doc) = sum_{retrievers} 1 / (rrf_k + rank(doc))

    Args:
        bm25_ranked: indices documents ordonnés (rang 1..n)
        dense_ranked: indices documents ordonnés (rang 1..n)
        rrf_k: constante qui amortit le poids du rang (valeur classique ~60)

    Returns:
        Liste (idx_doc, score_rrf) triée décroissante.
    """
    fused: Dict[int, float] = {}

    for rank, idx in enumerate(bm25_ranked, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)

    for rank, idx in enumerate(dense_ranked, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def run_hybrid_rrf_benchmark(
    *,
    chunks: List[dict],
    queries: List[dict],
    run_dir: str,
    k: int,
    text_field: str,
    embedding_model: str,
    cache_dir: str,
    cache_tag: str,
    device: str,
    bm25_shortlist: int,
    rrf_k: int,
    show_verbatim: bool,
    preview_chars: int,
    logger,
    as_of: Optional[str] = None,
) -> None:
    """
    Lance un benchmark hybride (BM25 + Dense) multi-queries et sauvegarde les résultats.

    Args:
        chunks: chunks filtrés
        queries: liste de dicts {id, text}
        run_dir: dossier du run
        k: top-k final
        text_field: champ texte BM25 + embeddings
        embedding_model: nom SentenceTransformers
        cache_dir: dossier cache embeddings
        cache_tag: tag stable (corpus + filtres)
        bm25_shortlist: profondeur interne (top-N BM25 et top-N dense) avant fusion
        rrf_k: constante RRF
        show_verbatim: si True, affiche une trace lisible des top-k (audit)
        preview_chars: nombre de caractères affichés en preview
        logger: logger configuré au niveau du run
    """
    # Index BM25 construit sur le corpus filtré (même ordre que chunks)
    bm25_index = build_bm25_index(chunks, text_field=text_field)

    # Embeddings dense construits sur le corpus filtré (même ordre que chunks)
    model, doc_emb = build_dense_embeddings(
        documents=chunks,
        model_name=embedding_model,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        text_field=text_field,
        device=device,
        logger=logger,
    )

    out_path = os.path.join(run_dir, "hybrid_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for q in queries:
            qid = q["id"]
            qtext = q["text"]

            # Shortlists : on travaille sur des rangs plutôt que des scores bruts
            bm25_pairs = bm25_index.search_indices(qtext, k=bm25_shortlist)
            bm25_ranked = [idx for idx, _score in bm25_pairs]

            dense_pairs = dense_topk(query=qtext, model=model, doc_embeddings=doc_emb, k=bm25_shortlist)
            dense_ranked = [idx for idx, _score in dense_pairs]

            # Fusion RRF sur un classement large, puis déduplication par doc_key avant tronquage à k.
            fused_all = _rrf_fuse(
                bm25_ranked=bm25_ranked,
                dense_ranked=dense_ranked,
                rrf_k=rrf_k,
            )

            seen_doc_keys = set()
            fused = []

            for idx, score in fused_all:
                chunk = chunks[idx]

                # doc_key doit venir du chunk (meta.doc_key idéalement), jamais du doc_id (qui est un chemin).
                doc_key = extract_doc_key(chunk)

                # Si on ne peut pas identifier le document, on ne peut pas dédupliquer proprement.
                if not doc_key:
                    continue

                # Anti-doublons intra-requête.
                if doc_key in seen_doc_keys:
                    continue

                seen_doc_keys.add(doc_key)
                fused.append((idx, score))

                if len(fused) >= k:
                    break

            if show_verbatim:
                logger.info(
                    "HybridRRF dedup | query_id=%s | uniques=%d (raw=%d)",
                    qid,
                    len(fused),
                    len(fused_all),
                )

            for rank, (idx, score) in enumerate(fused, start=1):
                chunk = chunks[idx]
                meta = chunk.get("meta") or {}

                # doc_key stable reconstruit pour l'évaluation / audit car on contruit un record par hit
                doc_key = extract_doc_key(chunk)

                # doc_id sert à l'audit (souvent source_path via run.py), ne doit pas être remplacé par doc_key
                doc_id = chunk.get("doc_id")

                record = {
                    "retriever": "hybrid_rrf",
                    "query_id": qid,
                    "query_text": qtext,
                    "rank": rank,
                    "score": float(score),
                    "doc_key": doc_key,
                    "doc_id": doc_id,
                    "doc_type": chunk.get("doc_type"),
                    "titre": meta.get("titre"),
                    "text_preview": (chunk.get(text_field) or "")[:preview_chars],
                    # Champs temporels : audit juridique
                    "as_of": as_of,
                    "date_debut": meta.get("date_debut"),
                    "date_fin": meta.get("date_fin"),
                    "etat": meta.get("etat"),
                    "cache_tag": cache_tag,  # ✅ AJOUT
                }

                # Écriture JSONL : une ligne par hit
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if show_verbatim:
                    logger.info(
                        "  #%d score=%.6f doc_key=%s as_of=%s valid=[%s,%s[ doc_id=%s",
                        rank,
                        score,
                        record["doc_key"],
                        as_of,
                        record["date_debut"],
                        record["date_fin"],
                        doc_id,
                    )

    logger.info("Hybrid results écrits : %s", out_path)
