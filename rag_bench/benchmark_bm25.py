# -*- coding: utf-8 -*-
"""
Benchmark BM25 multi-queries.

Rôle :
- exécuter plusieurs requêtes
- produire un top-k par requête
- sauvegarder les résultats pour analyse ultérieure

Remarque temporelle :
- le filtrage `as_of` est appliqué en amont (dans filtering.py).
- on écrit néanmoins `as_of`, `valid_from`, `valid_to`, `status` dans les résultats (V3)
  pour rendre l’audit juridique explicite dans les artefacts.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from rag_bench.bm25 import build_bm25_index
from rag_bench.ids import extract_doc_key


def run_bm25_benchmark(
    chunks: List[dict],
    queries: List[dict],
    run_dir: str,
    k: int = 10,
    text_field: str = "text",
    as_of: Optional[str] = None,
    cache_tag: Optional[str] = None,  # ✅ AJOUTÉ (None pour BM25)
) -> None:
    """
    Lance un benchmark BM25 multi-queries et sauvegarde les résultats.

    Args:
        chunks: chunks filtrés
        queries: liste de dicts {id, text}
        run_dir: dossier du run
        k: top-k
        text_field: champ texte à indexer
        as_of: date de référence ISO "YYYY-MM-DD" (écrite dans les résultats pour audit)
    """
    index = build_bm25_index(chunks, text_field=text_field)

    out_path = os.path.join(run_dir, "bm25_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for q in queries:
            results = index.search(q["text"], k=k)
            for rank, (chunk, score) in enumerate(results, start=1):
                meta = chunk.get("meta") or {}
                # doc_key doit venir du chunk complet (meta.doc_key), pas de doc_id
                doc_key = extract_doc_key(chunk)
                if not doc_key:
                    continue

                doc_id = chunk.get("doc_id")

                record = {
                    "retriever": "bm25",
                    "query_id": q["id"],
                    "query_text": q["text"],
                    "rank": rank,
                    "score": float(score),
                    "doc_key": doc_key,
                    "doc_id": doc_id,
                    "doc_type": chunk.get("doc_type"),
                    "titre": meta.get("titre"),
                    "text_preview": (chunk.get(text_field) or "")[:400],
                    # Champs temporels V3 : lecture depuis root chunk avec fallback meta
                    "as_of": as_of,
                    "valid_from": chunk.get("valid_from") or meta.get("valid_from") or meta.get("date_debut"),
                    "valid_to": chunk.get("valid_to") or meta.get("valid_to") or meta.get("date_fin"),
                    "status": chunk.get("status") or meta.get("status") or meta.get("etat"),
                    "cache_tag": None,  # ✅ BM25 n'utilise pas de cache, mais on uniformise
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
