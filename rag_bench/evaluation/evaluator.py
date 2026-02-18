#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/evaluator.py

Évaluation retrieval générique (BM25, Dense, Hybrid, Elasticsearch).
Calcul des métriques IR : Recall@k, MRR, nDCG@k.

Anciennement : evaluate_bm25.py (nom trompeur car évalue TOUS les retrievers)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import math

import yaml

from rag_bench.metrics import recall_at_k, mean_reciprocal_rank, ndcg_at_k


def load_qrels(qrels_file: str, logger: logging.Logger | None = None) -> Dict[str, Dict[str, float]]:
    """
    Charge les qrels depuis un fichier YAML.
    
    Format attendu :
```yaml
    q1:
      LEGIARTI000123: 1
      LEGIARTI000456: 0.5
    q2:
      LEGIARTI000789: 1
```
    
    Args:
        qrels_file: Chemin vers le fichier qrels YAML
        logger: Logger optionnel
    
    Returns:
        Dict {query_id: {doc_key: score}}
    
    Raises:
        FileNotFoundError: Si fichier qrels introuvable
        ValueError: Si format YAML invalide
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    qrels_path = Path(qrels_file)
    
    if not qrels_path.exists():
        raise FileNotFoundError(f"Fichier qrels introuvable : {qrels_file}")
    
    logger.info("📋 Chargement qrels : %s", qrels_file)
    
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = yaml.safe_load(f)
    
    # Valider format
    if not isinstance(qrels, dict):
        raise ValueError(f"Qrels invalides (doit être un dict) : {qrels_file}")
    
    # Compter queries et docs pertinents
    num_queries = len(qrels)
    num_relevant = sum(len(docs) for docs in qrels.values())
    
    logger.info("✅ Qrels chargés : %d queries, %d docs pertinents", num_queries, num_relevant)
    
    return qrels


def load_results(results_file: str, logger: logging.Logger | None = None) -> List[Dict[str, Any]]:
    """
    Charge les résultats retrieval depuis un fichier JSONL.
    
    Format attendu (1 ligne par hit) :
```json
    {"query_id": "q1", "doc_key": "LEGIARTI000123", "score": 0.95, "rank": 1}
    {"query_id": "q1", "doc_key": "LEGIARTI000456", "score": 0.87, "rank": 2}
    ...
```
    
    Args:
        results_file: Chemin vers *_results.jsonl
        logger: Logger optionnel
    
    Returns:
        Liste de dicts (résultats)
    
    Raises:
        FileNotFoundError: Si fichier résultats introuvable
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    results_path = Path(results_file)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Fichier résultats introuvable : {results_file}")
    
    logger.info("📄 Chargement résultats : %s", results_file)
    
    results = []
    
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line.strip())
            results.append(result)
    
    logger.info("✅ Résultats chargés : %d hits", len(results))
    
    return results


def evaluate_retrieval(
    results_file: str,
    qrels_file: str,
    metrics: List[str],
    k_values: List[int],
    output_file: str,
    logger: logging.Logger | None = None
) -> Dict[str, Any]:
    """
    Évalue un retriever (BM25, Dense, Hybrid, ES) avec des métriques IR.
    
    Métriques supportées :
    - recall : Recall@k (proportion de docs pertinents retrouvés)
    - mrr : Mean Reciprocal Rank (rang du 1er pertinent)
    - ndcg : Normalized Discounted Cumulative Gain (qualité ranking)
    
    Args:
        results_file: Chemin vers *_results.jsonl
        qrels_file: Chemin vers qrels YAML
        metrics: Liste de métriques (ex: ["recall", "mrr", "ndcg"])
        k_values: Liste de k (ex: [1, 3, 5, 10, 20])
        output_file: Chemin vers metrics_*.json (sortie)
        logger: Logger optionnel
    
    Returns:
        Dict contenant toutes les métriques calculées
    
    Example:
        >>> metrics_dict = evaluate_retrieval(
        ...     results_file="runs/20260125_test/bm25_results.jsonl",
        ...     qrels_file="configs/qrels_v4.yaml",
        ...     metrics=["recall", "mrr", "ndcg"],
        ...     k_values=[1, 3, 5, 10],
        ...     output_file="runs/20260125_test/metrics_bm25.json"
        ... )
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("ÉVALUATION RETRIEVAL")
    logger.info("=" * 80)
    
    # 1. Charger qrels
    qrels = load_qrels(qrels_file, logger)
    
    # 2. Charger résultats
    results = load_results(results_file, logger)
    
    # 3. Organiser résultats par query
    results_by_query = {}
    
    for result in results:
        query_id = result["query_id"]
        doc_key = result["doc_key"]
        
        if query_id not in results_by_query:
            results_by_query[query_id] = []
        
        results_by_query[query_id].append(doc_key)
    
    # 4. Calculer métriques
    avg_metrics = {}  # Métriques moyennes
    per_query_metrics = {}  # Métriques par query

    # Initialiser per_query
    for query_id in qrels.keys():
        per_query_metrics[query_id] = {}

    for metric_name in metrics:
        metric_name_lower = metric_name.lower()
        
        if metric_name_lower == "recall":
            # Recall@k pour chaque k
            for k in k_values:
                # Moyenne globale
                avg_score = recall_at_k(
                    results_by_query=results_by_query,
                    qrels=qrels,
                    k=k
                )
                avg_metrics[f"Recall@{k}"] = avg_score
                logger.info(f"  Recall@{k} : {avg_score:.4f}")
                
                # Par query
                for query_id, relevant_docs in qrels.items():
                    if query_id not in results_by_query:
                        per_query_metrics[query_id][f"Recall@{k}"] = 0.0
                        continue
                    
                    retrieved = results_by_query[query_id][:k]
                    relevant_set = set(relevant_docs.keys())
                    hits = sum(1 for doc_key in retrieved if doc_key in relevant_set)
                    
                    if len(relevant_set) > 0:
                        per_query_metrics[query_id][f"Recall@{k}"] = hits / len(relevant_set)
                    else:
                        per_query_metrics[query_id][f"Recall@{k}"] = 0.0
        
        elif metric_name_lower == "mrr":
            # MRR moyenne
            avg_score = mean_reciprocal_rank(
                results_by_query=results_by_query,
                qrels=qrels
            )
            avg_metrics["MRR"] = avg_score
            logger.info(f"  MRR : {avg_score:.4f}")
            
            # MRR par query
            for query_id, relevant_docs in qrels.items():
                if query_id not in results_by_query:
                    per_query_metrics[query_id]["MRR"] = 0.0
                    continue
                
                retrieved = results_by_query[query_id]
                relevant_set = set(relevant_docs.keys())
                
                for rank, doc_key in enumerate(retrieved, start=1):
                    if doc_key in relevant_set:
                        per_query_metrics[query_id]["MRR"] = 1.0 / rank
                        break
                else:
                    per_query_metrics[query_id]["MRR"] = 0.0
        
        elif metric_name_lower == "ndcg":
            # nDCG@k pour chaque k
            for k in k_values:
                # Moyenne globale
                avg_score = ndcg_at_k(
                    results_by_query=results_by_query,
                    qrels=qrels,
                    k=k
                )
                avg_metrics[f"nDCG@{k}"] = avg_score
                logger.info(f"  nDCG@{k} : {avg_score:.4f}")
                
                # Par query
                for query_id, relevant_docs in qrels.items():
                    if query_id not in results_by_query:
                        per_query_metrics[query_id][f"nDCG@{k}"] = 0.0
                        continue
                    
                    retrieved = results_by_query[query_id][:k]
                    
                    # DCG
                    dcg = 0.0
                    for rank, doc_key in enumerate(retrieved, start=1):
                        if doc_key in relevant_docs:
                            gain = relevant_docs[doc_key]
                            dcg += gain / math.log2(rank + 1)
                    
                    # IDCG
                    ideal_scores = sorted(relevant_docs.values(), reverse=True)[:k]
                    idcg = sum(score / math.log2(rank + 1) for rank, score in enumerate(ideal_scores, start=1))
                    
                    if idcg > 0:
                        per_query_metrics[query_id][f"nDCG@{k}"] = dcg / idcg
                    else:
                        per_query_metrics[query_id][f"nDCG@{k}"] = 0.0

    # Ajouter num_relevant et num_retrieved par query
    for query_id in qrels.keys():
        per_query_metrics[query_id]["num_relevant"] = len(qrels[query_id])
        per_query_metrics[query_id]["num_retrieved"] = len(results_by_query.get(query_id, []))

    # Ajouter num_queries à average
    avg_metrics["num_queries"] = len(qrels)

    # 5. Construire rapport complet
    all_metrics = {
        "average": avg_metrics,
        "per_query": per_query_metrics
    }

    # 6. Sauvegarder métriques
    logger.info("💾 Sauvegarde métriques : %s", output_file)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    logger.info("✅ Métriques sauvegardées")
    logger.info("=" * 80)

    return all_metrics