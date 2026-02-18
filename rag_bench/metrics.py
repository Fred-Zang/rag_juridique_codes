# -*- coding: utf-8 -*-
"""
Métriques standard IR :
- Recall@k
- MRR (Mean Reciprocal Rank)
- nDCG@k
"""

from __future__ import annotations

import math
from typing import Dict, List


def recall_at_k(
    results_by_query: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, float]],
    k: int
) -> float:
    """
    Calcule Recall@k moyen sur toutes les queries.
    
    Args:
        results_by_query: {query_id: [doc_key1, doc_key2, ...]}
        qrels: {query_id: {doc_key: score}}
        k: Profondeur (top-k)
    
    Returns:
        Recall@k moyen (0.0 à 1.0)
    """
    scores = []
    
    for query_id, relevant_docs in qrels.items():
        if query_id not in results_by_query:
            scores.append(0.0)
            continue
        
        retrieved = results_by_query[query_id][:k]
        relevant_set = set(relevant_docs.keys())
        
        hits = sum(1 for doc_key in retrieved if doc_key in relevant_set)
        
        if len(relevant_set) > 0:
            scores.append(hits / len(relevant_set))
        else:
            scores.append(0.0)
    
    return sum(scores) / len(scores) if scores else 0.0


def mean_reciprocal_rank(
    results_by_query: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, float]]
) -> float:
    """
    Calcule MRR (Mean Reciprocal Rank) moyen.
    
    Args:
        results_by_query: {query_id: [doc_key1, doc_key2, ...]}
        qrels: {query_id: {doc_key: score}}
    
    Returns:
        MRR moyen (0.0 à 1.0)
    """
    scores = []
    
    for query_id, relevant_docs in qrels.items():
        if query_id not in results_by_query:
            scores.append(0.0)
            continue
        
        retrieved = results_by_query[query_id]
        relevant_set = set(relevant_docs.keys())
        
        # Trouver rang du 1er doc pertinent
        for rank, doc_key in enumerate(retrieved, start=1):
            if doc_key in relevant_set:
                scores.append(1.0 / rank)
                break
        else:
            scores.append(0.0)
    
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(
    results_by_query: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, float]],
    k: int
) -> float:
    """
    Calcule nDCG@k moyen.
    
    Args:
        results_by_query: {query_id: [doc_key1, doc_key2, ...]}
        qrels: {query_id: {doc_key: score}}
        k: Profondeur (top-k)
    
    Returns:
        nDCG@k moyen (0.0 à 1.0)
    """
    scores = []
    
    for query_id, relevant_docs in qrels.items():
        if query_id not in results_by_query:
            scores.append(0.0)
            continue
        
        retrieved = results_by_query[query_id][:k]
        
        # DCG
        dcg = 0.0
        for rank, doc_key in enumerate(retrieved, start=1):
            if doc_key in relevant_docs:
                # Gain = score du qrel (1 ou 0.5 généralement)
                gain = relevant_docs[doc_key]
                dcg += gain / math.log2(rank + 1)
        
        # IDCG (classement idéal)
        ideal_scores = sorted(relevant_docs.values(), reverse=True)[:k]
        idcg = sum(score / math.log2(rank + 1) for rank, score in enumerate(ideal_scores, start=1))
        
        if idcg > 0:
            scores.append(dcg / idcg)
        else:
            scores.append(0.0)
    
    return sum(scores) / len(scores) if scores else 0.0