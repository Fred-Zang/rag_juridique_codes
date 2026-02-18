# -*- coding: utf-8 -*-
"""
BM25 (retrieval lexical) pour un corpus chunké.

Ce module encapsule :
- la tokenisation (simple, robuste, reproductible)
- la construction de l'index BM25
- la recherche top-k

Objectif :
- fournir un baseline solide et explicable
- préparer la comparaison future avec dense/hybride
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from rank_bm25 import BM25Okapi


_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    """
    Tokenise un texte de manière simple et stable.

    - conserve lettres accentuées + chiffres
    - met en minuscules
    - évite une dépendance lourde (pas de spaCy ici)

    Args:
        text: texte brut

    Returns:
        liste de tokens
    """
    if not text:
        return []
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


@dataclass
class BM25Index:
    """
    Index BM25 + mapping vers les chunks.

    Attributes:
        bm25: objet BM25Okapi (rank_bm25)
        chunks: liste des chunks d'origine (dict)
        tokenized_corpus: corpus tokenisé (liste de liste de tokens)
        text_field: champ utilisé pour indexer le texte (souvent 'text')
    """
    bm25: BM25Okapi
    chunks: List[Dict[str, Any]]
    tokenized_corpus: List[List[str]]
    text_field: str = "text"

    def search(self, query: str, k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Recherche top-k chunks via BM25.

        Args:
            query: question utilisateur
            k: nombre de résultats

        Returns:
            liste de tuples (chunk, score) triée par score décroissant
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)
        # On récupère les indices des meilleurs scores
        best_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in best_idx]
    def search_indices(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Recherche top-k mais retourne des indices dans la liste self.chunks.

        Pourquoi c'est utile :
        - l’hybride RRF fusionne des rankings d’indices (plus stable que des dicts)
        - on évite les ambiguïtés quand deux chunks ont des contenus proches

        Returns:
            liste de tuples (index_chunk, score) triée par score décroissant
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)
        best_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(int(i), float(scores[i])) for i in best_idx]


# -----------------------------------------------------------------------------------------------------

def build_bm25_index(chunks: Sequence[Dict[str, Any]], text_field: str = "text") -> BM25Index:
    """
    Construit un index BM25 à partir des chunks.

    Args:
        chunks: séquence de chunks (dict), chaque chunk doit contenir `text_field`
        text_field: champ contenant le texte à indexer

    Returns:
        BM25Index prêt pour la recherche
    """
    chunk_list = list(chunks)
    tokenized = [tokenize(str(c.get(text_field, ""))) for c in chunk_list]
    bm25 = BM25Okapi(tokenized)
    return BM25Index(bm25=bm25, chunks=chunk_list, tokenized_corpus=tokenized, text_field=text_field)
