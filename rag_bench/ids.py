# -*- coding: utf-8 -*-
"""
Gestion des identifiants documentaires stables (doc_key).

Rôle :
- extraire des identifiants Legifrance (LEGIARTI, LEGITEXT)
- fournir une clé portable pour l'évaluation et la comparaison
"""

from __future__ import annotations

import re
from typing import Optional


_LEGI_ID_RE = re.compile(r"(LEGIARTI\d+|LEGITEXT\d+)", re.IGNORECASE)


def extract_doc_key(value: Optional[str]) -> Optional[str]:
    """
    Extrait un identifiant Legifrance stable depuis un doc_id ou un chemin.

    Priorité :
    - LEGIARTI... (article) : clé la plus utile pour évaluer un retrieval d'articles
    - sinon LEGITEXT... (texte/code) en fallback

    Args:
        value: doc_id (chemin complet ou identifiant brut)

    Returns:
        Identifiant normalisé (ex: LEGIARTI000006646450) ou None
    """
    if not value:
        return None

    v = str(value)

    # On priorise l'article si présent dans le chemin
    m_art = re.search(r"(LEGIARTI\d+)", v, flags=re.IGNORECASE)
    if m_art:
        return m_art.group(1).upper()

    # Fallback : identifiant du texte/code
    m_text = re.search(r"(LEGITEXT\d+)", v, flags=re.IGNORECASE)
    if m_text:
        return m_text.group(1).upper()

    return None

