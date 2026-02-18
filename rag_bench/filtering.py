from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional


def _meta(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Retourne meta si présent, sinon dict vide (compat JSONL/Parquet)."""
    m = chunk.get("meta")
    return m if isinstance(m, dict) else {}


def _get_doc_key(chunk: Dict[str, Any]) -> str:
    """doc_key peut être au root (anciens JSONL) ou dans meta (Parquet gold)."""
    return str(chunk.get("doc_key") or _meta(chunk).get("doc_key") or "")


def _get_source_path(chunk: Dict[str, Any]) -> str:
    """source_path est souvent dans meta pour le Parquet gold."""
    return str(chunk.get("source_path") or _meta(chunk).get("source_path") or "")


def _infer_doc_type(doc_key: str, source_path: str) -> str:
    """
    Infère un type minimal pour que filters.doc_types fonctionne sur Parquet.
    - LEGIARTI... => article
    - sinon heuristique via le chemin
    """
    if doc_key.startswith("LEGIARTI"):
        return "article"
    if "/article/" in source_path.replace("\\", "/"):
        return "article"
    return ""


def _get_doc_type(chunk: Dict[str, Any]) -> str:
    """doc_type au root (anciens JSONL) sinon inféré via doc_key/source_path."""
    dt = chunk.get("doc_type")
    if dt:
        return str(dt)
    doc_key = _get_doc_key(chunk)
    sp = _get_source_path(chunk)
    return _infer_doc_type(doc_key, sp)


def _get_corpus_juridique(chunk: Dict[str, Any]) -> str:
    """Lecture directe du corpus_juridique depuis chunk ou meta."""
    meta = chunk.get("meta", {})
    return str(chunk.get("corpus_juridique") or meta.get("corpus_juridique") or "")


def extract_corpus_juridique(source_path: str) -> str:
    """
    Extrait le code LEGITEXT (corpus_juridique) depuis un source_path.

    Example:
        >>> extract_corpus_juridique("/path/.../LEGITEXT000006072050/LEGIARTI000123.xml")
        'LEGITEXT000006072050'
    """
    import re
    match = re.search(r'(LEGITEXT\d+)', source_path)
    return match.group(1) if match else ""

# Ajoute ce helper avec les autres getters (_get_doc_type, _get_corpus_juridique, etc.)
def _get_status(chunk: Dict[str, Any]) -> str:
    """
    Récupère le statut/état de l'article (version) de manière robuste.
    Selon tes formats, l'info peut être :
    - au root: chunk["status"]
    - dans meta: meta["status"]
    - dans meta: meta["etat"] (souvent dérivé de version_key)
    """
    meta = _meta(chunk)
    return str(chunk.get("status") or meta.get("status") or meta.get("etat") or "")


def filter_chunks(
    chunks: List[Dict[str, Any]],
    *,
    doc_types: Optional[List[str]] = None,
    code_title_contains: Optional[str] = None,
    corpus_juridique: Optional[str] = None,
    status_in: Optional[List[str]] = None,
    as_of: Optional[str] = None,
    strict_temporal: bool = False,
    stats: Optional[Dict] = None,
    logger=None
) -> List[Dict[str, Any]]:
    """
    Filtrage compatible anciens JSONL et Parquet gold.

    Args:
        doc_types: liste de types de documents autorisés (ex: ["article"])
        code_title_contains: filtre sur le titre du code (substring)
        corpus_juridique: code LEGITEXT du corpus juridique (ex: "LEGITEXT000006072050")
        status_in: liste de statuts autorisés (ex: ["VIGUEUR"] ou ["VIGUEUR","MODIFIE"])
        as_of: date de référence pour le filtrage temporel (format "YYYY-MM-DD")
        strict_temporal: si True, rejette les chunks sans dates valides
    """
    if stats is None:
        stats = {}

    stats.update({
        "seen": 0,
        "drop_doc_types": 0,
        "drop_corpus_juridique": 0,
        "drop_status": 0,
        "keep_temporal_missing_dates_keep": 0,
        "drop_temporal_missing_dates_strict": 0,
        "drop_temporal_outside_range": 0,
        "kept": 0,
    })

    # Normalisation une fois pour éviter de refaire .upper().strip() à chaque chunk
    status_in_norm = None
    if status_in:
        status_in_norm = {str(s).strip().upper() for s in status_in if str(s).strip()}

    as_of_dt = None
    if as_of:
        as_of_dt = datetime.strptime(as_of, "%Y-%m-%d").date()

    kept: List[Dict[str, Any]] = []
    for c in chunks:
        stats["seen"] += 1

        dt = _get_doc_type(c)
        if doc_types and dt not in doc_types:
            stats["drop_doc_types"] += 1
            continue


        # Filtre sur corpus juridique (LEGITEXT)
        if corpus_juridique:
            # Lecture directe du corpus_juridique depuis chunk ou meta
            chunk_corpus = _get_corpus_juridique(c)

            # Fallback: extraction depuis source_path si pas présent
            if not chunk_corpus:
                source_path = _get_source_path(c)
                chunk_corpus = extract_corpus_juridique(source_path)

            if chunk_corpus != corpus_juridique:
                stats["drop_corpus_juridique"] += 1
                continue

        # Filtre statut (piloté par YAML)
        if status_in_norm is not None:
            st = _get_status(c).strip().upper()
            if st not in status_in_norm:
                stats["drop_status"] += 1
                continue

        # Filtrage temporel existant (as_of / strict_temporal)
        # ... ton code actuel reste inchangé ici ...

        kept.append(c)
        stats["kept"] += 1

    return kept
