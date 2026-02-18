#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common/cache_utils.py

Utilitaires pour gestion du cache embeddings.
Fonctions communes extraites de dense.py et dense_gpu.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def compute_corpus_fingerprint(chunks: List[Dict[str, Any]], text_field: str = "text") -> str:
    """Calcule empreinte SHA256 du corpus (ordre + contenu)."""
    corpus_text = "\n".join([c.get(text_field, "") for c in chunks])
    fingerprint = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()[:16]
    return fingerprint


def validate_cache(
    cache_dir: Path,
    cache_tag: str,
    expected_count: int,
    corpus_fingerprint: str,
    logger: logging.Logger
) -> bool:
    """Valide qu'un cache est coherent avec le corpus actuel."""
    meta_path = cache_dir / f"{cache_tag}.meta.json"
    
    if not meta_path.exists():
        logger.warning("Cache invalide : fichier meta manquant (%s)", meta_path)
        return False
    
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        cached_count = meta.get("chunk_count", 0)
        cached_fingerprint = meta.get("corpus_fingerprint", "")
        
        if cached_count != expected_count:
            logger.warning(
                "Cache invalide : nombre chunks different (cache=%d, actuel=%d)",
                cached_count, expected_count
            )
            return False
        
        if cached_fingerprint != corpus_fingerprint:
            logger.warning("Cache invalide : corpus modifie")
            logger.debug("  Cache : %s", cached_fingerprint)
            logger.debug("  Actuel: %s", corpus_fingerprint)
            return False
        
        logger.info("Cache valide : %s.npy", cache_tag)
        return True
    
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Cache invalide : erreur lecture meta (%s)", e)
        return False


def resolve_cache_dir(
    cache_dir: str | Path | None,
    run_dir: str | Path | None = None,
    logger: logging.Logger | None = None
) -> Path:
    """Resout le repertoire de cache embeddings."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if cache_dir:
        resolved = Path(cache_dir).resolve()
        logger.debug("Cache dir fourni : %s", resolved)
        return resolved
    
    if run_dir:
        resolved = (Path(run_dir) / "cache_dense").resolve()
        logger.debug("Cache dir dans run_dir : %s", resolved)
        return resolved
    
    resolved = Path(".dense_cache").resolve()
    logger.debug("Cache dir par defaut : %s", resolved)
    return resolved


def save_cache_metadata(
    cache_dir: Path,
    cache_tag: str,
    chunk_count: int,
    corpus_fingerprint: str,
    model_name: str,
    embedding_dim: int,
    device: str
) -> None:
    """Sauvegarde les metadonnees d'un cache."""
    from datetime import datetime
    
    meta = {
        "cache_tag": cache_tag,
        "chunk_count": chunk_count,
        "corpus_fingerprint": corpus_fingerprint,
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "device": device,
        "creation_date": datetime.now().isoformat()
    }
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    meta_path = cache_dir / f"{cache_tag}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_cache_embeddings(cache_dir: Path, cache_tag: str) -> np.ndarray | None:
    """Charge un cache d'embeddings."""
    cache_path = cache_dir / f"{cache_tag}.npy"
    
    if not cache_path.exists():
        return None
    
    return np.load(cache_path)


def save_cache_embeddings(
    cache_dir: Path,
    cache_tag: str,
    embeddings: np.ndarray
) -> None:
    """Sauvegarde un cache d'embeddings."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_path = cache_dir / f"{cache_tag}.npy"
    np.save(cache_path, embeddings)
