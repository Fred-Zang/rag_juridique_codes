#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py

Utilitaires communs pour le pipeline datalake.
"""

from __future__ import annotations

import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Charge un fichier YAML en dictionnaire Python.
    
    Args:
        path: Chemin vers le fichier YAML
    
    Returns:
        Dictionnaire des configurations
    
    Raises:
        RuntimeError: Si PyYAML absent ou fichier illisible
    """
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "PyYAML non disponible. Installez-le: pip install pyyaml"
        ) from e
    
    try:
        logger.info("Chargement configuration: %s", path)
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        if config is None:
            logger.warning("⚠️ Fichier YAML vide ou invalide: %s", path)
            return {}
        
        logger.info("✅ Configuration chargée: %d clés racine", len(config))
        return config
    
    except FileNotFoundError as e:
        raise RuntimeError(f"Fichier YAML introuvable: {path}") from e
    except yaml.YAMLError as e:
        raise RuntimeError(f"Erreur parsing YAML {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Erreur inattendue lors du chargement de {path}: {e}") from e


def human_readable_size(size_bytes: int) -> str:
    """
    Convertit une taille en octets en format lisible (Ko, Mo, Go).
    
    Args:
        size_bytes: Taille en octets
    
    Returns:
        String formaté (ex: "1.23 Mo")
    """
    for unit in ['o', 'Ko', 'Mo', 'Go', 'To']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} Po"