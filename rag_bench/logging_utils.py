# -*- coding: utf-8 -*-
"""
Configuration centralisée du logging pour rag_bench.

Objectifs :
- logs lisibles en console (développement)
- logs persistants dans runs/<run_dir>/run.log (traçabilité)
- configuration unique (évite les doublons de handlers)
"""

from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(run_dir: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Configure un logger 'rag_bench' avec sortie console + fichier (optionnel).

    Args:
        run_dir: dossier du run. Si fourni, écrit aussi dans run.log.
        level: niveau de log ("DEBUG", "INFO", "WARNING", "ERROR").

    Returns:
        logger configuré
    """
    logger = logging.getLogger("rag_bench")

    # Convertit "INFO" -> logging.INFO
    level_value = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level_value)
    logger.propagate = False

    # Évite d'ajouter plusieurs fois les handlers si setup_logging est rappelé
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level_value)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler fichier dans le run_dir
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "run.log")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level_value)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
