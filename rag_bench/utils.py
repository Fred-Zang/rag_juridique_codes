# -*- coding: utf-8 -*-
"""
Utilitaires transverses.

Ce module contient notamment un wrapper de timing pour instrumenter les étapes du pipeline
sans polluer le code métier (BM25 / dense / hybride).

Usage typique :
    timer = TimingCollector(logger=logger, run_dir=run_dir)
    with timer.timed("chargement + filtrage"):
        filtered_chunks = _validate_io_and_filter(cfg, logger)
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional


@dataclass
class TimingCollector:
    """
    Collecte des durées de traitement, avec sortie console et écriture optionnelle sur disque.

    Args:
        logger: logger Python (doit exposer .info()).
        run_dir: si fourni, permet d’écrire un fichier timings.json.
        enabled: permet de désactiver facilement le timing si besoin.
    """
    logger: any
    run_dir: Optional[str] = None
    enabled: bool = True
    timings_s: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def timed(self, label: str) -> Iterator[None]:
        """
        Context manager de timing.

        Exemple :
            with timer.timed("BM25"):
                run_bm25_benchmark(...)

        Le résultat est stocké dans `timings_s[label]` et loggué en INFO.
        """
        if not self.enabled:
            yield
            return

        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            # En cas de label répété, on cumule (pratique si une étape est appelée plusieurs fois)
            self.timings_s[label] = float(self.timings_s.get(label, 0.0) + dt)
            if self.logger is not None:
                self.logger.info("Timing | %s | %.3fs", label, dt)

    def write_json(self, filename: str = "timings.json") -> Optional[str]:
        """
        Écrit les timings dans le run_dir (si défini).

        Returns:
            Chemin du fichier écrit, ou None si run_dir absent / désactivé.
        """
        if not self.enabled or not self.run_dir:
            return None

        path = os.path.join(self.run_dir, filename)
        payload = {
            "timings_seconds": self.timings_s,
            "total_seconds": float(sum(self.timings_s.values())),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if self.logger is not None:
            self.logger.info("Timings sauvegardés : %s", path)
        return path


@contextmanager
def timed(label: str, logger=None) -> Iterator[None]:
    """
    Variante simple sans collecteur (utile dans un script isolé).

    Exemple :
        with timed("chargement", logger):
            ...
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if logger is not None:
            logger.info("Timing | %s | %.3fs", label, dt)
