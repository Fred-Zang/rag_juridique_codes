"""
Gestionnaire de runs pour le RAG juridique.

Gère l'exécution d'un ensemble de requêtes avec :
- Logging local (fichiers)
- Envoi vers LangFuse
- Métriques de performance
- Audit trail complet

Usage:
    from rag_app.observability import RAGRunManager

    manager = RAGRunManager(
        run_name="test-hybrid-v2",
        output_dir="runs/",
        langfuse_enabled=True,
    )

    with manager.run() as run:
        for query in queries:
            result = run.execute(chain, query)

    print(manager.get_summary())
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rag_bench.paths import get_project_paths
from rag_bench.logging_utils import setup_logging
from rag_app.observability.langfuse_setup import (
    get_langfuse_handler,
    flush_langfuse,
    create_trace,
)

logger = setup_logging()


@dataclass
class QueryResult:
    """Résultat d'une requête RAG."""
    query_id: str
    question: str
    answer: str
    sources: List[Dict]
    latency_ms: float
    no_answer: bool = False
    no_answer_reason: Optional[str] = None
    confidence: float = 0.0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RunSummary:
    """Résumé d'un run de benchmark/test."""
    run_name: str
    start_time: str
    end_time: str
    total_queries: int
    successful_queries: int
    no_answer_queries: int
    error_queries: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float


class RAGRunManager:
    """
    Gestionnaire de runs pour le RAG juridique.

    Orchestre l'exécution de requêtes avec logging et monitoring.

    Attributes:
        run_name: Nom du run (utilisé pour les fichiers et traces)
        output_dir: Dossier de sortie pour les logs
        langfuse_enabled: Activer l'envoi vers LangFuse

    Example:
        >>> manager = RAGRunManager("test-run", langfuse_enabled=True)
        >>> with manager.run() as run:
        ...     result = run.execute(chain, {"question": "..."})
        >>> print(manager.get_summary())
    """

    def __init__(
        self,
        run_name: str,
        output_dir: Optional[str] = None,
        langfuse_enabled: bool = True,
        session_id: Optional[str] = None,
    ):
        """
        Initialise le gestionnaire de run.

        Args:
            run_name: Nom unique du run
            output_dir: Dossier de sortie (défaut: runs/)
            langfuse_enabled: Activer LangFuse
            session_id: ID de session LangFuse
        """
        self.run_name = run_name
        self.langfuse_enabled = langfuse_enabled
        self.session_id = session_id or f"run-{run_name}"

        # Configurer le dossier de sortie
        if output_dir is None:
            paths = get_project_paths()
            output_dir = str(paths.runs_dir)

        self.output_dir = Path(output_dir) / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # État du run
        self.results: List[QueryResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Configurer le logging vers fichier
        self._setup_file_logging()

        logger.info("RAGRunManager initialisé: %s", run_name)

    def _setup_file_logging(self) -> None:
        """Configure le logging vers fichier dans le dossier du run."""
        setup_logging(run_dir=str(self.output_dir), level="INFO")

    @contextmanager
    def run(self):
        """
        Context manager pour exécuter un run.

        Yields:
            RunContext pour exécuter les requêtes
        """
        self.start_time = datetime.now()
        logger.info("=== Début du run: %s ===", self.run_name)

        # Créer une trace LangFuse pour le run complet
        trace = None
        if self.langfuse_enabled:
            trace = create_trace(
                name=f"run-{self.run_name}",
                session_id=self.session_id,
                metadata={"run_name": self.run_name},
                tags=["run", "batch"],
            )

        context = _RunContext(self, trace)

        try:
            yield context
        finally:
            self.end_time = datetime.now()
            self._save_results()

            if self.langfuse_enabled:
                flush_langfuse()

            logger.info("=== Fin du run: %s | %d requêtes ===",
                        self.run_name, len(self.results))

    def _save_results(self) -> None:
        """Sauvegarde les résultats en JSON."""
        output_path = self.output_dir / "results.json"

        data = {
            "run_name": self.run_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "results": [
                {
                    "query_id": r.query_id,
                    "question": r.question,
                    "answer": r.answer,
                    "sources": r.sources,
                    "latency_ms": r.latency_ms,
                    "no_answer": r.no_answer,
                    "no_answer_reason": r.no_answer_reason,
                    "confidence": r.confidence,
                    "error": r.error,
                    "timestamp": r.timestamp,
                }
                for r in self.results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Résultats sauvegardés: %s", output_path)

    def get_summary(self) -> RunSummary:
        """
        Calcule le résumé du run.

        Returns:
            RunSummary avec statistiques
        """
        if not self.results:
            return RunSummary(
                run_name=self.run_name,
                start_time=self.start_time.isoformat() if self.start_time else "",
                end_time=self.end_time.isoformat() if self.end_time else "",
                total_queries=0,
                successful_queries=0,
                no_answer_queries=0,
                error_queries=0,
                avg_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
            )

        latencies = [r.latency_ms for r in self.results if r.error is None]
        sorted_latencies = sorted(latencies) if latencies else [0]

        return RunSummary(
            run_name=self.run_name,
            start_time=self.start_time.isoformat() if self.start_time else "",
            end_time=self.end_time.isoformat() if self.end_time else "",
            total_queries=len(self.results),
            successful_queries=sum(1 for r in self.results if r.error is None and not r.no_answer),
            no_answer_queries=sum(1 for r in self.results if r.no_answer),
            error_queries=sum(1 for r in self.results if r.error is not None),
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            p50_latency_ms=sorted_latencies[len(sorted_latencies) // 2],
            p95_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.95)],
        )


class _RunContext:
    """Contexte d'exécution pour un run."""

    def __init__(self, manager: RAGRunManager, trace):
        self.manager = manager
        self.trace = trace
        self._query_count = 0

    def execute(
        self,
        chain,
        input_data: Dict[str, Any],
        query_id: Optional[str] = None,
    ) -> QueryResult:
        """
        Exécute une requête via la chain.

        Args:
            chain: Chain LangChain à exécuter
            input_data: Données d'entrée ({"question": "..."})
            query_id: ID de la requête (auto-généré si None)

        Returns:
            QueryResult avec le résultat
        """
        self._query_count += 1
        query_id = query_id or f"q{self._query_count:04d}"
        question = input_data.get("question", "")

        # Préparer le handler LangFuse si activé
        config = {}
        if self.manager.langfuse_enabled:
            handler = get_langfuse_handler(
                session_id=self.manager.session_id,
                trace_name=f"query-{query_id}",
                metadata={"query_id": query_id},
            )
            if handler:
                config["callbacks"] = [handler]

        # Exécuter avec mesure du temps
        start = time.perf_counter()
        error = None
        result_data = {}

        try:
            result_data = chain.invoke(input_data, config=config)
        except Exception as e:
            error = str(e)
            logger.error("Erreur query %s: %s", query_id, error)

        latency_ms = (time.perf_counter() - start) * 1000

        # Construire le résultat
        result = QueryResult(
            query_id=query_id,
            question=question,
            answer=result_data.get("answer", "") if isinstance(result_data, dict) else str(result_data),
            sources=result_data.get("sources", []) if isinstance(result_data, dict) else [],
            latency_ms=latency_ms,
            no_answer=result_data.get("no_answer", False) if isinstance(result_data, dict) else False,
            no_answer_reason=result_data.get("no_answer_reason") if isinstance(result_data, dict) else None,
            confidence=result_data.get("confidence", 0.0) if isinstance(result_data, dict) else 0.0,
            error=error,
        )

        self.manager.results.append(result)

        logger.info(
            "Query %s | %.0fms | no_answer=%s | error=%s",
            query_id,
            latency_ms,
            result.no_answer,
            result.error is not None,
        )

        return result
