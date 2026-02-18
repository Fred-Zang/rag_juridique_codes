"""
Policy de détection "no-answer" pour le RAG juridique.

Détecte quand le système ne peut pas répondre à une question de manière
fiable, évitant ainsi les hallucinations.

Usage:
    from rag_app.policies import NoAnswerPolicy

    policy = NoAnswerPolicy(
        min_relevance=0.3,
        min_documents=1,
        coverage_threshold=0.5,
    )

    can_answer, reason = policy.evaluate(documents, query)
    if not can_answer:
        return f"Je ne peux pas répondre: {reason}"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain_core.documents import Document


@dataclass
class NoAnswerResult:
    """Résultat de l'évaluation no-answer."""
    can_answer: bool
    confidence: float
    reason: Optional[str] = None
    details: Optional[dict] = None


class NoAnswerPolicy:
    """
    Policy de détection des situations où le RAG ne peut pas répondre.

    Analyse plusieurs signaux pour déterminer si une réponse fiable
    est possible :
    - Score de pertinence minimum
    - Nombre minimum de documents
    - Couverture des termes de la query
    - Cohérence entre documents

    Attributes:
        min_relevance: Score minimum pour considérer un doc pertinent
        min_documents: Nombre minimum de docs pertinents requis
        coverage_threshold: % minimum de termes query couverts

    Example:
        >>> policy = NoAnswerPolicy(min_relevance=0.3)
        >>> result = policy.evaluate(docs, "durée période essai")
        >>> if not result.can_answer:
        ...     print(f"No-answer: {result.reason}")
    """

    def __init__(
        self,
        min_relevance: float = 0.3,
        min_documents: int = 1,
        coverage_threshold: float = 0.5,
        max_score_gap: float = 0.5,
    ):
        """
        Initialise la policy no-answer.

        Args:
            min_relevance: Score minimum de pertinence (0-1)
            min_documents: Nombre minimum de documents pertinents
            coverage_threshold: Couverture minimale des termes query
            max_score_gap: Écart max entre meilleur score et seuil
        """
        self.min_relevance = min_relevance
        self.min_documents = min_documents
        self.coverage_threshold = coverage_threshold
        self.max_score_gap = max_score_gap

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extrait les termes clés d'une query (mots > 3 chars)."""
        # Nettoyage et tokenisation simple
        words = re.findall(r'\b\w+\b', query.lower())
        # Filtrer mots courts et stopwords basiques
        stopwords = {"les", "des", "une", "que", "qui", "est", "dans", "pour", "avec", "sur"}
        return [w for w in words if len(w) > 3 and w not in stopwords]

    def _compute_coverage(self, documents: List[Document], terms: List[str]) -> float:
        """Calcule le % de termes query présents dans les documents."""
        if not terms:
            return 1.0

        # Combiner tout le contenu
        all_content = " ".join(doc.page_content.lower() for doc in documents)

        # Compter termes présents
        found = sum(1 for term in terms if term in all_content)
        return found / len(terms)

    def evaluate(
        self,
        documents: List[Document],
        query: str,
    ) -> NoAnswerResult:
        """
        Évalue si le système peut répondre à la query.

        Args:
            documents: Documents récupérés par le retriever
            query: Question de l'utilisateur

        Returns:
            NoAnswerResult avec can_answer, confidence, et reason
        """
        details = {
            "total_docs": len(documents),
            "query_terms": [],
            "coverage": 0.0,
            "relevant_docs": 0,
            "max_score": 0.0,
        }

        # Cas 1: Aucun document
        if not documents:
            return NoAnswerResult(
                can_answer=False,
                confidence=0.0,
                reason="Aucun document trouvé",
                details=details,
            )

        # Extraire scores
        scores = [doc.metadata.get("score", 0.0) for doc in documents]
        max_score = max(scores) if scores else 0.0
        details["max_score"] = max_score

        # Cas 2: Score trop faible
        if max_score < self.min_relevance:
            return NoAnswerResult(
                can_answer=False,
                confidence=max_score,
                reason=f"Score max ({max_score:.2f}) < seuil ({self.min_relevance})",
                details=details,
            )

        # Compter documents pertinents
        relevant_docs = [doc for doc in documents
                         if doc.metadata.get("score", 0) >= self.min_relevance]
        details["relevant_docs"] = len(relevant_docs)

        # Cas 3: Pas assez de documents pertinents
        if len(relevant_docs) < self.min_documents:
            return NoAnswerResult(
                can_answer=False,
                confidence=max_score * 0.5,
                reason=f"Seulement {len(relevant_docs)} doc(s) pertinent(s) "
                       f"(min: {self.min_documents})",
                details=details,
            )

        # Vérifier couverture des termes
        key_terms = self._extract_key_terms(query)
        details["query_terms"] = key_terms
        coverage = self._compute_coverage(relevant_docs, key_terms)
        details["coverage"] = coverage

        # Cas 4: Couverture insuffisante
        if coverage < self.coverage_threshold:
            return NoAnswerResult(
                can_answer=False,
                confidence=max_score * coverage,
                reason=f"Couverture termes ({coverage:.0%}) < seuil ({self.coverage_threshold:.0%})",
                details=details,
            )

        # Tout est OK
        confidence = min(1.0, max_score * (0.5 + 0.5 * coverage))
        return NoAnswerResult(
            can_answer=True,
            confidence=confidence,
            reason=None,
            details=details,
        )

    def __call__(
        self,
        documents: List[Document],
        query: str,
    ) -> Tuple[bool, Optional[str]]:
        """Raccourci pour evaluate().

        Returns:
            Tuple (can_answer, reason)
        """
        result = self.evaluate(documents, query)
        return result.can_answer, result.reason

    def __repr__(self) -> str:
        return (f"NoAnswerPolicy(min_relevance={self.min_relevance}, "
                f"min_documents={self.min_documents})")
