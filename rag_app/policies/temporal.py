"""
Policy de gestion temporelle pour les documents juridiques.

Les textes juridiques ont une validité temporelle (date_debut, date_fin, etat).
Cette policy permet de filtrer les documents selon une date de référence.

Usage:
    from rag_app.policies import TemporalPolicy

    policy = TemporalPolicy(as_of="2024-01-15", strict=True)
    valid_docs = policy.filter(documents)

    # Dans une chain LCEL
    chain = retriever | policy.as_runnable() | format_docs | llm
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


class TemporalPolicy:
    """
    Policy de filtrage temporel des documents juridiques.

    Filtre les documents selon leur validité à une date donnée,
    en utilisant les champs valid_from, valid_to et status.

    Attributes:
        as_of: Date de référence (None = aujourd'hui)
        strict: Si True, exclut les documents sans dates
        valid_statuses: Statuts considérés comme valides

    Example:
        >>> policy = TemporalPolicy(as_of="2024-01-15")
        >>> valid_docs = policy.filter(documents)
        >>> print(f"{len(valid_docs)} documents valides au 15/01/2024")
    """

    def __init__(
        self,
        as_of: Optional[Union[str, date]] = None,
        strict: bool = False,
        valid_statuses: Optional[List[str]] = None,
    ):
        """
        Initialise la policy temporelle.

        Args:
            as_of: Date de référence (format YYYY-MM-DD ou objet date)
            strict: Si True, exclut documents sans métadonnées temporelles
            valid_statuses: Liste des statuts valides (défaut: ["VIGUEUR"])
        """
        self.as_of = self._parse_date(as_of) if as_of else date.today()
        self.strict = strict
        self.valid_statuses = valid_statuses or ["VIGUEUR"]

    def _parse_date(self, d: Union[str, date]) -> date:
        """Convertit une date string ou date en objet date."""
        if isinstance(d, date):
            return d
        if isinstance(d, datetime):
            return d.date()
        return datetime.strptime(d, "%Y-%m-%d").date()

    def _is_valid(self, doc: Document) -> bool:
        """Vérifie si un document est valide à la date de référence."""
        meta = doc.metadata

        # Vérification du statut
        status = meta.get("status")
        if status and status not in self.valid_statuses:
            return False

        # Récupérer les dates
        valid_from = meta.get("valid_from")
        valid_to = meta.get("valid_to")

        # Si pas de dates et mode strict, exclure
        if self.strict and not valid_from and not valid_to:
            return False

        # Vérification de la période de validité
        if valid_from:
            start = self._parse_date(valid_from)
            if start > self.as_of:
                return False

        if valid_to:
            end = self._parse_date(valid_to)
            if end < self.as_of:
                return False

        return True

    def filter(self, documents: List[Document]) -> List[Document]:
        """
        Filtre les documents selon leur validité temporelle.

        Args:
            documents: Liste de Documents LangChain

        Returns:
            Liste des documents valides à la date de référence
        """
        return [doc for doc in documents if self._is_valid(doc)]

    def annotate(self, documents: List[Document]) -> List[Document]:
        """
        Annote les documents avec leur statut de validité temporelle.

        Ajoute 'temporal_valid' et 'temporal_reason' aux métadonnées.

        Args:
            documents: Liste de Documents LangChain

        Returns:
            Documents annotés (copies, originaux non modifiés)
        """
        annotated = []
        for doc in documents:
            new_meta = doc.metadata.copy()
            is_valid = self._is_valid(doc)
            new_meta["temporal_valid"] = is_valid

            if not is_valid:
                status = doc.metadata.get("status")
                if status and status not in self.valid_statuses:
                    new_meta["temporal_reason"] = f"status={status}"
                else:
                    new_meta["temporal_reason"] = f"hors période au {self.as_of}"

            annotated.append(Document(
                page_content=doc.page_content,
                metadata=new_meta,
            ))
        return annotated

    def as_runnable(self, annotate_only: bool = False) -> RunnableLambda:
        """
        Convertit la policy en Runnable LangChain pour LCEL.

        Args:
            annotate_only: Si True, annote seulement (ne filtre pas)

        Returns:
            RunnableLambda compatible LCEL
        """
        if annotate_only:
            return RunnableLambda(self.annotate)
        return RunnableLambda(self.filter)

    def __repr__(self) -> str:
        return f"TemporalPolicy(as_of={self.as_of}, strict={self.strict})"
