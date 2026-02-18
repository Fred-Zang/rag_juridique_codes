"""
Filtres métier pour le RAG juridique.

Fournit des filtres de haut niveau pour les cas d'usage courants
du domaine juridique français.

Usage:
    from rag_app.policies import MetierFilters

    # Créer un ensemble de filtres
    filters = MetierFilters.code_travail_vigueur()

    # Utiliser avec un retriever
    retriever = LegalBM25Retriever(
        corpus_path="...",
        filters=filters.to_dict(),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional


@dataclass
class MetierFilters:
    """
    Ensemble de filtres métier pour le corpus juridique.

    Encapsule les filtres courants (corpus, statut, date, type)
    avec des méthodes factory pour les cas d'usage fréquents.

    Attributes:
        corpus_juridique: ID LEGITEXT du corpus (ex: Code du Travail)
        status_in: Statuts acceptés (VIGUEUR, ABROGE, etc.)
        doc_types: Types de documents (ARTICLE, SECTION, etc.)
        as_of: Date de validité
        strict_temporal: Filtrage temporel strict

    Example:
        >>> filters = MetierFilters.code_travail_vigueur()
        >>> print(filters)
        MetierFilters(corpus=Code du Travail, status=[VIGUEUR])
    """

    corpus_juridique: Optional[str] = None
    status_in: List[str] = field(default_factory=lambda: ["VIGUEUR"])
    doc_types: Optional[List[str]] = None
    as_of: Optional[str] = None
    strict_temporal: bool = False

    # Mapping des corpus juridiques connus
    CORPUS_IDS = {
        "code_travail": "LEGITEXT000006072050",
        "code_civil": "LEGITEXT000006070721",
        "code_commerce": "LEGITEXT000005634379",
        "code_penal": "LEGITEXT000006070719",
        "code_procedure_civile": "LEGITEXT000006070716",
        "code_securite_sociale": "LEGITEXT000006073189",
    }

    @classmethod
    def code_travail_vigueur(cls, as_of: Optional[str] = None) -> "MetierFilters":
        """Filtre pour Code du Travail, textes en vigueur."""
        return cls(
            corpus_juridique=cls.CORPUS_IDS["code_travail"],
            status_in=["VIGUEUR"],
            as_of=as_of or str(date.today()),
        )

    @classmethod
    def code_civil_vigueur(cls, as_of: Optional[str] = None) -> "MetierFilters":
        """Filtre pour Code Civil, textes en vigueur."""
        return cls(
            corpus_juridique=cls.CORPUS_IDS["code_civil"],
            status_in=["VIGUEUR"],
            as_of=as_of or str(date.today()),
        )

    @classmethod
    def multi_codes(
        cls,
        codes: List[str],
        as_of: Optional[str] = None,
    ) -> "MetierFilters":
        """
        Filtre pour plusieurs codes juridiques.

        Args:
            codes: Liste de noms de codes (code_travail, code_civil, etc.)
            as_of: Date de validité

        Note:
            Pour l'instant, ne supporte qu'un seul corpus_juridique.
            TODO: Supporter filtrage multi-corpus
        """
        if len(codes) == 1:
            corpus_id = cls.CORPUS_IDS.get(codes[0])
            return cls(
                corpus_juridique=corpus_id,
                status_in=["VIGUEUR"],
                as_of=as_of or str(date.today()),
            )
        # Multi-corpus: pas de filtre corpus (tout le corpus)
        return cls(
            corpus_juridique=None,
            status_in=["VIGUEUR"],
            as_of=as_of or str(date.today()),
        )

    @classmethod
    def articles_only(cls, corpus: Optional[str] = None) -> "MetierFilters":
        """Filtre pour ne garder que les articles."""
        corpus_id = cls.CORPUS_IDS.get(corpus) if corpus else None
        return cls(
            corpus_juridique=corpus_id,
            doc_types=["ARTICLE"],
            status_in=["VIGUEUR"],
        )

    @classmethod
    def historique(
        cls,
        corpus: str,
        include_abroge: bool = True,
    ) -> "MetierFilters":
        """
        Filtre pour inclure textes historiques (abrogés).

        Utile pour recherche historique ou comparaison de versions.
        """
        corpus_id = cls.CORPUS_IDS.get(corpus)
        statuses = ["VIGUEUR", "ABROGE"] if include_abroge else ["VIGUEUR"]
        return cls(
            corpus_juridique=corpus_id,
            status_in=statuses,
            strict_temporal=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dict pour passer aux retrievers."""
        result = {}
        if self.corpus_juridique:
            result["corpus_juridique"] = self.corpus_juridique
        if self.status_in:
            result["status_in"] = self.status_in
        if self.doc_types:
            result["doc_types"] = self.doc_types
        if self.as_of:
            result["as_of"] = self.as_of
        if self.strict_temporal:
            result["strict_temporal"] = self.strict_temporal
        return result

    def with_date(self, as_of: str) -> "MetierFilters":
        """Retourne une copie avec une nouvelle date."""
        return MetierFilters(
            corpus_juridique=self.corpus_juridique,
            status_in=self.status_in,
            doc_types=self.doc_types,
            as_of=as_of,
            strict_temporal=self.strict_temporal,
        )

    def __repr__(self) -> str:
        corpus_name = "All"
        if self.corpus_juridique:
            for name, id_ in self.CORPUS_IDS.items():
                if id_ == self.corpus_juridique:
                    corpus_name = name.replace("_", " ").title()
                    break
        return f"MetierFilters(corpus={corpus_name}, status={self.status_in})"
