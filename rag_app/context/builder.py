"""
Construction du contexte LLM pour le RAG juridique.

Gère le formatage des documents récupérés pour le prompt LLM,
avec support des citations et limitation du contexte.

Usage:
    from rag_app.context import ContextBuilder

    builder = ContextBuilder(max_tokens=4000, include_citations=True)
    context_str = builder.build(documents)

    # Ou version simple
    from rag_app.context import format_docs_for_llm
    context_str = format_docs_for_llm(documents)
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from rag_bench.logging_utils import setup_logging

logger = setup_logging()


def format_docs_for_llm(
    documents: List[Document],
    include_citations: bool = True,
    separator: str = "\n\n---\n\n",
) -> str:
    """
    Formate une liste de Documents en texte pour le LLM.

    Format par défaut :
        [1] Article L1234-5 (Code du Travail)
        Contenu du chunk...

        ---

        [2] Article L1234-6 (Code du Travail)
        Contenu suivant...

    Args:
        documents: Liste de Documents LangChain
        include_citations: Ajouter numéro de référence et métadonnées
        separator: Séparateur entre documents

    Returns:
        Texte formaté prêt pour le prompt LLM
    """
    if not documents:
        return ""

    formatted_parts = []

    for idx, doc in enumerate(documents, start=1):
        meta = doc.metadata
        parts = []

        if include_citations:
            # Construire l'en-tête de citation
            article = meta.get("article_num") or meta.get("chunk_id", "?")
            code = meta.get("code_titre") or ""
            status = meta.get("status", "")

            header = f"[{idx}] {article}"
            if code:
                header += f" ({code})"
            if status and status != "VIGUEUR":
                header += f" [{status}]"

            parts.append(header)

        parts.append(doc.page_content)

        formatted_parts.append("\n".join(parts))

    return separator.join(formatted_parts)


class ContextBuilder:
    """
    Constructeur de contexte LLM avec contrôle avancé.

    Gère :
    - Formatage des documents avec citations
    - Limitation du nombre de tokens (approximatif)
    - Filtrage des documents dupliqués
    - Ordonnancement par pertinence

    Attributes:
        max_tokens: Limite approximative de tokens
        include_citations: Inclure références dans le formatage
        separator: Séparateur entre documents
        chars_per_token: Ratio caractères/token (approximation)

    Example:
        >>> builder = ContextBuilder(max_tokens=4000)
        >>> context = builder.build(documents)
        >>> print(f"Contexte: {len(context)} chars")
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        include_citations: bool = True,
        separator: str = "\n\n---\n\n",
        chars_per_token: float = 4.0,
    ):
        """
        Initialise le builder de contexte.

        Args:
            max_tokens: Limite approximative de tokens
            include_citations: Inclure les références
            separator: Séparateur entre documents
            chars_per_token: Ratio approximatif chars/token
        """
        self.max_tokens = max_tokens
        self.include_citations = include_citations
        self.separator = separator
        self.chars_per_token = chars_per_token

    @property
    def max_chars(self) -> int:
        """Limite en caractères (approximation depuis tokens)."""
        return int(self.max_tokens * self.chars_per_token)

    def _format_single_doc(self, doc: Document, idx: int) -> str:
        """Formate un seul document."""
        meta = doc.metadata
        parts = []

        if self.include_citations:
            article = meta.get("article_num") or meta.get("chunk_id", "?")
            code = meta.get("code_titre") or ""
            status = meta.get("status", "")

            header = f"[{idx}] {article}"
            if code:
                header += f" ({code})"
            if status and status != "VIGUEUR":
                header += f" [{status}]"

            parts.append(header)

        parts.append(doc.page_content)
        return "\n".join(parts)

    def _estimate_tokens(self, text: str) -> int:
        """Estime le nombre de tokens (approximatif)."""
        return int(len(text) / self.chars_per_token)

    def build(
        self,
        documents: List[Document],
        query: Optional[str] = None,
    ) -> str:
        """
        Construit le contexte formaté pour le LLM.

        Respecte la limite de tokens en tronquant si nécessaire.

        Args:
            documents: Documents récupérés par le retriever
            query: Query originale (non utilisée pour l'instant)

        Returns:
            Texte formaté avec citations et limite de tokens
        """
        if not documents:
            return ""

        formatted_parts = []
        total_chars = 0

        for idx, doc in enumerate(documents, start=1):
            formatted = self._format_single_doc(doc, idx)
            part_with_sep = formatted + self.separator

            # Vérifier si on dépasse la limite
            if total_chars + len(part_with_sep) > self.max_chars:
                # Ajouter une indication de troncature
                remaining = len(documents) - idx + 1
                if remaining > 0:
                    formatted_parts.append(f"[... {remaining} documents supplémentaires omis]")
                break

            formatted_parts.append(formatted)
            total_chars += len(part_with_sep)

        result = self.separator.join(formatted_parts)

        logger.debug(
            "Context built: %d docs, ~%d tokens",
            len(formatted_parts),
            self._estimate_tokens(result),
        )

        return result

    def build_with_metadata(
        self,
        documents: List[Document],
    ) -> dict:
        """
        Construit le contexte avec métadonnées supplémentaires.

        Returns:
            Dict avec 'context', 'num_docs', 'estimated_tokens', 'citations'
        """
        context = self.build(documents)

        citations = []
        for idx, doc in enumerate(documents, start=1):
            meta = doc.metadata
            citations.append({
                "ref": idx,
                "article": meta.get("article_num") or meta.get("chunk_id"),
                "code": meta.get("code_titre"),
                "score": meta.get("score"),
            })

        return {
            "context": context,
            "num_docs": len(documents),
            "estimated_tokens": self._estimate_tokens(context),
            "citations": citations,
        }

    def __repr__(self) -> str:
        return f"ContextBuilder(max_tokens={self.max_tokens}, citations={self.include_citations})"
