"""
Policies métier pour le RAG juridique.

Ces policies implémentent les règles métier spécifiques au domaine juridique :
- Temporalité des textes (validité, versions)
- Détection de situations "no-answer"
- Filtres métier avancés

Usage:
    from rag_app.policies import TemporalPolicy, NoAnswerPolicy

    # Vérifier la validité temporelle des documents
    temporal = TemporalPolicy(as_of="2024-01-15")
    valid_docs = temporal.filter(documents)

    # Détecter si une réponse est possible
    no_answer = NoAnswerPolicy(min_relevance=0.3)
    can_answer, reason = no_answer.evaluate(documents, query)
"""

from rag_app.policies.temporal import TemporalPolicy
from rag_app.policies.no_answer import NoAnswerPolicy
from rag_app.policies.filters import MetierFilters

__all__ = [
    "TemporalPolicy",
    "NoAnswerPolicy",
    "MetierFilters",
]
