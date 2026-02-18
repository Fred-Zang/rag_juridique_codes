"""
Schémas Pydantic pour l'API RAG.

Définit les modèles de requête et réponse pour l'API REST.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    """Requête RAG."""

    question: str = Field(..., description="Question à poser au système RAG")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filtres métier optionnels (corpus_juridique, status_in, as_of)",
    )

    # k est optionnel pour permettre au runtime_online.yaml de piloter le défaut.
    k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Nombre de documents à récupérer (si absent, défaut runtime)",
    )

    # retriever_type est optionnel pour permettre au runtime_online.yaml de piloter le défaut.
    retriever_type: Optional[str] = Field(
        default=None,
        description="Type de retriever: bm25, dense, ou hybrid (si absent, défaut runtime)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Quelles sont les règles de priorité aux intersections ?",
                "filters": {"status_in": ["VIGUEUR"]},
                "k": 10,
                "retriever_type": "bm25",
            }
        }



class SourceInfo(BaseModel):
    """Information sur une source utilisée."""

    chunk_id: Optional[str] = None
    article: Optional[str] = Field(None, description="Numéro d'article (ex: L1234-5)")
    code: Optional[str] = Field(None, description="Nom du code juridique")
    score: Optional[float] = Field(None, description="Score de pertinence")


class RAGResponse(BaseModel):
    """Réponse RAG."""

    request_id: str = Field(..., description="Identifiant de corrélation de la requête (X-Request-ID)")
    question: str = Field(..., description="Question posée")
    answer: str = Field(..., description="Réponse générée")
    sources: List[SourceInfo] = Field(default_factory=list, description="Sources utilisées")
    no_answer: bool = Field(default=False, description="True si pas de réponse possible")
    no_answer_reason: Optional[str] = Field(None, description="Raison du no-answer")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Score de confiance")
    latency_ms: Optional[float] = Field(None, description="Latence en millisecondes")
    timings_ms: Optional[Dict[str, float]] = Field(None, description="Latences par étape (ms)")
    resolved_corpus_juridique: Optional[str] = Field(None, description="Valeur résolue de filters.corpus_juridique")
    resolved_retriever_type: Optional[str] = Field(None, description="Retriever résolu")
    resolved_k: Optional[int] = Field(None, description="k résolu")    

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Quelle est la durée de la période d'essai pour un CDI ?",
                "answer": "La durée de la période d'essai pour un CDI varie selon la catégorie...",
                "sources": [
                    {"chunk_id": "L1221-19_chunk_0", "article": "L1221-19", "code": "Code du travail", "score": 0.85},
                ],
                "no_answer": False,
                "confidence": 0.85,
                "latency_ms": 234.5,
            }
        }


class ContextResponse(BaseModel):
    """Réponse pour l'endpoint /rag/context."""

    request_id: str = Field(..., description="Identifiant de corrélation de la requête (X-Request-ID)")
    question: str = Field(..., description="Question posée")
    context: str = Field(..., description="Contexte agrégé (sans génération LLM)")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Sources brutes du retriever")
    latency_ms: Optional[float] = Field(None, description="Latence en millisecondes")
    timings_ms: Optional[Dict[str, float]] = Field(None, description="Latences par étape (ms)")
    resolved_corpus_juridique: Optional[str] = Field(None, description="Valeur résolue de filters.corpus_juridique")
    resolved_retriever_type: Optional[str] = Field(None, description="Retriever résolu")
    resolved_k: Optional[int] = Field(None, description="k résolu")



class HealthResponse(BaseModel):
    """Réponse health check."""

    status: str = Field(default="ok")
    version: str = Field(default="0.1.0")

    # Liste des retrievers disponibles côté API
    retriever_types: List[str] = Field(
        default_factory=lambda: ["bm25", "dense", "hybrid"],
        description="Types de retrievers supportés par l'API",
    )

    # Indique si Langfuse est réellement configuré (clés présentes)
    langfuse_enabled: bool = Field(
        default=False,
        description="True si LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY sont définies",
    )

    # ----------------------------
    # Infos runtime (nouveau)
    # ----------------------------

    default_retriever: Optional[str] = Field(
        default=None,
        description="Retriever par défaut appliqué si la requête ne fournit pas retriever_type",
    )

    default_k: Optional[int] = Field(
        default=None,
        description="k par défaut appliqué si la requête ne fournit pas k",
    )

    default_corpus_juridique: Optional[str] = Field(
        default=None,
        description="corpus_juridique (LEGITEXT...) utilisé par défaut si absent des filtres",
    )

    corpora_count: int = Field(
        default=0,
        description="Nombre de corpus configurés dans runtime_online.yaml",
    )

    available_corpora: List[str] = Field(
        default_factory=list,
        description="Liste des corpus_juridique (LEGITEXT...) configurés",
    )
    llm_enabled_runtime: bool = Field(
        default=False,
        description="True si runtime_online.yaml a llm.enabled=true",
    )

    llm_ready: bool = Field(
        default=False,
        description="True si le LLM est initialisé (app.state.llm != None)",
    )

    llm_provider: Optional[str] = Field(
        default=None,
        description="Fournisseur LLM (ex: openai)",
    )

    llm_model: Optional[str] = Field(
        default=None,
        description="Nom du modèle LLM (ex: gpt-4o-mini)",
    )



class ErrorResponse(BaseModel):
    """Réponse d'erreur."""

    error: str
    detail: Optional[str] = None
    status_code: int = 500

# ---------------------------------------------------------------------------
# Pydantic v2: reconstruction des modèles pour résoudre les annotations différées
# (utile quand on utilise `from __future__ import annotations` + Optional + Field)
# Cela évite un 500 sur /openapi.json alors que les endpoints runtime fonctionnent.
# ---------------------------------------------------------------------------

for _model in (RAGRequest, SourceInfo, RAGResponse, HealthResponse, ErrorResponse):
    _model.model_rebuild()
