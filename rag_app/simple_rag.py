"""
Chain RAG simple pour le corpus juridique.

Implémente le pattern RAG classique :
    question → retriever → format_docs → prompt → llm → réponse

Usage CLI:
    # Test rapide (nécessite OPENAI_API_KEY ou autre LLM configuré)
    python -c "
    from rag_app.chains import create_rag_chain
    chain = create_rag_chain(corpus_path='...', retriever_type='bm25')
    print(chain.invoke({'question': 'durée période essai'}))
    "

Usage:
    from rag_app.chains import create_rag_chain

    # Créer la chain
    chain = create_rag_chain(
        corpus_path="/path/to/gold/chunks",
        retriever_type="hybrid",
        k=10,
        llm=my_llm,  # ou None pour retourner juste le contexte
    )

    # Invoquer
    result = chain.invoke({"question": "Quelle est la durée du préavis ?"})
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import time
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.language_models import BaseLanguageModel

from rag_app.retrievers import LegalBM25Retriever, LegalDenseRetriever, LegalHybridRetriever
from rag_app.context import format_docs_for_llm
from rag_app.prompts import get_prompt
from rag_bench.paths import get_project_paths
from rag_bench.logging_utils import setup_logging

logger = setup_logging()


def create_retriever(
    corpus_path: Optional[str] = None,
    retriever_type: str = "hybrid",
    k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[LegalBM25Retriever, LegalDenseRetriever, LegalHybridRetriever]:
    """
    Factory pour créer un retriever selon le type demandé.

    Args:
        corpus_path: Chemin vers gold/chunks (None = auto-détection)
        retriever_type: "bm25", "dense", ou "hybrid"
        k: Nombre de résultats
        filters: Filtres métier (corpus_juridique, status_in, etc.)
        **kwargs: Arguments supplémentaires passés au retriever

    Returns:
        Instance de LegalBM25Retriever, LegalDenseRetriever ou LegalHybridRetriever
    """
    # Auto-détection du corpus si non spécifié
    if corpus_path is None:
        paths = get_project_paths()
        corpus_path = str(paths.gold_corpus_dir)
        logger.info("Corpus auto-détecté: %s", corpus_path)

    filters = filters or {}

    if retriever_type == "bm25":
        return LegalBM25Retriever(
            corpus_path=corpus_path,
            k=k,
            filters=filters,
            **kwargs,
        )
    elif retriever_type == "dense":
        return LegalDenseRetriever(
            corpus_path=corpus_path,
            k=k,
            filters=filters,
            cache_dir=kwargs.pop("cache_dir", str(get_project_paths().cache_dense_dir)),
            **kwargs,
        )
    elif retriever_type == "hybrid":
        return LegalHybridRetriever(
            corpus_path=corpus_path,
            k=k,
            filters=filters,
            cache_dir=kwargs.pop("cache_dir", str(get_project_paths().cache_dense_dir)),
            **kwargs,
        )
    else:
        raise ValueError(f"Type de retriever inconnu: {retriever_type}. "
                         f"Utilisez 'bm25', 'dense', ou 'hybrid'")


def create_rag_chain(
    corpus_path: Optional[str] = None,
    retriever_type: str = "hybrid",
    k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    llm: Optional[BaseLanguageModel] = None,
    prompt_name: str = "rag_qa",
    prompt_version: Optional[str] = None,
    include_sources: bool = True,
) -> Runnable:
    """
    Crée une chain RAG complète pour le corpus juridique.

    Architecture LCEL :
        {question} → retriever → format_docs → prompt → llm → output

    Args:
        corpus_path: Chemin vers gold/chunks (None = auto)
        retriever_type: "bm25", "dense", ou "hybrid"
        k: Nombre de documents à récupérer
        filters: Filtres métier
        llm: LLM à utiliser (None = retourne contexte sans génération)
        prompt_name: Nom du prompt dans le registre
        prompt_version: Version du prompt (None = default)
        include_sources: Inclure les sources dans la sortie

    Returns:
        Runnable LCEL invocable avec {"question": "..."}

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> chain = create_rag_chain(retriever_type="hybrid", llm=llm)
        >>> result = chain.invoke({"question": "durée période essai CDI"})
    """
    # 1. Créer le retriever
    retriever = create_retriever(
        corpus_path=corpus_path,
        retriever_type=retriever_type,
        k=k,
        filters=filters,
    )

    # 2. Fonction de formatage
    def format_docs(docs):
        return format_docs_for_llm(docs, include_citations=True)

    # Étape retrieval instrumentée (un seul invoke retriever)
    def retrieve_step(inputs: dict) -> dict:
        t0 = time.perf_counter()
        docs = retriever.invoke(inputs["question"])
        retrieve_ms = (time.perf_counter() - t0) * 1000.0

        timings = dict(inputs.get("timings_ms") or {})
        timings["retrieve_ms"] = retrieve_ms

        return {
            **inputs,
            "docs": docs,
            "timings_ms": timings,
        }

    # Étape build context + sources instrumentée
    def build_context_sources_step(inputs: dict) -> dict:
        t0 = time.perf_counter()
        docs = inputs["docs"]
        context = format_docs(docs)
        sources = [
            {
                "chunk_id": doc.metadata.get("chunk_id"),
                "article": doc.metadata.get("article_num"),
                "code": doc.metadata.get("code_titre"),
                "score": doc.metadata.get("score"),
            }
            for doc in docs
        ]
        context_build_ms = (time.perf_counter() - t0) * 1000.0

        timings = dict(inputs.get("timings_ms") or {})
        timings["context_build_ms"] = context_build_ms

        return {
            **inputs,
            "context": context,
            "sources": sources,
            "timings_ms": timings,
        }

    # 3. Si pas de LLM, retourner juste le contexte
    if llm is None:
        logger.info("Chain créée sans LLM (mode contexte seul)")

        if include_sources:
            def context_only_output(inputs: dict) -> dict:
                return {
                    "question": inputs["question"],
                    "context": inputs["context"],
                    "sources": inputs["sources"],
                    "timings_ms": inputs.get("timings_ms") or {},
                }

            return (
                RunnableLambda(retrieve_step)
                | RunnableLambda(build_context_sources_step)
                | RunnableLambda(context_only_output)
            )
        else:
            def context_only_output(inputs: dict) -> dict:
                return {
                    "question": inputs["question"],
                    "context": inputs["context"],
                    "timings_ms": inputs.get("timings_ms") or {},
                }

            return (
                RunnableLambda(retrieve_step)
                | RunnableLambda(
                    lambda x: {
                        **x,
                        "context": format_docs(x["docs"]),
                        "timings_ms": {
                            **(x.get("timings_ms") or {}),
                            "context_build_ms": 0.0,  # tu peux aussi mesurer ici si tu veux
                        },
                    }
                )
                | RunnableLambda(context_only_output)
            )

    # 4. Chain complète avec LLM
    prompt = get_prompt(prompt_name, prompt_version)
    llm_chain = prompt | llm | StrOutputParser()

    def generate_step(inputs: dict) -> dict:
        t0 = time.perf_counter()
        answer = llm_chain.invoke(inputs)
        llm_ms = (time.perf_counter() - t0) * 1000.0

        timings = dict(inputs.get("timings_ms") or {})
        timings["llm_ms"] = llm_ms

        return {**inputs, "answer": answer, "timings_ms": timings}

    def final_output(inputs: dict) -> dict:
        out = {
            "question": inputs["question"],
            "answer": inputs["answer"],
        }
        if include_sources:
            out["sources"] = inputs.get("sources") or []
        out["timings_ms"] = inputs.get("timings_ms") or {}
        return out

    if include_sources:
        chain = (
            RunnableLambda(retrieve_step)
            | RunnableLambda(build_context_sources_step)
            | RunnableLambda(generate_step)
            | RunnableLambda(final_output)
        )
    else:
        chain = (
            RunnableLambda(retrieve_step)
            | RunnableLambda(lambda x: {**x, "context": format_docs(x["docs"])})
            | RunnableLambda(generate_step)
            | RunnableLambda(final_output)
        )

    return chain
