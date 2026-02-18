"""
Chain RAG avec détection de non-réponse.

Étend la chain simple avec la capacité de détecter quand
le système ne peut pas répondre de manière fiable.

Usage:
    from rag_app.chains import create_rag_chain_with_no_answer

    chain = create_rag_chain_with_no_answer(
        corpus_path="/path/to/gold/chunks",
        retriever_type="hybrid",
        llm=my_llm,
        min_relevance=0.3,
    )

    result = chain.invoke({"question": "Question hors sujet ?"})
    # result["no_answer"] sera True si pas de réponse possible
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import time
import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.language_models import BaseLanguageModel

from rag_app.chains.simple_rag import create_retriever
from rag_app.context import format_docs_for_llm
from rag_app.policies import NoAnswerPolicy
from rag_app.prompts import get_prompt
from rag_bench.logging_utils import setup_logging

logger = setup_logging()

# Cache en mémoire pour éviter de relire le Parquet à chaque requête.
# Clé: (corpus_path, retriever_type, k, filters_serialized)
_RETRIEVER_CACHE: dict[tuple[str, str, int, str], Runnable] = {}

def create_rag_chain_with_no_answer(
    corpus_path: Optional[str] = None,
    retriever_type: str = "hybrid",
    k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    llm: Optional[BaseLanguageModel] = None,
    min_relevance: float = 0.3,
    min_documents: int = 1,
    coverage_threshold: float = 0.5,
    no_answer_message: str = "Je ne dispose pas d'informations suffisantes pour répondre à cette question.",
    retriever: Optional[Runnable] = None,
) -> Runnable:
    """
    Crée une chain RAG avec détection de non-réponse.

    Cette chain évalue d'abord si les documents récupérés permettent
    de répondre à la question. Si non, elle retourne un message
    explicite au lieu de risquer une hallucination.

    Args:
        corpus_path: Chemin vers gold/chunks
        retriever_type: "bm25", "dense", ou "hybrid"
        k: Nombre de documents à récupérer
        filters: Filtres métier
        llm: LLM à utiliser (None = évaluation no-answer seulement)
        min_relevance: Score minimum pour considérer un doc pertinent
        min_documents: Nombre minimum de docs pertinents requis
        coverage_threshold: Couverture minimale des termes query
        no_answer_message: Message à retourner si no-answer
        retriever: retriever déjà construit (permet cache côté API, évite reload parquet)

    Returns:
        Runnable LCEL avec sortie structurée :
        {
            "question": str,
            "answer": str,
            "no_answer": bool,
            "no_answer_reason": str | None,
            "sources": list,
            "confidence": float,
        }

    Example:
        >>> chain = create_rag_chain_with_no_answer(
        ...     retriever_type="hybrid",
        ...     llm=llm,
        ...     min_relevance=0.4,
        ... )
        >>> result = chain.invoke({"question": "Comment cuisiner des pâtes ?"})
        >>> if result["no_answer"]:
        ...     print(f"Pas de réponse: {result['no_answer_reason']}")
    """
    # 1) Retriever (avec cache) : évite de relire le Parquet à chaque requête
    filters_key = json.dumps(filters or {}, ensure_ascii=False, sort_keys=True)
    cache_key = (str(corpus_path), str(retriever_type), int(k), filters_key)

    retriever = _RETRIEVER_CACHE.get(cache_key)
    if retriever is None:
        retriever = create_retriever(
            corpus_path=corpus_path,
            retriever_type=retriever_type,
            k=k,
            filters=filters,
        )
        _RETRIEVER_CACHE[cache_key] = retriever

    def retrieve_step(inputs: dict) -> dict:
        # Appel retriever unique + mesure latency
        t0 = time.perf_counter()
        docs = retriever.invoke(inputs["question"])
        retrieve_ms = (time.perf_counter() - t0) * 1000.0

        # On garde un dictionnaire timings_ms cumulatif
        timings = dict(inputs.get("timings_ms") or {})
        timings["retrieve_ms"] = retrieve_ms

        return {
            **inputs,
            "docs": docs,
            "timings_ms": timings,
        }

    def dedup_step(inputs: dict) -> dict:
        """
        Déduplique la liste de documents pour réduire les doublons de chunks
        (ex. même chunk récupéré plusieurs fois par BM25/hybride).
        La dédup se fait sur chunk_id si présent, sinon sur doc.metadata['doc_key']+chunk_index,
        avec conservation de l'ordre et du meilleur score.
        """
        t0 = time.perf_counter()
        docs = inputs.get("docs") or []

        seen = {}
        ordered_keys = []

        for doc in docs:
            md = getattr(doc, "metadata", {}) or {}
            key = md.get("chunk_id")
            if not key:
                key = f"{md.get('doc_key','NA')}|{md.get('chunk_index','NA')}"

            score = md.get("score", 0.0)

            if key not in seen:
                seen[key] = (doc, score)
                ordered_keys.append(key)
            else:
                prev_doc, prev_score = seen[key]
                if score > prev_score:
                    seen[key] = (doc, score)

        deduped_docs = [seen[k][0] for k in ordered_keys]
        dedup_ms = (time.perf_counter() - t0) * 1000.0

        timings = dict(inputs.get("timings_ms") or {})
        timings["dedup_ms"] = dedup_ms

        return {
            **inputs,
            "docs": deduped_docs,
            "timings_ms": timings,
        }

    # 2. Créer la policy no-answer
    no_answer_policy = NoAnswerPolicy(
        min_relevance=min_relevance,
        min_documents=min_documents,
        coverage_threshold=coverage_threshold,
    )

    # 3. Fonction de formatage
    def format_docs(docs):
        return format_docs_for_llm(docs, include_citations=True)

    # 4. Fonction d'évaluation no-answer
    def evaluate_and_build(inputs: dict) -> dict:
        question = inputs["question"]
        docs = inputs["docs"]

        # Évaluer si on peut répondre
        t_policy = time.perf_counter()
        result = no_answer_policy.evaluate(docs, question)
        policy_eval_ms = (time.perf_counter() - t_policy) * 1000.0

        # Construire le contexte
        t_ctx = time.perf_counter()
        context = format_docs(docs)
        context_build_ms = (time.perf_counter() - t_ctx) * 1000.0

        timings = dict(inputs.get("timings_ms") or {})
        timings["policy_eval_ms"] = policy_eval_ms
        timings["context_build_ms"] = context_build_ms

        return {
            "question": question,
            "docs": docs,
            "context": context,
            "no_answer": not result.can_answer,
            "no_answer_reason": result.reason,
            "confidence": result.confidence,
            "sources": [
                {
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "article": doc.metadata.get("article_num"),
                    "code": doc.metadata.get("code_titre"),
                    "score": doc.metadata.get("score"),
                }
                for doc in docs
            ],
            "timings_ms": timings,
        }

    # 5. Chain de base avec évaluation
    # Chaque RunnableLambda est nommé pour produire des spans lisibles dans Langfuse.
    retrieve_r = RunnableLambda(retrieve_step).with_config({"run_name": "retrieve_step"})
    dedup_r = RunnableLambda(dedup_step).with_config({"run_name": "dedup_chunks"})
    evaluate_r = RunnableLambda(evaluate_and_build).with_config({"run_name": "evaluate_and_build"})

    base_chain = (retrieve_r | dedup_r | evaluate_r).with_config({"run_name": "base_chain"})


    # 6. Si pas de LLM, retourner l'évaluation seule
    if llm is None:
        logger.info("Chain no-answer créée sans LLM (mode évaluation seule)")

        def build_eval_only_output(x: dict) -> dict:
            return {
                "question": x["question"],
                "answer": no_answer_message if x["no_answer"] else "[LLM requis pour générer la réponse]",
                "no_answer": x["no_answer"],
                "no_answer_reason": x["no_answer_reason"],
                "confidence": x["confidence"],
                "sources": x["sources"],
                "timings_ms": x.get("timings_ms") or {},
            }

        output_r = RunnableLambda(build_eval_only_output).with_config({"run_name": "build_output_no_llm"})

        # On nomme aussi la chain racine pour que la trace Langfuse ne s'appelle pas "RunnableSequence"
        return (base_chain | output_r).with_config({"run_name": "rag_with_no_answer_no_llm"})

    # 7. Chain complète avec génération conditionnelle
    prompt = get_prompt("rag_no_answer", version="v1")

    llm_chain = prompt | llm | StrOutputParser()


    def generate_or_no_answer(inputs: dict) -> dict:
        if inputs["no_answer"]:
            return {
                "question": inputs["question"],
                "answer": no_answer_message,
                "no_answer": True,
                "no_answer_reason": inputs["no_answer_reason"],
                "confidence": inputs["confidence"],
                "sources": inputs["sources"],
                "timings_ms": inputs.get("timings_ms") or {},
            }

        # Générer la réponse via LLM (un seul appel) + mesure
        t_llm = time.perf_counter()
        answer = llm_chain.invoke(
            {
                "context": inputs["context"],
                "question": inputs["question"],
            }
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000.0

        timings = dict(inputs.get("timings_ms") or {})
        timings["llm_ms"] = llm_ms


        # Détecter si le LLM a lui-même signalé NO_ANSWER
        if answer.strip().startswith("NO_ANSWER:"):
            reason = answer.split("NO_ANSWER:")[-1].strip()
            return {
                "question": inputs["question"],
                "answer": no_answer_message,
                "no_answer": True,
                "no_answer_reason": f"LLM: {reason}",
                "confidence": inputs["confidence"] * 0.5,
                "sources": inputs["sources"],
                "timings_ms": timings,
            }

        return {
            "question": inputs["question"],
            "answer": answer,
            "no_answer": False,
            "no_answer_reason": None,
            "confidence": inputs["confidence"],
            "sources": inputs["sources"],
            "timings_ms": timings,
        }

    generate_r = RunnableLambda(generate_or_no_answer).with_config({"run_name": "generate_or_no_answer"})
    chain = (base_chain | generate_r).with_config({"run_name": "rag_with_no_answer"})
    return chain

