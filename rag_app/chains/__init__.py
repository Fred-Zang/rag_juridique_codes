"""
Chains LCEL pour le RAG juridique.

Ce module fournit des chains pré-construites :
- simple_rag: RAG basique (retriever → prompt → llm)
- with_no_answer: RAG avec détection de non-réponse
- conversational: RAG avec historique de conversation

Usage:
    from rag_app.chains import create_rag_chain

    chain = create_rag_chain(
        corpus_path="/path/to/gold/chunks",
        retriever_type="hybrid",
        llm=my_llm,
    )

    result = chain.invoke({"question": "Quelle est la durée du préavis ?"})
"""

from rag_app.chains.simple_rag import create_rag_chain, create_retriever
from rag_app.chains.with_no_answer import create_rag_chain_with_no_answer

__all__ = [
    "create_rag_chain",
    "create_retriever",
    "create_rag_chain_with_no_answer",
]
