"""
rag_app - Application RAG Runtime pour corpus juridique français

Ce package fournit une couche LangChain au-dessus de nos modules
de benchmark existants (rag_bench/), permettant :

- Orchestration LCEL des retrievers (BM25, Dense, Hybrid)
- Policies métier juridiques (temporalité, no-answer, citations)
- Monitoring via LangFuse (self-hosted)
- API REST via LangServe
- Versioning des prompts

Architecture :
    rag_app/
    ├── retrievers/      # Wrappers LangChain → rag_bench/*
    ├── policies/        # Règles métier juridiques
    ├── context/         # Construction contexte LLM
    ├── prompts/         # Templates versionnés
    ├── chains/          # Orchestration LCEL
    ├── observability/   # LangFuse + audit
    └── api/             # LangServe endpoints

Usage:
    from rag_app.chains.simple_rag import create_rag_chain
    from rag_app.observability.langfuse_setup import get_langfuse_handler

    chain = create_rag_chain(corpus_path="...", retriever_type="hybrid")
    handler = get_langfuse_handler(session_id="test-001")
    result = chain.invoke({"question": "..."}, config={"callbacks": [handler]})

Voir README_RAG_APP.md pour la documentation complète.
"""

__version__ = "0.1.0"
__author__ = "Fred"
