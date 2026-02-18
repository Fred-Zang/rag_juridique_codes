"""
Gestion des prompts versionnés pour le RAG juridique.

Ce module fournit :
- Un registre centralisé des prompts
- Versionnage des templates
- Support multi-langues (FR/EN)

Usage:
    from rag_app.prompts import PromptRegistry, get_prompt

    # Obtenir le prompt par défaut
    prompt = get_prompt("rag_qa")

    # Ou une version spécifique
    prompt = get_prompt("rag_qa", version="v2")

    # Dans une chain
    chain = retriever | format_docs | prompt | llm
"""

from rag_app.prompts.registry import PromptRegistry, get_prompt, list_prompts
from rag_app.prompts.templates import PROMPT_TEMPLATES

__all__ = [
    "PromptRegistry",
    "get_prompt",
    "list_prompts",
    "PROMPT_TEMPLATES",
]
