"""
Templates de prompts pour le RAG juridique.

Définit les prompts versionnés utilisés par les chains RAG.
Chaque prompt a un identifiant unique et peut avoir plusieurs versions.

Convention de nommage :
- {task}_{version} : ex. rag_qa_v1, rag_qa_v2
- Versions : v1 (initial), v2 (amélioré), etc.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# ═══════════════════════════════════════════════════════════════════════════
# PROMPTS RAG QA (Question-Answering)
# ═══════════════════════════════════════════════════════════════════════════

RAG_QA_V1 = """Tu es un assistant juridique spécialisé en droit du travail français.

Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.
Si le contexte ne contient pas l'information nécessaire, dis-le clairement.

Contexte :
{context}

Question : {question}

Réponse :"""


RAG_QA_V2 = """Tu es un assistant juridique expert en droit du travail français.

INSTRUCTIONS :
1. Réponds UNIQUEMENT à partir du contexte fourni ci-dessous
2. Cite les articles pertinents entre crochets [1], [2], etc.
3. Si l'information n'est pas dans le contexte, réponds : "Je ne dispose pas d'information suffisante pour répondre à cette question."
4. Sois précis et concis

CONTEXTE JURIDIQUE :
{context}

QUESTION : {question}

RÉPONSE :"""


RAG_QA_V3_WITH_CITATIONS = """Tu es un assistant juridique expert en droit du travail français.

RÈGLES STRICTES :
1. Utilise UNIQUEMENT les documents fournis dans le contexte
2. Cite TOUJOURS les sources avec leur numéro [1], [2], etc.
3. Structure ta réponse en points si la question le justifie
4. Si tu ne trouves pas l'information : "Cette information ne figure pas dans les documents consultés."

CONTEXTE JURIDIQUE :
{context}

QUESTION : {question}

RÉPONSE (avec citations) :"""


# ═══════════════════════════════════════════════════════════════════════════
# PROMPTS NO-ANSWER (Détection d'absence de réponse)
# ═══════════════════════════════════════════════════════════════════════════

RAG_WITH_NO_ANSWER_V1 = """Tu es un assistant juridique spécialisé en droit du travail français.

INSTRUCTIONS IMPORTANTES :
1. Analyse d'abord si le contexte permet de répondre à la question
2. Si OUI : réponds en citant les sources [1], [2], etc.
3. Si NON : réponds EXACTEMENT "NO_ANSWER: [raison]"

Raisons possibles pour NO_ANSWER :
- Hors sujet : la question ne concerne pas le droit du travail
- Information manquante : le contexte ne contient pas cette information
- Trop vague : la question nécessite des précisions

CONTEXTE :
{context}

QUESTION : {question}

ANALYSE ET RÉPONSE :"""


# ═══════════════════════════════════════════════════════════════════════════
# PROMPTS DE REFORMULATION
# ═══════════════════════════════════════════════════════════════════════════

QUERY_REWRITE_V1 = """Reformule cette question en langage juridique formel pour une recherche dans le Code du Travail français.

Question originale : {question}

Question reformulée :"""


QUERY_EXPAND_V1 = """Génère 3 variantes de cette question pour améliorer la recherche documentaire.
Inclus des synonymes juridiques et des termes techniques.

Question originale : {question}

Variantes (une par ligne) :
1."""


# ═══════════════════════════════════════════════════════════════════════════
# PROMPTS DE SYNTHÈSE
# ═══════════════════════════════════════════════════════════════════════════

SUMMARIZE_CONTEXT_V1 = """Synthétise les documents juridiques suivants en conservant les points clés et références.

Documents :
{context}

Synthèse concise :"""


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRE DES TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATES = {
    # RAG QA
    "rag_qa": {
        "v1": PromptTemplate.from_template(RAG_QA_V1),
        "v2": PromptTemplate.from_template(RAG_QA_V2),
        "v3": PromptTemplate.from_template(RAG_QA_V3_WITH_CITATIONS),
        "default": "v2",
    },
    # RAG avec no-answer
    "rag_no_answer": {
        "v1": PromptTemplate.from_template(RAG_WITH_NO_ANSWER_V1),
        "default": "v1",
    },
    # Reformulation
    "query_rewrite": {
        "v1": PromptTemplate.from_template(QUERY_REWRITE_V1),
        "default": "v1",
    },
    "query_expand": {
        "v1": PromptTemplate.from_template(QUERY_EXPAND_V1),
        "default": "v1",
    },
    # Synthèse
    "summarize": {
        "v1": PromptTemplate.from_template(SUMMARIZE_CONTEXT_V1),
        "default": "v1",
    },
}


# Versions ChatPromptTemplate pour les modèles chat
CHAT_PROMPT_TEMPLATES = {
    "rag_qa_chat": {
        "v1": ChatPromptTemplate.from_messages([
            ("system", "Tu es un assistant juridique expert en droit du travail français. "
                       "Réponds uniquement à partir du contexte fourni."),
            ("human", "Contexte :\n{context}\n\nQuestion : {question}"),
        ]),
        "default": "v1",
    },
}
