"""
Configuration LangFuse pour le RAG juridique.

LangFuse est une plateforme d'observabilité open-source pour LLM apps.
Ce module configure l'intégration avec LangChain.

Prérequis :
    1. Lancer LangFuse : docker-compose -f docker-compose.langfuse.yml up -d
    2. Créer un projet dans l'UI : http://localhost:3000
    3. Configurer les variables d'environnement :
       - LANGFUSE_PUBLIC_KEY=pk-lf-...
       - LANGFUSE_SECRET_KEY=sk-lf-...
       - LANGFUSE_HOST=http://localhost:3000

Usage:
    from rag_app.observability import get_langfuse_handler

    handler = get_langfuse_handler(session_id="user-session-123")

    # Avec une chain LangChain
    result = chain.invoke(
        {"question": "..."},
        config={"callbacks": [handler]},
    )
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Dict, Optional

from rag_bench.logging_utils import setup_logging

logger = setup_logging()

def _env_langfuse_public_key() -> str:
    return os.getenv("LANGFUSE_PUBLIC_KEY", "")

def _env_langfuse_secret_key() -> str:
    return os.getenv("LANGFUSE_SECRET_KEY", "")

def _env_langfuse_host() -> str:
    return os.getenv("LANGFUSE_HOST", os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"))

def get_langfuse_client():
    """
    Compat: certaines parties du code importent encore get_langfuse_client().
    En SDK v3, on utilise get_client() côté langfuse.
    """
    if not _check_langfuse_config():
        return None
    try:
        from langfuse import get_client
        return get_client()
    except ImportError:
        logger.error("Package langfuse non installé. Installez avec: pip install langfuse")
        return None


def create_trace(name: str, *, session_id: Optional[str] = None, user_id: Optional[str] = None, tags: Optional[list[str]] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Compat: création manuelle d'une trace Langfuse, utilisée par certains managers.

    Remarque:
    - Dans notre intégration principale LangChain, les traces sont créées automatiquement
      via CallbackHandler + propagate_attributes.
    - Cette fonction sert uniquement aux usages "manual tracing".
    """
    client = get_langfuse_client()
    if client is None:
        return None

    # En SDK v3, la création se fait via client.trace(...)
    try:
        return client.trace(
            name=name,
            session_id=session_id,
            user_id=user_id,
            tags=tags or [],
            metadata=metadata or {},
        )
    except TypeError:
        # Fallback si la signature diffère selon version
        return client.trace(name=name)


def flush_langfuse() -> None:
    """
    Compat: forcer l'envoi des événements.
    À éviter dans l'API sur chaque requête, mais utile en scripts/tests.
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        # On ne casse pas le runtime si flush échoue
        return

def _check_langfuse_config() -> bool:
    public_key = _env_langfuse_public_key()
    secret_key = _env_langfuse_secret_key()

    if not public_key or not secret_key:
        logger.warning("Langfuse non configuré: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY manquantes.")
        return False
    return True


def _to_str(value: Any) -> str:
    """
    Convertit une valeur en string de façon stable pour Langfuse.
    Objectif: éviter les 'Dropping value' quand un champ metadata n'est pas un string.
    """
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        import json
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def sanitize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    Retourne un dict metadata où toutes les valeurs sont des strings.
    """
    if not metadata:
        return {}
    return {str(k): _to_str(v) for k, v in metadata.items()}

def get_langfuse_handler(*_args, **_kwargs):
    """
    Handler LangChain -> Langfuse (SDK v3).

    Compat:
    - Certains appels passent encore session_id / trace_name / metadata.
      Dans SDK v3, ces attributs passent via propagate_attributes(...),
      donc on ignore ces kwargs ici pour rester compatible.
    """

    if not _check_langfuse_config():
        return None

    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except ImportError:
        logger.error("Package langfuse non installé. Installez avec: pip install langfuse")
        return None


def get_langfuse_context(
    *,
    session_id: Optional[str],
    user_id: Optional[str],
    tags: Optional[list[str]],
    metadata: Optional[Dict[str, Any]],
):
    """
    Retourne un context manager Langfuse qui propage des attributs au trace courant.

    Si Langfuse n'est pas configuré, on renvoie un nullcontext() pour ne pas impacter le runtime.
    """
    if not _check_langfuse_config():
        return nullcontext()

    try:
        from langfuse import propagate_attributes

        return propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=tags or [],
            metadata=sanitize_metadata(metadata),
        )
    except ImportError:
        logger.error("Package langfuse non installé. Installez avec: pip install langfuse")
        return nullcontext()
