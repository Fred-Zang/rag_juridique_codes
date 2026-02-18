"""
API REST pour le RAG juridique (via LangServe / FastAPI).

Ce module expose les chains RAG via une API HTTP :
- POST /rag/invoke : Requête RAG simple
- POST /rag/stream : Requête RAG en streaming
- GET /health : Health check

Usage (démarrage serveur):
    python -m rag_app.api.app

    # Ou avec uvicorn directement
    uvicorn rag_app.api.app:app --host 0.0.0.0 --port 8000

Usage (client):
    import requests

    response = requests.post(
        "http://localhost:8000/rag/invoke",
        json={"input": {"question": "Quelle est la durée du préavis ?"}}
    )
    print(response.json())
"""

from rag_app.api.app import create_app, app
from rag_app.api.schemas import RAGRequest, RAGResponse

__all__ = [
    "create_app",
    "app",
    "RAGRequest",
    "RAGResponse",
]
