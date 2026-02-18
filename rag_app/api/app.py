"""
Application FastAPI pour l'API RAG juridique.

Expose les chains RAG via une API REST avec LangServe.

Usage:
    # Démarrage direct
    python -m rag_app.api.app

    # Ou avec uvicorn
    uvicorn rag_app.api.app:app --host 0.0.0.0 --port 8000 --reload

    # Test rapide
curl -s -X POST "http://127.0.0.1:8000/rag/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "question":"Quelle sanction est prévue si un conducteur ne respecte pas les règles de priorité ?",
    "retriever_type":"bm25",
    "k": 8,
    "filters":{"corpus://":"LEGITEXT000006074228"},
    "llm_enabled": true
  }' | jq '.no_answer,.no_answer_reason,.answer'

# Règle de configuration:
# - YAML définit les valeurs par défaut (reproductible, versionné)
# - Les variables d'environnement restent prioritaires (override prod/CI)

"""

from __future__ import annotations

import os
import time
from typing import Optional
from pathlib import Path
import uuid
import json
from dotenv import load_dotenv as dotenv_load

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse

from rag_app.api.schemas import (
    RAGRequest,
    RAGResponse,
    ContextResponse,
    SourceInfo,
    HealthResponse,
    ErrorResponse,
)
from rag_app.chains import create_rag_chain, create_rag_chain_with_no_answer
from rag_app.observability import get_langfuse_handler, get_langfuse_context
from rag_app.observability.langfuse_setup import sanitize_metadata
from rag_bench.paths import get_project_paths
from rag_bench.logging_utils import setup_logging
from rag_app.runtime.config_loader import apply_runtime_online_yaml


logger = setup_logging()
# log “startup_ms” côté app
_APP_IMPORT_T0 = time.perf_counter()

def create_app(
    corpus_path: Optional[str] = None,
    default_retriever: str = "hybrid",
    enable_langfuse: bool = True,
    llm=None,
) -> FastAPI:
    """
    Crée l'application FastAPI pour l'API RAG.

    Args:
        corpus_path: Chemin vers le corpus (None = auto-détection)
        default_retriever: Retriever par défaut (bm25, dense, hybrid)
        enable_langfuse: Activer le monitoring LangFuse
        llm: LLM à utiliser (None = mode sans génération)

    Returns:
        Application FastAPI configurée
    """
    # timer app
    t_create0 = time.perf_counter()
    # Charger explicitement le .env pour le process uvicorn (sinon variables absentes)
    paths = get_project_paths()
    dotenv_load(dotenv_path=str(paths.repo_root / ".env"), override=False)
    logger.info(
        "ENV check: openai=%s langfuse_public=%s langfuse_secret=%s",
        bool(os.getenv("OPENAI_API_KEY")),
        bool(os.getenv("LANGFUSE_PUBLIC_KEY")),
        bool(os.getenv("LANGFUSE_SECRET_KEY")),
    )
    app = FastAPI(
        title="Legal RAG API",
        description="API RAG pour le corpus juridique français (Code du Travail)",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS (à restreindre en production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Middleware de corrélation (request_id)
    #
    # Pourquoi:
    # - Permet de tracer une requête bout-en-bout (logs, Langfuse, client HTTP).
    # - Si le client fournit X-Request-ID, on le réutilise (corrélation amont).
    # - Sinon, on génère un UUID.
    #
    # Règle:
    # - Le request_id est stocké dans request.state.request_id
    # - Il est renvoyé en header X-Request-ID dans toutes les réponses
    # - Il est aussi renvoyé dans le JSON des endpoints principaux
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def add_request_id_middleware(request: Request, call_next):
        incoming = request.headers.get("X-Request-ID")
        request_id = incoming.strip() if isinstance(incoming, str) and incoming.strip() else str(uuid.uuid4())
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception:
            # Les exception handlers ci-dessous ajouteront aussi X-Request-ID
            raise

        response.headers["X-Request-ID"] = request_id
        return response


    # ------------------------------------------------------------------
    # Exception handlers enrichis avec request_id
    #
    # Pourquoi:
    # - Même en cas d'erreur, le client doit récupérer X-Request-ID.
    # - Les logs peuvent ainsi être corrélés facilement.
    # ------------------------------------------------------------------
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        payload = {"error": exc.detail, "request_id": request_id}
        response = JSONResponse(status_code=exc.status_code, content=payload)
        response.headers["X-Request-ID"] = request_id
        return response


    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        payload = {"error": str(exc), "request_id": request_id}
        response = JSONResponse(status_code=500, content=payload)
        response.headers["X-Request-ID"] = request_id
        return response

    # Charger runtime_online.yaml si configuré (YAML = défaut, ENV = override)
    runtime_cfg_path = os.getenv("RAG_RUNTIME_CONFIG")
    if runtime_cfg_path:
        try:
            # On conserve le dict YAML en mémoire (app.state.runtime_cfg) pour pouvoir
            # résoudre dynamiquement, à chaque requête, quel parquet charger en fonction
            # de filters.corpus_juridique (LEGITEXT...) sans redémarrer l'API.
            cfg = apply_runtime_online_yaml(runtime_cfg_path)
            app.state.runtime_cfg = cfg

            logger.info("runtime_online.yaml appliqué via RAG_RUNTIME_CONFIG=%s", runtime_cfg_path)
        except Exception as e:
            logger.error("Impossible d'appliquer runtime_online.yaml (%s): %s", runtime_cfg_path, str(e))
            raise
    else:
        # Fallback dev: on tente d'abord configs/runtime_online.yaml (convention projet),
        # puis runtime_online.yaml à la racine si besoin.
        candidates = [
            Path.cwd() / "configs" / "runtime_online.yaml",
            Path.cwd() / "runtime_online.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    cfg = apply_runtime_online_yaml(str(candidate))
                    app.state.runtime_cfg = cfg

                    logger.info("runtime_online.yaml appliqué depuis %s", str(candidate))
                    break
                except Exception as e:
                    logger.error("Impossible d'appliquer runtime_online.yaml (%s): %s", str(candidate), str(e))
                    raise
        # Si aucun YAML n'a été chargé, on initialise un dict vide pour éviter les None checks partout.
        if not hasattr(app.state, "runtime_cfg"):
            app.state.runtime_cfg = {}
            
    # Si le YAML a fourni un défaut de k, on l'applique au default de l'app
    default_k_env = os.getenv("RAG_DEFAULT_K")
    if default_k_env and default_k_env.isdigit():
        app.state.default_k = int(default_k_env)
    else:
        app.state.default_k = None

    # Auto-détection du corpus (avec override via variable d’environnement)
    if corpus_path is None:
        env_corpus = os.getenv("RAG_CORPUS_PATH")
        if env_corpus:
            corpus_path = env_corpus
        else:
            paths = get_project_paths()
            corpus_path = str(paths.gold_corpus_dir)

    # Normaliser le chemin (évite les différences CWD / relatif vs absolu)
    corpus_path = str(Path(corpus_path).expanduser().resolve())

    # Retriever par défaut configurable via env (pratique en dev)
    default_retriever = os.getenv("RAG_DEFAULT_RETRIEVER", default_retriever)

    app.state.corpus_path = corpus_path
    app.state.default_retriever = default_retriever
    app.state.enable_langfuse = enable_langfuse
    # Cache runtime:
    # - _chain_cache: évite de reconstruire les chains (et donc de recharger le parquet) à chaque requête
    app.state._chain_cache = {}


    # ------------------------------------------------------------------
    # Initialisation LLM (pilotée par runtime_online.yaml)
    #
    # Objectif:
    # - Par défaut: LLM désactivé => l'API est retrieval-only (safe, rapide)
    # - Si runtime_online.yaml met llm.enabled=true:
    #   - on tente d'initialiser un client LLM (OpenAI) et on le place dans app.state.llm
    #
    # Sécurité:
    # - La clé API ne doit pas être dans le YAML.
    # - Elle reste dans l'environnement (OPENAI_API_KEY), éventuellement chargée depuis un .env
    #   si llm.env_file est défini et python-dotenv est installé.
    # ------------------------------------------------------------------
    runtime_cfg = getattr(app.state, "runtime_cfg", {}) or {}
    llm_cfg = runtime_cfg.get("llm", {}) or {}

    # Si un llm est passé explicitement à create_app(), il reste prioritaire (override code).
    resolved_llm = llm

    if resolved_llm is None and llm_cfg.get("enabled") is True:
        provider = llm_cfg.get("provider", "openai")
        model = llm_cfg.get("model", "gpt-4o-mini")

        # Optionnel: charger un env_file (évite dépendance à un export manuel terminal)
        env_file = llm_cfg.get("env_file")
        if isinstance(env_file, str) and env_file.strip() and not os.getenv("OPENAI_API_KEY"):
            try:
                dotenv_load(env_file)
                logger.info("Variables d'environnement chargées depuis env_file=%s", env_file)
            except Exception:
                # Si python-dotenv n'est pas installé, on n'échoue pas ici.
                # On échouera plus bas uniquement si la clé est indispensable.
                logger.warning(
                    "Impossible de charger env_file=%s (python-dotenv absent ou erreur). "
                    "OPENAI_API_KEY doit être présent dans l'environnement.",
                    env_file,
                )

        if provider == "openai":
            # Import local pour éviter d'imposer la dépendance si LLM désactivé
            try:
                from langchain_openai import ChatOpenAI  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "LLM activé (llm.enabled=true) mais 'langchain-openai' n'est pas installé. "
                    "Installe-le via: pip install langchain-openai"
                ) from e

            # Vérification de la clé: indispensable si on active OpenAI
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "LLM activé (llm.enabled=true) mais OPENAI_API_KEY est absent. "
                    "Définis OPENAI_API_KEY dans l'environnement ou via llm.env_file."
                )

            # Initialisation du LLM (temp=0 pour des réponses plus déterministes)
            resolved_llm = ChatOpenAI(model=model, temperature=0)
            logger.info("LLM OpenAI initialisé (model=%s).", model)

        else:
            raise RuntimeError(
                f"LLM activé mais provider non supporté pour l'instant: {provider}. "
                "Valeurs supportées: openai"
            )

    # Le LLM résolu (None si disabled) est stocké dans l'état de l'app
    app.state.llm = resolved_llm


    logger.info("API RAG initialisée: corpus=%s, retriever=%s", corpus_path, default_retriever)

    # ------------------------------------------------------------------
    # Warmup au démarrage (évite le cold-start au premier /rag/invoke)
    #
    # Objectif:
    # - Forcer le chargement du corpus parquet + construction du retriever
    #   dès le startup de FastAPI, au lieu d'attendre la première requête.
    #
    # Remarque:
    # - Ici on déclenche un "invoke" léger de la chain no-LLM (pas de coût OpenAI),
    #   uniquement pour provoquer read_parquet / index / cache.
    # ------------------------------------------------------------------
    """ mis en ause car doublon (non aligné avec 1ere requete)
    @app.on_event("startup")
    async def _warmup_runtime() -> None:
        try:
            t0 = time.perf_counter()

            # On crée une chain en mode "no_llm" juste pour déclencher le chargement du corpus/retriever.
            warm_chain = create_rag_chain_with_no_answer(
                corpus_path=app.state.corpus_path,
                retriever_type=app.state.default_retriever,
                k=1,
                filters=None,
                llm=None,
            )

            # Question bidon: on veut juste forcer la lecture parquet / construction retriever.
            _ = warm_chain.invoke({"question": "warmup"})

            logger.info(
                "Warmup runtime OK (%.1f ms).",
                (time.perf_counter() - t0) * 1000.0,
            )
        except Exception:
            logger.exception("Warmup runtime échoué.")
    """
    # ═══════════════════════════════════════════════════════════════════════
    # ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Health check de l'API."""
        # ------------------------------------------------------------------
        # Enrichissement /health: expose des infos runtime utiles au debug prod
        # sans divulguer de chemins internes (parquet_path, cache_dir, etc.)
        # ------------------------------------------------------------------
        runtime_cfg = getattr(app.state, "runtime_cfg", {}) or {}
        corpora = runtime_cfg.get("corpora", {}) or {}

        # Valeur par défaut (YAML): runtime.default_corpus_juridique
        runtime_default_cj = (runtime_cfg.get("runtime", {}) or {}).get("default_corpus_juridique")

        # Liste des corpus disponibles (LEGITEXT...) triée pour stabilité
        available_corpora = sorted([k for k in corpora.keys() if isinstance(k, str)])

        llm_cfg = (runtime_cfg.get("llm", {}) or {})
        llm_enabled_runtime = bool(llm_cfg.get("enabled") is True)
        llm_provider = llm_cfg.get("provider")
        llm_model = llm_cfg.get("model")
        llm_ready = bool(getattr(app.state, "llm", None) is not None)


        return HealthResponse(
            status="ok",
            version="0.1.0",
            retriever_types=["bm25", "dense", "hybrid"],

            # Langfuse "réellement configuré" uniquement si les clés existent
            langfuse_enabled=bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")),

            # Defaults runtime issus du YAML/ENV (appliqués si absents de la requête)
            default_retriever=getattr(app.state, "default_retriever", None),
            default_k=getattr(app.state, "default_k", None),

            # Sélection de corpus par défaut (clé LEGITEXT...)
            default_corpus_juridique=runtime_default_cj if isinstance(runtime_default_cj, str) else None,

            # Registry corpora (discovery)
            corpora_count=len(available_corpora),
            available_corpora=available_corpora,

            llm_enabled_runtime=llm_enabled_runtime,
            llm_ready=llm_ready,
            llm_provider=llm_provider if isinstance(llm_provider, str) else None,
            llm_model=llm_model if isinstance(llm_model, str) else None,

        )


    @app.get("/", include_in_schema=False)
    async def root():
        """
        Page d’accueil minimale.

        Objectif:
        - éviter les 404 quand un navigateur ouvre http://...:8000/
        - rediriger vers la doc interactive FastAPI
        """
        return RedirectResponse(url="/docs")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        """
        Favicon optionnel.

        Objectif:
        - éviter le bruit 404 dans les logs quand un navigateur tente /favicon.ico
        - on renvoie simplement 204 (pas de contenu)
        """
        return JSONResponse(status_code=204, content=None)

    @app.post("/rag/invoke", response_model=RAGResponse, tags=["RAG"])
    async def rag_invoke(request: RAGRequest, http_request: Request):
        """
        Exécute une requête RAG.

        Récupère les documents pertinents et génère une réponse
        (si un LLM est configuré).
        """
        # Chrono total : défini avant le try pour être disponible même si une étape échoue
        start_time = time.perf_counter()

        # Identifiant de corrélation : fourni par middleware (ou fallback sécurité)
        request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))

        # Timings détaillés : initialisés au même niveau d'indentation que start_time
        timings_ms = {}
        t_resolve_start = time.perf_counter()

        try:
            # Appliquer les defaults runtime si la requête n'a pas fourni ces champs
            retriever_type = request.retriever_type or app.state.default_retriever
            k = request.k or app.state.default_k or 10


            # ------------------------------------------------------------------
            # Sélection du corpus runtime (parquet) à partir de la métadonnée métier
            # `corpus_juridique` (valeur LEGITEXT...).
            #
            # - Si filters.corpus_juridique est fourni, on choisit le parquet associé
            #   dans runtime_online.yaml: corpora[LEGITEXT...].parquet_path
            # - Sinon, on utilise le corpus chargé par défaut au boot (app.state.corpus_path)
            # ------------------------------------------------------------------
            runtime_cfg = app.state.runtime_cfg or {}
            corpora = runtime_cfg.get("corpora", {}) or {}

            corpus_path = app.state.corpus_path  # défaut boot

            corpus_juridique = None
            if request.filters and isinstance(request.filters, dict):
                corpus_juridique = request.filters.get("corpus_juridique")

            if isinstance(corpus_juridique, str) and corpus_juridique.strip():
                entry = corpora.get(corpus_juridique)
                if not entry or not isinstance(entry, dict) or not entry.get("parquet_path"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"corpus_juridique inconnu ou non configuré dans runtime_online.yaml: {corpus_juridique}",
                    )
                corpus_path = str(Path(entry["parquet_path"]).expanduser().resolve())

            # ------------------------------------------------------------------
            # Fusion des filtres:
            # - request.filters est prioritaire (c'est la demande explicite du client)
            # - default_filters (YAML) complète uniquement les champs absents
            #
            # Exemple:
            # - Si le client n'envoie pas status_in, on injecte le défaut ["VIGUEUR"] du YAML.
            # - Si le client envoie status_in, on respecte sa valeur.
            # ------------------------------------------------------------------
            effective_filters = dict(request.filters or {})

            if isinstance(corpus_juridique, str) and corpus_juridique.strip():
                entry = corpora.get(corpus_juridique) or {}
                default_filters = entry.get("default_filters") or {}
                if isinstance(default_filters, dict):
                    for key, value in default_filters.items():
                        effective_filters.setdefault(key, value)

            timings_ms["resolve_runtime_ms"] = (time.perf_counter() - t_resolve_start) * 1000.0

            t_chain_create_start = time.perf_counter()

            # Indique si un LLM est réellement prêt côté runtime (utilisé pour tags/metadata)
            llm_enabled_runtime = bool(getattr(app.state, "llm", None) is not None)

            # Créer la chain avec les paramètres de la requête
            chain = create_rag_chain_with_no_answer(
                corpus_path=corpus_path,
                retriever_type=retriever_type,
                k=k,
                filters=effective_filters,
                llm=app.state.llm,
            )

            timings_ms["chain_create_ms"] = (time.perf_counter() - t_chain_create_start) * 1000.0

            # Configuration callbacks Langfuse (SDK v3):
            # - Le handler LangChain est sans arguments.
            # - Les attributs (session/user/tags/metadata) se propagent via un context manager.
            config = {}

            langfuse_handler = None
            langfuse_ctx = None

            if app.state.enable_langfuse:
                langfuse_handler = get_langfuse_handler()
                if langfuse_handler:
                    config["callbacks"] = [langfuse_handler]
                    langfuse_ctx = get_langfuse_context(
                        session_id="rag_app_api",
                        user_id="fred",
                        tags=[
                            "api",
                            "endpoint:/rag/invoke",
                            f"retriever:{retriever_type}",
                            f"corpus:{corpus_juridique or 'NA'}",
                            f"mode:{'llm' if llm_enabled_runtime else 'no_llm'}",
                        ],
                        metadata=sanitize_metadata({
                            "request_id": request_id,
                            "retriever_type": retriever_type,
                            "k": k,
                            "corpus_juridique": corpus_juridique or "NA",
                            "filters": effective_filters,
                        }),
                    )

            # Exécuter la chain dans le contexte Langfuse (ou un nullcontext si désactivé)
            if langfuse_ctx is None:
                from contextlib import nullcontext
                langfuse_ctx = nullcontext()

            t_chain_invoke_start = time.perf_counter()
            with langfuse_ctx:
                result = chain.invoke({"question": request.question}, config=config)

            timings_ms["chain_invoke_ms"] = (time.perf_counter() - t_chain_invoke_start) * 1000.0
            logger.info("DEBUG result_keys=%s", list(result.keys()) if isinstance(result, dict) else type(result))
            logger.info("DEBUG inner_timings=%s", result.get("timings_ms") if isinstance(result, dict) else None)
            # Fusion des timings internes calculés par la chain (retrieve/context/llm)
            if isinstance(result, dict):
                inner_timings = result.get("timings_ms")
                if isinstance(inner_timings, dict):
                    timings_ms.update(inner_timings)

            latency_ms = (time.perf_counter() - start_time) * 1000
            timings_ms["total_ms"] = latency_ms

            # Construire la réponse
            sources = [
                SourceInfo(
                    chunk_id=s.get("chunk_id"),
                    article=s.get("article"),
                    code=s.get("code"),
                    score=s.get("score"),
                )
                for s in result.get("sources", [])
            ]

            logger.info(json.dumps({
                "event": "rag_request",
                "request_id": request_id,
                "endpoint": "/rag/invoke",
                "question_len": len(request.question or ""),
                "resolved_corpus_juridique": corpus_juridique,
                "resolved_retriever_type": retriever_type,
                "resolved_k": k,
                "llm_enabled_runtime": bool(app.state.llm),
                "no_answer": bool(result.get("no_answer", False)),
                "no_answer_reason": result.get("no_answer_reason"),
                "nb_sources": len(result.get("sources", []) or []),
                "top_score": (result.get("sources", [{}])[0].get("score") if (result.get("sources") and isinstance(result.get("sources"), list)) else None),
                "timings_ms": timings_ms,
            }, ensure_ascii=False))

            return RAGResponse(
                request_id=request_id,
                question=request.question,
                answer=result.get("answer", ""),
                sources=sources,
                no_answer=result.get("no_answer", False),
                no_answer_reason=result.get("no_answer_reason"),
                confidence=result.get("confidence", 0.0),
                latency_ms=latency_ms,
                timings_ms=timings_ms,
                resolved_corpus_juridique=corpus_juridique,
                resolved_retriever_type=retriever_type,
                resolved_k=k,                
            )

        except Exception as e:
            logger.error("Erreur API RAG: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/rag/context", response_model=ContextResponse, tags=["RAG"])
    async def rag_context(request: RAGRequest, http_request: Request):
        """
        Récupère uniquement le contexte (sans génération LLM).

        Utile pour debug ou quand on veut utiliser son propre LLM.
        """
        start_time = time.perf_counter()
        request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))

        try:
            timings_ms = {}
            t_resolve_start = time.perf_counter()            
            # Appliquer les defaults runtime si la requête n'a pas fourni ces champs
            retriever_type = request.retriever_type or app.state.default_retriever
            k = request.k or app.state.default_k or 10

            # préparation du runtime
            # ------------------------------------------------------------------
            # Sélection du corpus runtime (parquet) à partir de la métadonnée métier
            # `corpus_juridique` (valeur LEGITEXT...).
            #
            # Pourquoi:
            # - `corpus_juridique` est la variable de référence (métadonnée au niveau chunk).
            # - On réutilise cette variable pour sélectionner le parquet sans introduire
            #   un nouvel identifiant (pas de doublon "corpus_id").
            #
            # Règle:
            # - Si request.filters contient corpus_juridique=LEGITEXT..., on résout le parquet via:
            #   runtime_online.yaml -> corpora[LEGITEXT...].parquet_path
            # - Sinon, on reste sur le corpus chargé par défaut au boot (app.state.corpus_path)
            # ------------------------------------------------------------------
            runtime_cfg = app.state.runtime_cfg or {}
            corpora = runtime_cfg.get("corpora", {}) or {}

            corpus_path = app.state.corpus_path  # défaut boot

            corpus_juridique = None
            if request.filters and isinstance(request.filters, dict):
                corpus_juridique = request.filters.get("corpus_juridique")

            if isinstance(corpus_juridique, str) and corpus_juridique.strip():
                entry = corpora.get(corpus_juridique)
                if not entry or not isinstance(entry, dict) or not entry.get("parquet_path"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"corpus_juridique inconnu ou non configuré dans runtime_online.yaml: {corpus_juridique}",
                    )
                # Résolution absolue: évite les problèmes de CWD et rend le runtime reproductible
                corpus_path = str(Path(entry["parquet_path"]).expanduser().resolve())

            # ------------------------------------------------------------------
            # Fusion des filtres:
            # - request.filters est prioritaire
            # - default_filters (YAML) complète uniquement les champs absents
            # ------------------------------------------------------------------
            effective_filters = dict(request.filters or {})

            if isinstance(corpus_juridique, str) and corpus_juridique.strip():
                entry = corpora.get(corpus_juridique) or {}
                default_filters = entry.get("default_filters") or {}
                if isinstance(default_filters, dict):
                    for key, value in default_filters.items():
                        effective_filters.setdefault(key, value)

            timings_ms["resolve_runtime_ms"] = (time.perf_counter() - t_resolve_start) * 1000.0

            t_chain_create_start = time.perf_counter()

            # ------------------------------------------------------------------
            # Cache de chain (évite de reconstruire la chain et de recharger le parquet)
            #
            # Pourquoi:
            # - create_rag_chain_with_no_answer(...) reconstruit le retriever et recharge le parquet
            #   si on la recrée à chaque requête.
            # - Ici on met en cache la chain par (corpus_path résolu, retriever_type, k, filtres, mode llm/no_llm).
            #
            # Attention:
            # - On utilise corpus_path (résolu via corpus_juridique), pas app.state.corpus_path.
            # ------------------------------------------------------------------
            llm_enabled_runtime = bool(getattr(app.state, "llm", None) is not None)

            chain_cache_key = (
                str(corpus_path),
                str(retriever_type),
                int(k),
                json.dumps(effective_filters, ensure_ascii=False, sort_keys=True),
                "llm" if llm_enabled_runtime else "no_llm",
            )

            chain = app.state._chain_cache.get(chain_cache_key)
            if chain is None:
                chain = create_rag_chain_with_no_answer(
                    corpus_path=corpus_path,
                    retriever_type=retriever_type,
                    k=k,
                    filters=effective_filters,
                    llm=app.state.llm,
                )
                app.state._chain_cache[chain_cache_key] = chain
                logger.info("Chain construite et mise en cache: %s", chain_cache_key)

            timings_ms["chain_create_ms"] = (time.perf_counter() - t_chain_create_start) * 1000.0

            t_chain_invoke_start = time.perf_counter()
            result = chain.invoke({"question": request.question})
            timings_ms["chain_invoke_ms"] = (time.perf_counter() - t_chain_invoke_start) * 1000.0
            logger.info("DEBUG result_keys=%s", list(result.keys()) if isinstance(result, dict) else type(result))
            logger.info("DEBUG inner_timings=%s", result.get("timings_ms") if isinstance(result, dict) else None)
            # Fusion des timings internes calculés par la chain (retrieve/context/llm)
            if isinstance(result, dict):
                inner_timings = result.get("timings_ms")
                if isinstance(inner_timings, dict):
                    timings_ms.update(inner_timings)

 

            latency_ms = (time.perf_counter() - start_time) * 1000
            timings_ms["total_ms"] = latency_ms

            logger.info(json.dumps({
                "event": "rag_request",
                "request_id": request_id,
                "endpoint": "/rag/context",
                "question_len": len(request.question or ""),
                "resolved_corpus_juridique": corpus_juridique,
                "resolved_retriever_type": retriever_type,
                "resolved_k": k,
                "timings_ms": timings_ms,
            }, ensure_ascii=False))

            return {
                "request_id": request_id,
                "question": request.question,
                "context": result.get("context", ""),
                "sources": result.get("sources", []),
                "latency_ms": latency_ms,
                "timings_ms": timings_ms,
                "resolved_corpus_juridique": corpus_juridique,
                "resolved_retriever_type": retriever_type,
                "resolved_k": k,                
            }

        except Exception as e:
            logger.error("Erreur API context: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))


    @app.get("/corpora", tags=["Config"], description="Liste les corpus configurés dans runtime_online.yaml.")
    def list_corpora() -> dict:
        """
        Expose la registry des corpora (runtime_online.yaml) côté client.

        Pourquoi:
        - Le runtime est piloté par YAML et peut contenir 1..N corpus.
        - Le client a besoin de connaître les valeurs valides de `filters.corpus_juridique`
        (ex: LEGITEXT...) sans devoir lire le fichier YAML.

        Contenu:
        - Renvoie une liste d'objets: {corpus_juridique, label}
        - On peut aussi renvoyer parquet_path en dev, mais en prod on évite souvent
        d'exposer les chemins internes. Ici on ne renvoie pas le chemin.
        """
        runtime_cfg = getattr(app.state, "runtime_cfg", {}) or {}
        corpora = runtime_cfg.get("corpora", {}) or {}

        items = []
        for corpus_juridique, entry in corpora.items():
            if not isinstance(entry, dict):
                continue
            items.append(
                {
                    "corpus_juridique": corpus_juridique,
                    "label": entry.get("label", ""),
                }
            )

        # Tri stable pour UX (par id)
        items.sort(key=lambda x: x["corpus_juridique"])

        return {"corpora": items}

    @app.get("/prompts", tags=["Config"])
    async def list_prompts():
        """Liste les prompts disponibles."""
        from rag_app.prompts import list_prompts
        return list_prompts()

    @app.get("/filters/presets", tags=["Config"])
    async def list_filter_presets():
        """Liste les presets de filtres disponibles."""
        from rag_app.policies.filters import MetierFilters
        return {
            "code_travail_vigueur": MetierFilters.code_travail_vigueur().to_dict(),
            "code_civil_vigueur": MetierFilters.code_civil_vigueur().to_dict(),
            "corpus_ids": MetierFilters.CORPUS_IDS,
        }

    startup_ms = (time.perf_counter() - t_create0) * 1000.0
    logger.info("create_app startup_ms=%.1f", startup_ms)

    return app


# Instance par défaut de l'app
app = create_app()
logger.info("module_import_ms=%.1f", (time.perf_counter() - _APP_IMPORT_T0) * 1000.0)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("RAG_API_PORT", "8000"))
    host = os.getenv("RAG_API_HOST", "0.0.0.0")

    logger.info("Démarrage API RAG sur %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
