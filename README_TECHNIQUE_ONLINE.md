# README_TECHNIQUE_ONLINE.md
## Documentation ONLINE — Runtime RAG juridique (API + LangChain + Observabilité)

> Référence technique **ONLINE** (runtime) du projet RAG juridique.  
> Date de génération : 2026-02-13 (Europe/Paris)

---

## 0. Objet du document

Ce document est la **référence technique ONLINE** du projet.

Il décrit **uniquement l’état actuel** du runtime réellement présent dans le dépôt, à savoir :

- l’API **FastAPI** (`src/rag_app/api/`)
- l’orchestration **LangChain (LCEL)** (`src/rag_app/chains/`, `src/rag_app/context/`, `src/rag_app/prompts/`)
- le retrieval **BM25 / Dense / Hybrid** (`src/rag_app/retrievers/`)
- les **politiques runtime** (no-answer, filtres, temporalité) (`src/rag_app/policies/`)
- l’**observabilité** via **Langfuse** (`src/rag_app/observability/`)
- le pilotage par **YAML runtime** (`runtime_online.yaml`) + overrides ENV
- la résolution des chemins via `rag_bench/paths.py`

### Hors périmètre explicite

Ce document ne couvre pas :

- le pipeline **OFFLINE** (bronze/silver/gold, exports, benchmarks IR) : voir `README_TECHNIQUE_OFFLINE.md`
- l’industrialisation infra (Kubernetes, reverse-proxy, secrets management, CI/CD)
- la qualité juridique métier (validation par juristes, conformité, etc.)

### Source de vérité

La source de vérité est **le code** + la config `runtime_online.yaml`.
Les documents “plan / modifs” et brouillons servent uniquement d’historique, pas de référence.

---

## 1. Vue d’ensemble ONLINE

Le monde ONLINE a pour objectif de **servir une requête utilisateur** :

1) réception HTTP (`/rag/invoke` ou `/rag/context`)  
2) résolution des défauts runtime (retriever/k/corpus)  
3) retrieval (`bm25` / `dense` / `hybrid`)  
4) construction de contexte (dédup + mise en forme)  
5) politique **no-answer** (bloquer une génération risquée)  
6) génération (optionnelle) via LLM (OpenAI)  
7) réponse JSON + sources + timings + corrélation `request_id`

### Flux global ONLINE

Client
- FastAPI (request_id, validation Pydantic, endpoints)
- Résolution runtime (YAML/ENV + sélection de corpus)
- Chain LangChain (retrieve → policy → context → (LLM))
- Observabilité (Langfuse spans + metadata/tags)
- Réponse structurée (answer + sources + no_answer + timings)

---

## 2. Configuration runtime (runtime_online.yaml + ENV)

### 2.1 Principes

- **YAML = défaut versionné** (reproductibilité)
- **ENV = override prioritaire** (prod/CI/container)
- le YAML sert aussi de **registry corpora**, pour permettre de choisir un parquet **par requête**
  via `filters.corpus_juridique` (valeur `LEGITEXT...`) sans redémarrer l’API.

### 2.2 Structure du YAML (résumé)

- `runtime.default_corpus_juridique` : corpus par défaut (clé `LEGITEXT...`)
- `paths.cache_dense_dir` : cache embeddings (idéalement unique offline/online)
- `retrieval.default_retriever_type` : `bm25` | `dense` | `hybrid`
- `retrieval.default_k` : top-k par défaut
- `dense.*` : paramètres du retriever dense (modèle, device, etc.)
- `corpora.<LEGITEXT...>.parquet_path` : parquet à charger
- `corpora.<LEGITEXT...>.default_filters` (optionnel) : filtres par défaut pour ce corpus
- `llm.*` : activation et paramètres LLM (clé API **jamais** dans le YAML)

### 2.3 Variables d’environnement importantes

- `RAG_RUNTIME_CONFIG` : chemin vers `runtime_online.yaml` (prioritaire)
- `RAG_DEFAULT_RETRIEVER` : override du retriever par défaut
- `RAG_DEFAULT_K` : override du `k` par défaut
- `RAG_CORPUS_PATH` : parquet par défaut (si pas de sélection via `filters.corpus_juridique`)
- `RAG_BENCH_CACHE_DENSE_DIR` : override du cache dense (embeddings)
- `OPENAI_API_KEY` : clé OpenAI (obligatoire si `llm.enabled=true`)
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`/`LANGFUSE_BASE_URL` : Langfuse

### 2.4 Résolution des chemins (portabilité)

Le module `rag_bench/paths.py` centralise la résolution des chemins et les overrides ENV, afin d’éviter :
- chemins codés en dur,
- dépendance au dossier courant,
- problèmes Windows/WSL (chemins `D:\...`).

---

## 3. API FastAPI (`src/rag_app/api/`)

### 3.1 Endpoints

- `POST /rag/invoke`  
  Exécute la chain complète : retrieval + politique no-answer + contexte + (LLM optionnel).
- `POST /rag/context`  
  Exécute retrieval + construction de contexte (sans génération).
- `GET /health`  
  Diagnostic (état runtime, defaults, corpus, LLM, Langfuse).
- `GET /corpora`  
  Découverte des corpus configurés (`LEGITEXT...`) sans exposer des chemins sensibles.
- `GET /docs` / `GET /redoc`  
  Documentation OpenAPI.

### 3.2 Schémas de requête/réponse

**Requête** (`RAGRequest`) :
- `question: str`
- `filters: dict | null` (dont `corpus_juridique`, `status_in`, `as_of`, etc.)
- `k: int | null` (si absent → défaut runtime)
- `retriever_type: str | null` (si absent → défaut runtime)

**Réponse** (`RAGResponse`) :
- `request_id` (corrélation, identique à `X-Request-ID`)
- `answer`
- `sources[]` (métadonnées + score)
- `no_answer`, `no_answer_reason`, `confidence`
- `timings_ms` (détail) + `latency_ms` (global)

### 3.3 Corrélation (request_id)

L’API applique un middleware :
- si le client envoie `X-Request-ID`, nous le réutilisons,
- sinon nous générons un UUID,
- le `request_id` est renvoyé dans :
  - le header `X-Request-ID`
  - la réponse JSON.

Objectif : corréler **client → logs → traces Langfuse**.

### 3.4 Cache côté API

L’API maintient un cache de chaînes/retrievers en mémoire (par corpus / retriever / k / filtres) afin d’éviter :
- relecture du parquet,
- reconstruction d’index BM25,
- latences “explosives” en dev (reload) et en prod (cold start).

---

## 4. Chains LangChain (LCEL)

### 4.1 Variantes de chain

- `chains/simple_rag.py` : pipeline “retrieve → context → (LLM)”  
- `chains/with_no_answer.py` : ajoute une étape d’évaluation “doit-on répondre ?”

La chain est construite comme une suite d’étapes LCEL :
- `retrieve_step`
- `dedup_chunks` / construction du contexte
- `no_answer` (si activé)
- génération via LLM (optionnel)
- post-processing (sources + timings)

### 4.2 Format des sources

Chaque source est un chunk + métadonnées (selon disponibilité) :
- `chunk_id`
- identifiant article (si disponible)
- code / corpus
- score de pertinence

---

## 5. Retrieval (`src/rag_app/retrievers/`)

### 5.1 BM25 (in-memory)

- Avantages : simple, robuste, bon “baseline”.
- Risques : coût de construction d’index si on le reconstruit trop souvent → nécessité de cache.

### 5.2 Dense (embeddings)

- Avantages : capture la similarité sémantique.
- Points sensibles : device (CPU/GPU), batch, **cache embeddings** (réutilisation offline/online).

### 5.3 Hybrid

- Objectif : combiner BM25 (lexical) + dense (sémantique).
- Stratégie actuelle : “hybrid retriever” (fusion interne).
- Pour l’évaluation IR offline, une fusion RRF existe côté benchmarks ; online, on vise surtout la robustesse.

---

## 6. Politiques runtime (`src/rag_app/policies/`)

### 6.1 No-answer (anti-hallucination)

La politique no-answer vise à **bloquer** une génération lorsque le contexte est insuffisant.

Entrées typiques :
- `min_relevance`
- `min_documents`
- `coverage_threshold`

Sortie :
- `no_answer=true` + `no_answer_reason` + message explicite utilisateur.

### 6.2 Filtres métier

Les filtres sont la base de la sélection runtime (exemples) :
- `corpus_juridique = LEGITEXT...` (sélection du parquet)
- `status_in = ["VIGUEUR"]` (statut juridique)
- `as_of` (référence temporelle)

### 6.3 Temporalité

Objectif : ne pas mélanger des versions d’articles incompatibles (selon la date de référence).  
Le comportement exact dépend de la disponibilité des champs dans le parquet.

---

## 7. Prompts (`src/rag_app/prompts/`)

- Registry de prompts versionnés (nom, variables, template).
- Les prompts servent à :
  - cadrer le style de réponse,
  - imposer l’usage des citations `[n]`,
  - contraindre le modèle à ne répondre qu’avec le contexte fourni.

---

## 8. Observabilité (Langfuse)

### 8.1 Pourquoi Langfuse ici

Langfuse sert à :
- visualiser un **trace par requête**,
- découper en **spans** (retrieve, dedup, no-answer, llm),
- suivre :
  - latences (p95/p99)
  - tokens / coûts (LLM)
  - qualité proxy (no_answer_rate, nb_sources, top_score, etc.)

### 8.2 Stratégie d’instrumentation

- une trace par requête
- des tags “lisibles” :
  - `endpoint:/rag/invoke`
  - `retriever:bm25|dense|hybrid`
  - `corpus:<LEGITEXT...>`
  - `mode:llm|no_llm`
- metadata (valeurs scalaires ou sérialisées) :
  - `request_id`, `k`, `filters`, `timings_ms`, etc.

> Bon réflexe : forcer la sérialisation string des metadata pour éviter des champs rejetés côté Langfuse.

---

## 9. LLM (OpenAI) — activation contrôlée

### 9.1 Objectif

- **par défaut** : le runtime peut fonctionner en mode retrieval-only (pas de génération)
- en mode LLM : génération via `langchain-openai` (`ChatOpenAI`) si `OPENAI_API_KEY` est disponible

### 9.2 Recommandations de dev/prod

- **lazy import / lazy init** : éviter de payer des imports lourds au startup si le LLM est désactivé
- séparer clairement :
  - timings retrieval
  - timings LLM
  - timings post-processing

---

## 10. Points à améliorer (liste actionnable)

### 10.1 Robustesse & prod
- **CORS** : restreindre `allow_origins` en prod (éviter `*` avec credentials).
- durcir l’exposition d’infos diagnostics (`/health`, `/corpora`) selon environnement.

### 10.2 Portabilité config
- réduire les chemins absolus dans `runtime_online.yaml` :
  - privilégier variables d’environnement + chemins relatifs au repo.

### 10.3 Performance
- garantir que l’index BM25 et les objets lourds sont réellement **réutilisés** via le cache.
- séparer dans `timings_ms` :
  - `parquet_load_ms`
  - `index_build_ms`
  - `search_ms`
  - `cache_hit`

### 10.4 Qualité des sources
- dédup systématique et traçable :
  - `nb_sources_before_dedup`
  - `nb_sources_after_dedup`
- stabiliser les règles de tri et de score après dédup.

### 10.5 Qualité “RAG contract”
- renforcer les invariants :
  - citations présentes quand LLM ON
  - correspondance citations ↔ sources
  - “no-answer” déclenché de façon stable sur cas hors-sujet

---

## 11. Roadmap d’évolution (pour reprise dans quelques mois)

### 11.1 Remplacer FastAPI “maison” par LangServe : pourquoi et différences

**Ce que nous avons aujourd’hui (FastAPI)**  
- contrôle total des endpoints, middleware (request_id), erreurs, payloads
- mais nous gérons nous-mêmes :
  - validation + sérialisation
  - versioning endpoints
  - exposition “invoke/batch/stream” si nécessaire

**Ce que LangServe apporte**  
- exposition standard des Runnables LangChain en API :
  - endpoints cohérents (`/invoke`, `/batch`, `/stream`, `/stream_log` selon config)
  - compatibilité plus directe avec l’écosystème LangChain
- réduction du code “plomberie” FastAPI
- mais :
  - moins de liberté sur certains aspects (routing custom, schémas métier très spécifiques)
  - nécessite de cadrer proprement les modèles d’entrée/sortie

**Stratégie recommandée**  
- garder notre logique de chain (LCEL) inchangée
- encapsuler la chain dans LangServe
- conserver un “mince wrapper” FastAPI uniquement si nous avons des endpoints non-standards indispensables (ex: `/health` riche)

### 11.2 Remplacer Langfuse par LangSmith : pourquoi et différences

**Langfuse (actuel)**  
- open-source, self-hostable
- très bon pour tracing + coûts + dashboards, contrôle infra

**LangSmith**  
- outil LangChain (tracing + datasets + évaluation intégrée)
- intégration native avec LCEL, tests, “playground”, feedback loops
- dépendance plus forte à l’écosystème LangChain (et à l’offre SaaS selon choix)

**Décision pratique**  
- si priorité = self-host / contrôle infra → Langfuse reste excellent
- si priorité = boucle d’évaluation / datasets / tooling LangChain “end-to-end” → LangSmith peut réduire le travail

**Migration**  
- standardiser d’abord tags/metadata/spans
- isoler l’initialisation du tracer derrière une interface unique (adapter “LangfuseHandler” → “LangSmithTracer”)

### 11.3 Introduire LangGraph : quand ça vaut le coût

LangGraph devient utile quand la logique dépasse un pipeline linéaire, par exemple :
- routeur de requêtes (choix retriever / corpus / stratégie)
- branches conditionnelles (fallback BM25 → dense → hybrid)
- boucles (self-check citations, re-retrieval, query rewriting)
- politiques multi-étapes (garde-fous, re-ranking, validations)

**Idée de graphe minimal (futur)**  
- Node A : normalize / enrich query  
- Node B : retrieve (bm25/dense/hybrid)  
- Node C : dedup + context builder  
- Node D : no-answer gate  
- Node E : generate (LLM)  
- Node F : validate citations / format  
- Node G : response

---

## 12. Commandes de run (référence)

Les commandes opérationnelles sont maintenues dans : `RUNBOOK_COMMANDES_MAJ_2026-02-13_v3.md`  
Objectif : garder ce README ONLINE “stable” et déplacer les commandes fréquemment modifiées dans le runbook.

---

## 13. Glossaire (ONLINE)

- **Runtime ONLINE** : service qui reçoit des questions et renvoie des réponses (avec ou sans LLM).
- **Retriever** : composant qui récupère les chunks pertinents.
- **Context builder** : dédup + assemblage du contexte LLM.
- **No-answer** : règle qui empêche une réponse quand le contexte est insuffisant.
- **Trace / span** : instrumentation observabilité d’une requête (Langfuse/LangSmith).
- **k / top-k** : nombre de chunks récupérés au retrieval.
