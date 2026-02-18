# ⚖️ RAG juridique auditable — du datalake au runtime

Ce dépôt met en œuvre une approche **RAG juridique auditable, reproductible et traçable**, structurée en **deux mondes strictement séparés** :

- 🧱 **OFFLINE** : construire un socle de données *déterministe* + produire des **benchmarks IR** reproductibles (sans LLM)
- 🚀 **ONLINE** : servir des requêtes via une **API** (FastAPI) + orchestration LangChain (LCEL), avec **garde-fous** et **observabilité**

🎯 Objectif : pouvoir **remonter d’une réponse (ou d’un refus de réponse)** jusqu’aux **chunks sources**, au **corpus**, aux **paramètres**, et aux **traces d’exécution**.

---

## 🔎 1) Principe clé : auditabilité de bout en bout

L’auditabilité repose sur quelques invariants :

- 🧾 **Source de vérité = code + configs versionnées** (pas de “magie” non vérifiable)
- 🆔 **Identifiants stables** pour relier chaque résultat à sa provenance :
  - `doc_key` (document juridique stable)
  - `chunk_id` (fragment unique)
- 🔗 **Traçabilité chunk ↔ source** via un mapping et la propagation des métadonnées sur tout le pipeline
- ♻️ **Reproductibilité** : mêmes entrées + mêmes paramètres ⇒ mêmes sorties (datalake, index, scores, métriques)

---

## 🧱 2) OFFLINE : pipeline déterministe + benchmarks IR

Le monde OFFLINE transforme des **XML juridiques** en données exploitables pour le retrieval, puis évalue les méthodes via des métriques IR.

### 🏗️ Pipeline Bronze → Silver → Gold
- 🟤 **Bronze** : catalogage des XML (sans modifier le contenu)
- ⚪ **Silver** : parsing XML + normalisation + extraction (identifiants, dates, statuts) avec gestion d’erreurs
- 🟡 **Gold** : construction d’unités/chunks + propagation complète des métadonnées + export

### 📦 Formats de sortie (standardisés)
- 🧊 **Parquet (Gold)** : format colonne optimisé pour filtres et benchmarks in-memory
- 📚 **JSONL shardé** : projection dédiée à l’indexation et aux pipelines externes

### ⏳ Temporalité & statut (spécifique juridique)
Gestion explicite de :
- champs `valid_from`, `valid_to`, statuts (`VIGUEUR`, `MODIFIE`, `ABROGE`)
- filtrage par date de référence (`as_of`)
- comparabilité des benchmarks **uniquement si** les règles temporelles sont identiques

### 📊 Benchmarks de retrieval (sans LLM)
Comparaison reproductible de :
- 🧠 **BM25**, 🔎 **Dense**, 🧩 **Hybrid** (fusion BM25 + dense, dont RRF côté benchmarks)
- (optionnel) 🧰 **Elasticsearch** dans un cadre strictement benchmark

Métriques IR suivies : **Recall@k, MRR, nDCG@k**, avec **qrels versionnés**.

---

## 🚀 3) ONLINE : API + retrieval + garde-fous + observabilité

Le monde ONLINE expose un runtime RAG pouvant fonctionner en **retrieval-only** ou avec **LLM optionnel**.

### 🧩 API FastAPI
Endpoints principaux :
- `POST /rag/invoke` : retrieval + policy no-answer + contexte + (LLM optionnel)
- `POST /rag/context` : retrieval + contexte, sans génération
- `GET /health`, `GET /corpora`, docs OpenAPI

### 🧾 Config runtime auditable
- 🧷 **YAML versionné** (`runtime_online.yaml`) = défauts reproductibles
- 🌱 **ENV overrides** = adaptation prod/CI/container
- 🗂️ sélection d’un **corpus par requête** via `filters.corpus_juridique` (clé `LEGITEXT...`)

### ⚡ Retrieval & caches (robustesse/perf)
- retrievers **bm25 | dense | hybrid**
- 🗃️ **cache embeddings** (réutilisable OFFLINE/ONLINE)
- 🧠 cache en mémoire côté API (évite reload parquet / rebuild BM25)

### 🛑 Politique “no-answer” (anti-hallucination)
Refus explicite de la génération quand le contexte est insuffisant, avec :
- `no_answer=true`, raison, confiance, sources, timings

### 🪪 Observabilité (Langfuse)
- 1 trace par requête + spans (retrieve, dedup, no-answer, llm)
- tags/metadata : `request_id`, retriever, corpus, filtres, timings

---

## ✅ 4) Ce que “RAG auditable” signifie concrètement ici

Dans ce dépôt, “auditable” veut dire qu’on peut :

- 🔁 **rejouer** un run OFFLINE (mêmes XML + mêmes scripts/configs) et obtenir **les mêmes chunks / exports / métriques**
- 🧭 **expliquer** une réponse ONLINE via :
  - `request_id` (corrélation API ↔ logs ↔ traces)
  - `sources[]` (chunks + métadonnées + score)
  - règles de filtres (corpus, statut, temporalité `as_of`)
  - activation/désactivation du LLM (contrôlée)
- 🧯 **refuser proprement** (no-answer) plutôt que générer “à vide”

---

## 🧭 5) Repères rapides dans le dépôt

- 🧱 **OFFLINE** : scripts Bronze/Silver/Gold + exports + benchmarks + métriques IR
- 🚀 **ONLINE** : `src/rag_app/` (api, chains LCEL, retrievers, policies, observability) + `runtime_online.yaml`

