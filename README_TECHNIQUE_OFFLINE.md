# README_TECHNIQUE_OFFLINE.md
## Documentation OFFLINE — Pipeline & Benchmarks RAG Juridique

---

## 0. Objet du document

Ce document est la **référence technique OFFLINE** du projet RAG juridique.

Il décrit exclusivement les composants réellement présents dans le dépôt
concernant :

- le corpus juridique,
- le pipeline datalake (bronze → silver → gold),
- les formats de données produits,
- les caches techniques,
- les benchmarks de retrieval,
- les métriques IR,
- Elasticsearch dans un contexte de benchmark.

Ce document se base **uniquement sur l’état actuel du code et des scripts**
présents dans le dépôt.  
Toute information non vérifiable dans le code a été volontairement exclue.

### Hors périmètre explicite

Ce document **ne couvre pas** :

- l’orchestration LangChain / LCEL / LangGraph,
- le RAG runtime,
- les prompts,
- les API (LangServe),
- l’observabilité runtime (LangFuse),
- toute logique de génération LLM.

Ces sujets sont documentés séparément et ne doivent être chargés que sur
demande explicite (voir `CLAUDE.md`).

### Source de vérité

Les **scripts et configurations présents dans le dépôt** constituent la source
de vérité unique.  
Les anciens READMEs servent uniquement de matériau brut et ne font pas foi.

---

## 1. Vue d’ensemble OFFLINE

Le monde OFFLINE a pour objectif de produire un **socle de données et
d’évaluation fiable**, indépendant de toute orchestration LLM.

Il repose sur trois piliers :

1. **Un pipeline datalake déterministe**  
   Transformation des sources juridiques brutes en données structurées,
   versionnées et exploitables pour le retrieval.

2. **Des formats de sortie standardisés**  
   Parquet pour le travail et les benchmarks in-memory, JSONL shardé pour
   l’indexation.

3. **Des benchmarks reproductibles**  
   Comparaison de méthodes de retrieval à l’aide de métriques IR et de jeux de
   qrels versionnés.

### Flux global OFFLINE

Sources juridiques (XML)
- Datalake Bronze
- Datalake Silver
- Datalake Gold (units / chunks)
- Index & caches
- Benchmarks
- Métriques IR


### Séparation stricte OFFLINE / ONLINE

Le monde OFFLINE :
- ne dépend d’aucun LLM,
- ne dépend d’aucune chaîne LangChain,
- ne produit aucune réponse utilisateur finale.

Il fournit uniquement :
- des données,
- des index,
- des scores,
- des métriques.

---

## 2. Corpus juridique

Cette section décrit le **corpus réellement utilisé** par le pipeline OFFLINE.

### 2.1 Source des données
- Sources institutionnelles fournies sous forme de fichiers XML.
- Structure XML de type Légifrance.
- Les scripts Bronze travaillent directement sur les XML bruts.

### 2.2 Types de documents exploités
- Articles juridiques normatifs.
- Documents issus de codes juridiques.
- Les documents non textuels ou non normatifs sont exclus au fil du pipeline.

### 2.3 Périmètre fonctionnel
Le corpus est conçu pour :
- le retrieval juridique,
- l’évaluation IR,
- la comparaison de méthodes de recherche.

Il n’est pas conçu pour :
- la génération directe de réponses,
- l’usage conversationnel,
- l’interprétation métier finale.

### 2.4 Volumétrie
- Corpus complet : potentiellement plusieurs millions de chunks.
- Sous-corpus filtrés possibles pour tests, benchmarks ciblés et tutoriels.
- La volumétrie exacte doit être vérifiée via les scripts et outils présents.

### 2.5 Identifiants et stabilité
- `doc_key` : identifiant stable de document juridique.
- `chunk_id` : identifiant unique de chunk.
- Métadonnées de version et de temporalité propagées par le pipeline.

### 2.6 Limites connues
- Qualité hétérogène des XML.
- Métadonnées parfois absentes ou incomplètes.
- Forte dépendance à la qualité du parsing Silver.
- Contraintes mémoire liées à la volumétrie.

---

## 3. Pipeline datalake

Le pipeline datalake transforme les sources juridiques brutes en données
structurées exploitables pour le retrieval et les benchmarks.

### 3.1 Principe général
- Transformation progressive et irréversible.
- Conservation maximale des métadonnées.
- Séparation stricte entre données brutes et enrichies.
- Formats colonne pour performance et auditabilité.

### 3.2 Bronze — Catalogage
- Script principal : `01_bronze_catalog.py`
- Entrées : XML juridiques bruts.
- Sorties : catalogue structuré (Parquet) et métadonnées minimales.
- Aucun contenu juridique n’est modifié.

### 3.3 Silver — Parsing & normalisation
- Script principal : `02_silver_parse.py`
- Parsing XML, normalisation du texte.
- Extraction des identifiants, dates, statuts.
- Gestion des erreurs et cas non conformes.

### 3.4 Gold — Unités & chunks
- Scripts principaux :
  - `03_gold_build.py`
  - `04_build_source_map.py`
  - `05_export_jsonl_sharded.py`
- Chunking contrôlé et propagation complète des métadonnées.
- Génération des identifiants finaux.
- Production Parquet Gold et exports JSONL.

### 3.5 Propriétés
- Reproductibilité.
- Auditabilité.
- Indépendance totale du runtime RAG.

---

## 4. Modèles de données

### 4.1 Identifiants
- `doc_key` : identifiant persistant du document.
- `chunk_id` : identifiant unique du chunk.
- Clés dérivées éventuelles (composites).

### 4.2 Objets OFFLINE
- Objets Silver : structurés, non chunkés.
- Objets Gold : unités et chunks.
- Objets d’export : JSONL shardé.

### 4.3 Schéma Gold (Parquet)
Familles de champs :
- texte,
- métadonnées juridiques,
- temporalité,
- traçabilité.

### 4.4 Export JSONL
- Projection du Parquet Gold.
- Utilisé pour indexation et benchmarks externes.
- Structure adaptée aux moteurs de recherche.

### 4.5 Auditabilité
- Mapping chunk ↔ source juridique.
- Possibilité de remonter d’un score à un document source.

---

## 5. Temporalité & statut juridique

### 5.1 Champs temporels
- `valid_from`
- `valid_to`
- Dates ouvertes possibles.
- Invariants : `valid_from <= valid_to` si présent.

### 5.2 Statuts juridiques
- Valeurs observées : `VIGUEUR`, `MODIFIE`, `ABROGE`.
- Propagation Silver → Gold.

### 5.3 Filtrage OFFLINE
- Filtrage par date de référence (`as_of`).
- Mode strict ou permissif selon les scripts.
- Gestion explicite des valeurs manquantes.

### 5.4 Cas limites
- Versions multiples d’un même article.
- Chunks sans information temporelle.
- Incohérences de dates.

### 5.5 Impact sur l’évaluation
- Les benchmarks sont comparables uniquement si les règles temporelles sont identiques.
- Les runs doivent tracer les paramètres temporels utilisés.

---

## 6. Indexation & formats de sortie

### 6.1 Parquet
- Format colonne.
- Pushdown filters.
- Lecture optimisée pour benchmarks.

### 6.2 JSONL shardé
- Sharding contrôlé.
- Compression éventuelle.
- Support des pipelines d’indexation.

---

## 7. Caches & performances

### 7.1 Cache dense
- Cache des embeddings.
- `cache_tag` pour l’identification.
- Règles d’invalidation liées au corpus et au modèle.

### 7.2 Autres caches
- Corpus filtrés.
- Résultats intermédiaires.
- Artefacts de benchmark.

---

## 8. Benchmarks de retrieval

### 8.1 Principes
- Évaluation comparative.
- Reproductibilité stricte.

### 8.2 BM25
- Implémentation in-memory.
- Paramètres documentés dans les configs.

### 8.3 Dense
- Modèles d’embeddings.
- Batch, device, cache.

### 8.4 Hybrid
- Fusion BM25 + Dense.
- RRF et paramètres associés.

### 8.5 Elasticsearch
- Utilisé à des fins de benchmark.
- Comparaison avec approches in-memory.

---

## 9. Configuration benchmark (YAML)

- Structure générale.
- Paramètres impactant les caches.
- Paramètres impactant la comparabilité.

---

## 10. Métriques IR

- Recall@k
- MRR
- nDCG@k

Limites et interprétation documentées.

---

## 11. Résultats & performances

- Ordres de grandeur.
- Temps d’exécution.
- Comparaisons typiques.
- Points de vigilance.

---

## 12. Qrels & évaluation

### 12.1 Format
- Qrels au format YAML.
- Labels explicites.

### 12.2 Workflow
- Génération.
- Validation.
- Versioning.

---

## 13. Outils d’analyse (`tools/`)

- Analyse corpus.
- Analyse résultats.
- Validation pipeline.
- Scripts utilitaires critiques.

---

## 14. Elasticsearch — benchmark

- Mapping.
- Index.
- Recréation.
- Commandes utiles.
- Différences avec in-memory.

---

## 15. Limites connues & points d’attention

- Volumétrie.
- Mémoire.
- Sensibilité aux filtres.
- Comparabilité des runs.

---

## 16. Glossaire & conventions

### Glossaire OFFLINE
- **Corpus** : ensemble des documents juridiques traités.
- **Document** : entité juridique source.
- **Article** : unité juridique textuelle.
- **Chunk** : fragment textuel issu du découpage Gold.
- **Bronze / Silver / Gold** : niveaux du pipeline datalake.
- **Retrieval** : recherche de documents.
- **Benchmark** : évaluation comparative des méthodes.
- **Qrels** : jugements de pertinence.
- **Run** : exécution benchmarkée.
- **as_of** : date de référence temporelle.

### Conventions
- Champs en `snake_case`.
- Statuts en MAJUSCULES.
- Scripts préfixés par ordre logique (`01_`, `02_`, …).
- Paramètres YAML explicitement nommés.

---

## 17. Références internes

### Scripts clés
- Bronze : `01_bronze_catalog.py`
- Silver : `02_silver_parse.py`
- Gold : `03_gold_build.py`, `04_build_source_map.py`, `05_export_jsonl_sharded.py`

### Dépendances croisées
- Les outputs Gold alimentent tous les benchmarks.
- Les paramètres YAML conditionnent caches et comparabilité.


---

## Clause finale

Ce document décrit exclusivement le **monde OFFLINE**.  
Toute logique runtime, orchestration LangChain, LLM ou API est volontairement
hors périmètre.

