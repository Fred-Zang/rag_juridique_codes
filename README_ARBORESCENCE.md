# README_ARBORESCENCE.md
## Organisation physique du projet (référence disque)

⚠️ IMPORTANT
Ce document décrit UNIQUEMENT l’organisation PHYSIQUE des fichiers
et dossiers sur le disque.

Il ne décrit :
- ni l’architecture conceptuelle,
- ni le pipeline OFFLINE,
- ni l’orchestration RAG runtime.

Pour le raisonnement conceptuel, se référer à :
- README_TECHNIQUE.md (OFFLINE)
- README_RAG_PREP.md (préparation fonctionnelle)
- README_RAG_APP_RUNTIME.md (runtime)

---

## Routage documentaire (comment lire cette arborescence)
- Les dossiers sous `data_main/` sont décrits CONCEPTUELLEMENT dans :
  → README_TECHNIQUE.md
- Les scripts sous `rag_bench/src/datalake/` correspondent aux sections :
  → README_TECHNIQUE.md / Pipeline datalake
- Les scripts sous `rag_bench/src/rag_bench/` sont décrits dans :
  → README_TECHNIQUE.md / Benchmarks
- Le runtime réel est implémenté dans :
  → `rag_bench/src/rag_app/`
  et décrit dans :
  → README_RAG_APP_RUNTIME.md
Ce document ne doit JAMAIS être utilisé pour déduire une architecture logique.

## Conventions de chemins (portabilité)
- Les chemins sous `data_main/` sont des emplacements de stockage locaux (variables selon machine).
- Certains `source_path` peuvent être au format Windows (ex. `D:\...`) : à traiter comme métadonnée d’audit, pas comme chemin exploitable.
- Préférer les identifiants stables (`doc_key`, `chunk_id`) pour relier les objets.

## 1. Architecture des Dossiers

```
data_main/                          # Stockage PHYSIQUE des données (hors code)
    data/
        legifrance_download/        # Corpus compressé .tar.gz
        legifrance_extract_tmp/     # Corpus décompressé (archive XML)
        datalake_legifrance_v1/
            bronze/
                catalog/
                    bronze_files.parquet
                    bronze_stats.json

            silver/
                docs/            # Parquet (doc_key, body_text, source_path, ...)
                doc_versions/    # Parquet (doc_key, valid_from/to, status, ...)
                quality/         # Parquet (parse_ok, error_type, text_len, ...)

            gold/
                units/           # Parquet (unit_id, texte consolidé, ...)
                chunks/          # Parquet (chunk_id, chunk_text, source_path, ...)
                source_map/
                    source_map.parquet
                exports/
                    corpus_chunks_v1_jsonl_shards_gz/   # 256 shards gzip


rag_bench/                          # Code du projet : OFFLINE (datalake/bench) + runtime (rag_app/) — rôles séparés
    .env                            # environnement clé api provider + modèle
    configs/
        datalake_pipeline_v1.yaml       # Config pipeline data
        benchmark_cdtravail_generic.yaml # Config benchmark principal
        queries_cdtravail_v4.yaml       # Requêtes de test
        qrels_cdtravail_v4.yaml         # Ground truth annotations
        juridical_dictionary.yaml       # non connecté

    src/
        datalake/
            01_bronze_catalog.py        # Catalogage fichiers XML
            02_silver_parse.py          # Parsing XML enrichi
            03_gold_build.py            # Chunking + propagation métadonnées
            04_build_source_map.py      # Mapping source_id → metadata
            05_export_jsonl_sharded.py  # Export shardé pour ES
            utils_xml.py                # Parsers XML spécialisés

        rag_bench/
            run.py                      # Orchestrateur benchmark
            benchmark_bm25.py           # Benchmark BM25
            benchmark_dense.py          # Benchmark dense
            benchmark_elasticsearch.py  # Benchmark ES
            benchmark_hybrid_rrf.py     # Benchmark hybride RRF
            bm25.py                     # Module BM25
            dense.py                    # Module dense retrieval
            filtering.py                # Filtres corpus
            io_parquet.py               # I/O Parquet optimisé
            metrics.py                  # Calcul métriques IR
            paths.py                    # Gestion chemins
            utils.py                    # wrapper de timing
            logging_utils.py            # Configuration centralisée du logging pour rag_bench

        rag_app/                         # RAG runtime (orchestration réelle, hors benchmarks)
            api/
                app.py                   # Point d’entrée API (runtime)
                schemas.py               # Schémas (I/O) de l’API
                __init__.py
            chains/
                simple_rag.py            # Chaîne RAG de base
                with_no_answer.py        # Chaîne RAG avec politique no-answer
                __init__.py
            context/
                builder.py               # Construction du contexte LLM
                dedup.py                 # Déduplication des chunks
                __init__.py
            observability/
                langfuse_setup.py        # Initialisation Langfuse (si activé)
                run_manager.py           # Gestion des runs / traces
                __init__.py
            policies/
                filters.py               # Filtres (corpus, statuts, etc.)
                no_answer.py             # Politique no-answer
                temporal.py              # Politique temporelle (as_of)
                __init__.py
            prompts/
                registry.py              # Registre des prompts et versions
                templates.py             # Templates de prompts
                __init__.py
            retrievers/
                bm25_retriever.py        # Retriever BM25 (runtime)
                dense_retriever.py       # Retriever dense (runtime)
                hybrid_retriever.py      # Retriever hybride (runtime)
                __init__.py
            __init__.py


    tools/
        analyze_corpus_codes.py         # Stats corpus par code juridique
        analyze_results.py              # Analyse résultats benchmark
        auto_analyze.py                 # Analyse automatique runs
        inspect_chunk_xml_html.py       # Inspection chunks HTML
        inspect_dense_caches.py         # Debug cache embeddings
        match_cache_to_runs.py          # Association caches/runs
        validate_pipeline.py            # Validation pipeline
        create_qrels_v3.py              # Génère candidats qrels par consensus BM25+Dense+Hybrid
        validate_qrels_manual.py        # Interface interactive de validation manuelle
        analyze_queries.py              # Analyse couverture et difficulté des queries

    runs/                               # Résultats benchmark
        <timestamp>_<run_name>/
            config_used.yaml
            config_resolved.yaml
            *_results.jsonl
            metrics_*.json

    tutorials_runtime/                     # Préparation fonctionnelle RAG (phase RAG_PREP, hors runtime)
        01_check_dense_cache.py            # Vérifie présence/validité du cache dense (.npy + .meta.json) pour un cache_tag
        02_dense_retriever_query.py        # Test retrieval dense (invoke) : requête -> top-k docs + aperçu métadonnées
        03_context_builder.py              # Dédup + construction du contexte LLM (ContextBuilder) à partir des docs récupérés
        04_prompts_inspect.py              # Inventaire des prompts (types/versions) + inspection des variables d’entrée
        05_chain_no_llm.py                 # Mini pipeline RAG sans LLM : retrieval -> dédup -> contexte -> rendu prompt
        06_chain_with_openai.py            # Mini pipeline RAG avec OpenAI : retrieval -> contexte -> LLM + audit citations/preuves
        07_no_answer_policy_demo.py        # Démo NoAnswerPolicy (gating) : accepte/refuse selon couverture du contexte
        _common.py                         # Helper commun : fabrique un DenseRetriever aligné (parquet filtré + filtres + cache_tag)
        __init__.py                        # Package marker (permet `python -m tutorials_runtime.<script>`)

    data_sample/gold/code_route/           # Sous-corpus "Code de la route" pour tests rapides des scripts de tutorials_runtime/
        chunks_code_route_filtered.parquet # Parquet filtré (≈5089 chunks) : évite lecture du corpus complet (≈2,38M)

    .dense_cache/                          # Cache embeddings (plusieurs cache_tag possibles : corpus complet et/ou sous-corpus Code de la route)
        <cache_tag>.npy                    # Matrice embeddings (numpy) pour le corpus/filtre associé
        <cache_tag>.meta.json              # Métadonnées du cache (modèle, dimension, filtres, nb chunks, etc.)

---
## Clause de non-interprétation
Ce document ne décrit PAS :
- le flux logique du RAG,
- les dépendances fonctionnelles,
- l’ordre d’exécution conceptuel.
Il sert uniquement à localiser les fichiers sur le disque.
Toute analyse doit être effectuée à partir des READMEs conceptuels.


