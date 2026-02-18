# -*- coding: utf-8 -*-
"""
Benchmark Elasticsearch (BM25 / Dense kNN / Hybrid RRF) — script exécutable seul.

Objectif :
- Réutiliser la config YAML existante (benchmark_cdtravail.yaml)
- Réutiliser le filtrage métier/temporalité en amont
- Recalculer les embeddings en Python (dense.py) puis indexer dans Elasticsearch
- Produire des JSONL de résultats compatibles avec l’évaluation existante

Notes “projet” :
- On indexe des *chunks* (pas des documents entiers).
- On déduplique ensuite par doc_key (comme tes benchmarks dense/hybrid) 
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import yaml
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from rag_bench.filtering import filter_chunks
from rag_bench.ids import extract_doc_key
from rag_bench.io_parquet import read_chunks_from_parquet, read_jsonl  # ✅ Unifié avec run.py
from rag_bench.logging_utils import setup_logging
from rag_bench.core.dense import build_dense_embeddings
from rag_bench.common.cache_utils import resolve_cache_dir  # ✅ Depuis cache_utils comme run.py
from rag_bench.evaluation.evaluator import evaluate_retrieval  # ✅ Nouveau nom
from rag_bench.paths import get_repo_root, resolve_path

# ---------------------------
# Helpers config / paths
# ---------------------------

def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _noneify(x: Any) -> Any:
    """Convertit les chaînes 'None'/'null' (issues de YAML) en None Python."""
    if isinstance(x, str) and x.strip().lower() in {"none", "null", ""}:
        return None
    return x


# ---------------------------
# Elasticsearch index / mapping
# ---------------------------

def ensure_index(
    es: Elasticsearch,
    index_name: str,
    *,
    dims: Optional[int],
    text_field: str,
    force_recreate: bool,
    logger,
) -> None:
    """
    Crée l’index s’il n’existe pas.
    Si force_recreate=True : supprime et recrée (utile au 1er test).
    """
    exists = es.indices.exists(index=index_name)

    if exists and force_recreate:
        logger.warning("Suppression index ES existant: %s", index_name)
        es.indices.delete(index=index_name, ignore_unavailable=True)
        exists = False

    if exists:
        logger.info("Index ES déjà présent (réutilisé) : %s", index_name)
        return

    # dims peut être None (mode bm25 pur). On évite le %d qui plante avec None.
    logger.info("Création index ES : %s (dims=%s)", index_name, str(dims))

    # Mapping minimal :
    # - text_field: text (BM25)
    # - embedding: dense_vector (cosine) uniquement si dims est fourni
    #
    # Important :
    # - en mode bm25 pur, on peut créer un index sans champ embedding.
    # - si on veut ensuite passer en dense/hybrid, il faudra recréer l’index (mapping incompatible).


    body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                # Identifiants
                "doc_id": {"type": "keyword"},
                "doc_key": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "version_key": {"type": "keyword"},
                "unit_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},

                # Métadonnées structurelles
                "doc_type": {"type": "keyword"},
                "corpus_juridique": {"type": "keyword"},
                "article_num": {"type": "keyword"},
                "nature": {"type": "keyword"},

                # Texte
                "titre": {"type": "text"},
                "code_titre": {"type": "text"},
                "text": {"type": "text"},

                # Temporel (noms ES historiques, mappés depuis valid_from/valid_to/status)
                "date_debut": {"type": "date", "format": "strict_date_optional_time||yyyy-MM-dd"},
                "date_fin": {"type": "date", "format": "strict_date_optional_time||yyyy-MM-dd"},
                "etat": {"type": "keyword"},

                # Audit trail
                "source_path": {"type": "keyword"},

                # Enrichissements V3 (JSON strings)
                "liens": {"type": "text"},
                "struct_sections": {"type": "text"},
            }
        }
    }
    # On n'ajoute le champ embedding que si on a des embeddings à indexer (dense/hybrid).
    if dims is not None:
        body["mappings"]["properties"]["embedding"] = {
            "type": "dense_vector",
            "dims": int(dims),
            "index": True,
            "similarity": "cosine",
        }



    # Respecte le champ texte choisi dans la config
    if text_field != "text":
        body["mappings"]["properties"][text_field] = body["mappings"]["properties"].pop("text")

    es.indices.create(index=index_name, body=body)
    logger.info("Index ES créé : %s", index_name)


def _iter_bulk_actions(
    *,
    index_name: str,
    chunks: List[dict],
    embeddings,  # np.ndarray (N, D) ou None en mode bm25
    text_field: str,
) -> List[Dict[str, Any]]:
    """
    Construit les actions bulk. On indexe 1 doc ES par chunk.
    ID ES stable : <doc_key>#<chunk_idx>

    Mapping des champs gold → ES :
    - valid_from → date_debut
    - valid_to → date_fin
    - status → etat
    - code_titre → titre (fallback)
    """
    actions = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("meta") or {}

        # Identifiants : chunk_id est l'ID principal dans gold, doc_key pour dédup
        chunk_id = chunk.get("chunk_id") or meta.get("chunk_id")
        doc_key = meta.get("doc_key") or extract_doc_key(chunk_id) or f"NO_KEY_{i}"
        chunk_index = meta.get("chunk_index", i)
        es_id = f"{doc_key}#{chunk_index}"

        src = {
            # Identifiants
            "doc_id": chunk_id,  # Compat legacy : doc_id = chunk_id
            "doc_key": doc_key,
            "chunk_id": chunk_id,
            "version_key": meta.get("version_key"),
            "unit_id": meta.get("unit_id"),
            "chunk_index": chunk_index,

            # Métadonnées structurelles
            "doc_type": chunk.get("doc_type") or meta.get("doc_type"),
            "corpus_juridique": meta.get("corpus_juridique"),
            "article_num": meta.get("article_num"),
            "nature": meta.get("nature"),

            # Texte : code_titre comme titre, sinon vide
            "titre": meta.get("code_titre") or "",
            "code_titre": meta.get("code_titre"),
            text_field: chunk.get(text_field) or chunk.get("text") or "",

            # Temporel : MAPPING valid_from/valid_to/status → date_debut/date_fin/etat
            "date_debut": meta.get("valid_from"),
            "date_fin": meta.get("valid_to"),
            "etat": meta.get("status"),

            # Audit trail
            "source_path": meta.get("source_path"),

            # Enrichissements V3 (JSON strings)
            "liens": meta.get("liens"),
            "struct_sections": meta.get("struct_sections"),
        }

        # En mode bm25, embeddings=None : on n'indexe pas le champ embedding.
        if embeddings is not None:
            src["embedding"] = embeddings[i].tolist()

        actions.append({"_op_type": "index", "_index": index_name, "_id": es_id, "_source": src})
    return actions


def index_if_needed(
    es: Elasticsearch,
    *,
    index_name: str,
    chunks: List[dict],
    embeddings,
    text_field: str,
    logger,
) -> None:
    """
    Indexation “safe” :
    - On réutilise l’index, mais on n’essaie pas d’être trop intelligent.
    - Si l’index est vide : on bulk.
    - Sinon : on considère qu’il est déjà prêt.
    """
    count = es.count(index=index_name).get("count", 0)
    if count > 0:
        logger.info("Index ES non vide (count=%d). On suppose qu’il est déjà indexé.", count)
        return

    logger.info("Index ES vide -> bulk indexation (N=%d chunks)", len(chunks))
    actions = _iter_bulk_actions(index_name=index_name, chunks=chunks, embeddings=embeddings, text_field=text_field)

    t0 = time.perf_counter()
    es2 = es.options(request_timeout=180)
    ok, _ = bulk(es2, actions)

    logger.info("Bulk OK=%s | %.3fs", ok, time.perf_counter() - t0)

    es.indices.refresh(index=index_name)


# ---------------------------
# Retrieval ES
# ---------------------------

def es_bm25_search(
    es: Elasticsearch,
    *,
    index_name: str,
    query_text: str,
    k: int,
    text_field: str,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Renvoie [(es_id, score, source)] trié par score.
    """
    res = es.search(
        index=index_name,
        size=k,
        query={"match": {text_field: {"query": query_text}}},
    )
    hits = res.get("hits", {}).get("hits", []) or []
    return [(h["_id"], float(h["_score"] or 0.0), h["_source"]) for h in hits]


def es_dense_knn_search(
    es: Elasticsearch,
    *,
    index_name: str,
    query_vector: List[float],
    k: int,
    num_candidates: int,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    kNN vectoriel ES.
    """
    res = es.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": max(num_candidates, k),
        },
        _source=True,
        size=k,
    )
    hits = res.get("hits", {}).get("hits", []) or []
    return [(h["_id"], float(h["_score"] or 0.0), h["_source"]) for h in hits]


def rrf_fuse_ids(
    *,
    bm25_ids: List[str],
    dense_ids: List[str],
    rrf_k: int,
) -> List[Tuple[str, float]]:
    """
    Variante RRF sur des IDs ES (strings) au lieu d’indices.
    Score(doc) = Σ 1 / (rrf_k + rank)
    """
    fused: Dict[str, float] = {}

    for rank, _id in enumerate(bm25_ids, start=1):
        fused[_id] = fused.get(_id, 0.0) + 1.0 / (rrf_k + rank)

    for rank, _id in enumerate(dense_ids, start=1):
        fused[_id] = fused.get(_id, 0.0) + 1.0 / (rrf_k + rank)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def dedup_by_doc_key(
    ranked: List[Tuple[str, float, Dict[str, Any]]],
    *,
    k: int,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Déduplication “juridique” : on ne garde qu’un seul chunk par doc_key.
    """
    seen = set()
    out = []
    for _id, score, src in ranked:
        dk = (src.get("doc_key") or "").upper()
        if not dk or dk in seen:
            continue
        seen.add(dk)
        out.append((_id, score, src))
        if len(out) >= k:
            break
    return out


# ---------------------------
# Main script
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Chemin YAML benchmark (ex: configs/benchmark_cdtravail.yaml)")
    ap.add_argument("--es-host", default="http://localhost:9200", help="URL Elasticsearch")
    ap.add_argument("--es-index", default=None, help="Nom d’index ES (sinon dérivé du run_name)")
    ap.add_argument("--recreate-index", action="store_true", help="Supprime/recrée l’index ES")
    ap.add_argument(
        "--run-dir",
        default=None,
        help="(Optionnel) Dossier de run à utiliser (ex: runs/20260109_193133_benchmark_cdtravail). "
             "Si fourni, le script n'en crée pas un nouveau."
    )
    ap.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default=None, help="(Optionnel) Force le mode ES.")
    ap.add_argument("--skip-metrics", action="store_true", help="(Optionnel) N'écrit pas metrics.json (run.py s'en charge).")
    args = ap.parse_args()

    repo_root = get_repo_root()
    config_path = Path(args.config).expanduser().resolve()

    with open(str(config_path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    run_name = cfg.get("run", {}).get("run_name", "benchmark_es")
    out_dir_raw = cfg.get("run", {}).get("output_dir", "runs")
    out_dir = resolve_path(out_dir_raw, repo_root=repo_root, config_path=config_path)
    if not out_dir:
        raise ValueError("run.output_dir manquant ou invalide après résolution.")

    # Si run_dir est fourni (orchestration via run.py), on l'utilise tel quel.
    # Sinon, run autonome suffixé "_elasticsearch".
    if args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
    else:
        run_dir = os.path.join(out_dir, f"{_now_tag()}_{run_name}_elasticsearch")
 
 

    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logging(run_dir=run_dir, level=cfg.get("logging", {}).get("level", "INFO"))

    # Résolution des chemins du YAML (relatifs à la config)
    # ✅ Compat YAML canonique : queries.file avec fallback legacy bm25_benchmark.queries_file
    corpus_jsonl = resolve_path(cfg.get("data", {}).get("corpus_jsonl"), repo_root=repo_root, config_path=config_path)
    queries_file_raw = cfg.get("queries", {}).get("file") or cfg.get("bm25_benchmark", {}).get("queries_file")
    queries_file = resolve_path(queries_file_raw, repo_root=repo_root, config_path=config_path)
    qrels_file = resolve_path(cfg.get("evaluation", {}).get("qrels_file"), repo_root=repo_root, config_path=config_path)

    logger.info("Paths resolved | corpus=%s | queries=%s | qrels=%s", corpus_jsonl, queries_file, qrels_file)

    retrieval = cfg.get("retrieval", {}) or {}

    # Priorité à --mode si fourni, sinon fallback YAML.
    mode = (args.mode or retrieval.get("mode") or retrieval.get("retriever") or "hybrid")
    mode = str(mode).lower().strip()
    if mode not in {"bm25", "dense", "hybrid"}:
        raise ValueError(f"Mode ES invalide: {mode} (attendu: bm25 | dense | hybrid)")

    k = int(retrieval.get("k", cfg.get("bm25_benchmark", {}).get("k", 10)))
    bm25_shortlist = int(retrieval.get("bm25_shortlist", 200))
    rrf_k = int(retrieval.get("rrf_k", 60))

    # ✅ Compat YAML canonique : retrieval.text_field avec fallback legacy bm25_benchmark.text_field
    text_field = cfg.get("retrieval", {}).get("text_field") or cfg.get("bm25_benchmark", {}).get("text_field", "text")
    preview_chars = int(cfg.get("output", {}).get("preview_chars", 450))
    show_verbatim = bool(cfg.get("output", {}).get("show_verbatim", False))

    filters = cfg.get("filters", {}) or {}
    as_of = _noneify(filters.get("as_of"))
    strict_temporal = bool(filters.get("strict_temporal", False))
    doc_types = filters.get("doc_types")
    code_title_contains = _noneify(filters.get("code_title_contains"))
    corpus_juridique = _noneify(filters.get("corpus_juridique"))
    # status_in peut être string ou liste
    raw_status_in = filters.get("status_in")
    status_in = None
    if isinstance(raw_status_in, str) and raw_status_in.strip():
        status_in = [raw_status_in.strip()]
    elif isinstance(raw_status_in, list) and raw_status_in:
        status_in = [str(s).strip() for s in raw_status_in if str(s).strip()]

    dense_cfg = cfg.get("dense", {}) or {}
    embedding_model = dense_cfg.get("embedding_model")
    device = dense_cfg.get("device", "cpu")
    cache_dir_raw = dense_cfg.get("cache_dir")

    # Résolution portable du cache_dir (relatif -> ancré sur la racine projet),
    # puis validation par resolve_cache_dir (fallback dans runs/<run>/cache_dense si None).
    cache_dir_resolved = resolve_path(cache_dir_raw, repo_root=repo_root, config_path=config_path) if cache_dir_raw else None
    cache_dir = resolve_cache_dir(cache_dir_resolved, run_dir=run_dir, logger=logger)

    # En mode bm25 pur, on n’a pas besoin du modèle dense ni des embeddings.
    # On garde toutefois la possibilité de réutiliser un index existant (créé en dense/hybrid).
    need_dense = mode in {"dense", "hybrid"}

    # Cache tag stable (simple) : run_name + filtres principaux
    cache_tag = f"{run_name}|{doc_types}|{code_title_contains}|{corpus_juridique}|as_of={as_of}|strict={strict_temporal}"

    # Connexion ES
    es = Elasticsearch(args.es_host)
    index_name = args.es_index or f"rag_bench_{run_name}".lower()

    logger.info("Run dir: %s", run_dir)
    logger.info("Mode=%s | k=%d | index=%s | es=%s", mode, k, index_name, args.es_host)

    # --- Chargement corpus + filtres (même logique projet)
    # ✅ Auto-détection Parquet/JSONL comme run.py
    t0 = time.perf_counter()
    corpus_path = Path(corpus_jsonl)
    is_parquet = str(corpus_jsonl).endswith(".parquet") or corpus_path.is_dir()

    if is_parquet:
        logger.info("📦 Détection format Parquet: %s", corpus_jsonl)
        # Chercher source_map (optionnel)
        source_map_path = None
        if "gold/chunks" in str(corpus_path):
            gold_dir = str(corpus_path).split("gold/chunks")[0] + "gold"
            potential_source_map = Path(gold_dir) / "source_map" / "source_map.parquet"
            if potential_source_map.exists():
                source_map_path = str(potential_source_map)
                logger.info("✅ Source map trouvée: %s", source_map_path)

        chunks = read_chunks_from_parquet(
            parquet_path=corpus_jsonl,
            source_map_path=source_map_path,
            limit=cfg.get("data", {}).get("limit"),
            logger=logger,
            use_pandas=True
        )
    else:
        logger.info("📄 Détection format JSONL: %s", corpus_jsonl)
        chunks = read_jsonl(path=corpus_jsonl, limit=cfg.get("data", {}).get("limit"))

    logger.info("Corpus chargé: %d chunks | %.3fs", len(chunks), time.perf_counter() - t0)

    t0 = time.perf_counter()
    filter_stats = {}
    chunks_f = filter_chunks(
        chunks,
        doc_types=doc_types,
        code_title_contains=code_title_contains,
        corpus_juridique=corpus_juridique,
        status_in=status_in,
        as_of=as_of,
        strict_temporal=strict_temporal,
        stats=filter_stats,
        logger=logger,
    )
    logger.info("Stats filtrage : %s", filter_stats)

    logger.info("Corpus filtré: %d chunks | %.3fs", len(chunks_f), time.perf_counter() - t0)
    
    # --- Embeddings (uniquement si dense/hybrid)
    model = None
    doc_emb = None
    dims: Optional[int] = None
    if need_dense:
        if not embedding_model:
            raise ValueError("dense.embedding_model manquant dans la config YAML (requis pour dense/hybrid).")

        t0 = time.perf_counter()
        model, doc_emb = build_dense_embeddings(
            documents=chunks_f,
            model_name=embedding_model,
            cache_dir=cache_dir,
            cache_tag=cache_tag,
            text_field=text_field,
            device=device,
            logger=logger,
        )
        dims = int(doc_emb.shape[1])
        logger.info("Embeddings prêts: shape=%s | %.3fs", tuple(doc_emb.shape), time.perf_counter() - t0)
    else:
        logger.info("Mode bm25 -> embeddings non requis. Indexation possible sans champ embedding.")
        logger.info("Si tu veux ensuite dense/hybrid dans le même index, utilise --recreate-index.")
 

    ensure_index(
        es,
        index_name,
        dims=dims,
        text_field=text_field,
        force_recreate=bool(args.recreate_index),
        logger=logger,
    )

    index_if_needed(
        es,
        index_name=index_name,
        chunks=chunks_f,
        embeddings=doc_emb,
        text_field=text_field,
        logger=logger,
    )

    # --- Queries
    with open(queries_file, "r", encoding="utf-8") as f:
        qdata = yaml.safe_load(f) or {}
    queries = qdata.get("queries", []) or []
    logger.info("Queries chargées: %d | %s", len(queries), queries_file)

    # --- Run retrieval
    out_jsonl = os.path.join(run_dir, f"{mode}_results.jsonl" if mode != "hybrid" else "hybrid_results.jsonl")

    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for q in queries:
            qid = q["id"]
            qtext = q["text"]

            if mode == "bm25":
                hits = es_bm25_search(es, index_name=index_name, query_text=qtext, k=max(k * 5, 50), text_field=text_field)
                # Dédup doc_key puis top-k final
                ranked = dedup_by_doc_key([(i, s, src) for i, s, src in hits], k=k)

                for rank, (_id, score, src) in enumerate(ranked, start=1):
                    rec = {
                        "retriever": "bm25_es",
                        "query_id": qid,
                        "query_text": qtext,
                        "rank": rank,
                        "score": float(score),
                        # Identifiants
                        "doc_key": src.get("doc_key"),
                        "doc_id": src.get("doc_id"),
                        "chunk_id": src.get("chunk_id"),
                        # Métadonnées structurelles
                        "doc_type": src.get("doc_type"),
                        "corpus_juridique": src.get("corpus_juridique"),
                        "article_num": src.get("article_num"),
                        # Texte
                        "titre": src.get("titre"),
                        "code_titre": src.get("code_titre"),
                        "text_preview": (src.get(text_field) or "")[:preview_chars],
                        # Temporel
                        "as_of": as_of,
                        "date_debut": src.get("date_debut"),
                        "date_fin": src.get("date_fin"),
                        "etat": src.get("etat"),
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            elif mode == "dense":
                # En dense, le modèle est forcément chargé ci-dessus.
                # Ce garde-fou évite un crash silencieux si on modifie le code plus tard.
                assert model is not None, "model dense non initialisé (bug de flux)"
 
                # On encode la requête avec le même modèle que dense.py (cohérence embeddings)
                q_vec = model.encode([qtext], normalize_embeddings=True).astype("float32")[0].tolist()
                raw_hits = es_dense_knn_search(es, index_name=index_name, query_vector=q_vec, k=max(k * 20, 200), num_candidates=1000)
                ranked = dedup_by_doc_key([(i, s, src) for i, s, src in raw_hits], k=k)

                for rank, (_id, score, src) in enumerate(ranked, start=1):
                    rec = {
                        "retriever": "dense_es",
                        "query_id": qid,
                        "query_text": qtext,
                        "rank": rank,
                        "score": float(score),
                        # Identifiants
                        "doc_key": src.get("doc_key"),
                        "doc_id": src.get("doc_id"),
                        "chunk_id": src.get("chunk_id"),
                        # Métadonnées structurelles
                        "doc_type": src.get("doc_type"),
                        "corpus_juridique": src.get("corpus_juridique"),
                        "article_num": src.get("article_num"),
                        # Texte
                        "titre": src.get("titre"),
                        "code_titre": src.get("code_titre"),
                        "text_preview": (src.get(text_field) or "")[:preview_chars],
                        # Temporel
                        "as_of": as_of,
                        "date_debut": src.get("date_debut"),
                        "date_fin": src.get("date_fin"),
                        "etat": src.get("etat"),
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            else:
                # Hybrid RRF : BM25 shortlist + Dense shortlist + RRF côté Python (aligné benchmark_hybrid_rrf.py)
                assert model is not None, "model dense non initialisé (bug de flux)"
                bm25_hits = es_bm25_search(es, index_name=index_name, query_text=qtext, k=bm25_shortlist, text_field=text_field)
                bm25_ids = [i for i, _s, _src in bm25_hits]

                q_vec = model.encode([qtext], normalize_embeddings=True).astype("float32")[0].tolist()
                dense_hits = es_dense_knn_search(es, index_name=index_name, query_vector=q_vec, k=bm25_shortlist, num_candidates=max(2000, bm25_shortlist * 5))
                dense_ids = [i for i, _s, _src in dense_hits]

                fused = rrf_fuse_ids(bm25_ids=bm25_ids, dense_ids=dense_ids, rrf_k=rrf_k)

                # Reconstituer sources par id
                src_by_id: Dict[str, Dict[str, Any]] = {}

                for _id, sc, src in bm25_hits:
                    src_by_id.setdefault(_id, src)

                for _id, sc, src in dense_hits:
                    src_by_id.setdefault(_id, src)
                    
                    # on garde score dense séparément si besoin, mais ici on écrit score RRF au final

                fused_ranked = [(_id, float(rrf_score), src_by_id.get(_id, {})) for _id, rrf_score in fused]
                fused_ranked = dedup_by_doc_key(fused_ranked, k=k)

                if show_verbatim:
                    logger.info("HybridES | query_id=%s | fused=%d", qid, len(fused_ranked))

                for rank, (_id, rrf_score, src) in enumerate(fused_ranked, start=1):
                    rec = {
                        "retriever": "hybrid_rrf_es",
                        "query_id": qid,
                        "query_text": qtext,
                        "rank": rank,
                        "score": float(rrf_score),
                        # Identifiants
                        "doc_key": src.get("doc_key"),
                        "doc_id": src.get("doc_id"),
                        "chunk_id": src.get("chunk_id"),
                        # Métadonnées structurelles
                        "doc_type": src.get("doc_type"),
                        "corpus_juridique": src.get("corpus_juridique"),
                        "article_num": src.get("article_num"),
                        # Texte
                        "titre": src.get("titre"),
                        "code_titre": src.get("code_titre"),
                        "text_preview": (src.get(text_field) or "")[:preview_chars],
                        # Temporel
                        "as_of": as_of,
                        "date_debut": src.get("date_debut"),
                        "date_fin": src.get("date_fin"),
                        "etat": src.get("etat"),
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Résultats écrits: %s", out_jsonl)

    # --- Évaluation (via evaluate_retrieval, aligné sur run.py)
    if args.skip_metrics:
        logger.info("skip-metrics=1 -> metrics.json non écrit (orchestration run.py).")
        return

    if qrels_file and os.path.exists(qrels_file):
        eval_cfg = cfg.get("evaluation", {}) or {}
        metrics_list = eval_cfg.get("metrics", ["recall", "mrr", "ndcg"])
        k_values = eval_cfg.get("k_values", [1, 3, 5, 10, 20])
        metrics_path = os.path.join(run_dir, "metrics.json")

        report = evaluate_retrieval(
            results_file=out_jsonl,
            qrels_file=qrels_file,
            metrics=metrics_list,
            k_values=k_values,
            output_file=metrics_path,
            logger=logger,
        )

        logger.info("Metrics écrites: %s", metrics_path)
        logger.info("AVG: %s", report.get("average"))
    else:
        logger.warning("Pas de qrels_file trouvé -> évaluation sautée. qrels=%s", qrels_file)

  


if __name__ == "__main__":
    main()
