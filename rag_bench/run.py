# -*- coding: utf-8 -*-
"""
CLI `rag_bench` — exécution reproductible de benchmarks retrieval (BM25 / Dense / Hybride RRF).

python -m rag_bench.run --config configs/benchmark_cdtravail_generic.yaml


Ce script orchestre un run complet et traçable :
- charge une configuration YAML,
- résout les chemins (corpus, requêtes, qrels, cache) de manière robuste pour fonctionner
  quel que soit le dossier courant (Windows / Linux / WSL),
- charge un corpus chunké (JSONL) et applique les filtres,
- exécute les retrievers :
  - BM25 (baseline lexical),
  - Dense (embeddings SentenceTransformers + cos-sim),
  - Hybride (fusion BM25 + Dense via RRF sur les rangs),
- évalue les rankings via des qrels (Recall@k, MRR, nDCG@k),
- écrit les artefacts dans un dossier de run `runs/<timestamp>_<run_name>/`.

Traçabilité :
- `config_used.yaml` conserve la config telle que fournie (audit).
- `config_resolved.yaml` conserve la config avec chemins résolus (démo / reproductibilité).
- `result.json` résume les sorties produites.

Cache embeddings Dense :
- Si `dense.cache_dir` est fourni, les embeddings sont mis en cache dans ce dossier (mode performance,
  réutilisable entre runs).
- Sinon, fallback dans `runs/<run>/cache_dense/` (mode audit, run autonome mais recalcul au run suivant).
- Le cache est protégé par un garde-fou d’alignement : si le nombre d’embeddings ne correspond pas
  au corpus filtré, le cache est reconstruit pour éviter des résultats incohérents.

Note :
- La partie "query understanding" (dictionnaire) est prévue par la config mais volontairement non
  câblée à ce stade ; elle sera implémentée et évaluée dans une étape séparée.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import hashlib
from datetime import datetime
import time
from pathlib import Path
from typing import Any, Dict, List
import re

import yaml

from rag_bench.utils import TimingCollector
from rag_bench.logging_utils import setup_logging
from rag_bench.io_parquet import read_chunks_from_parquet, read_jsonl  # ✅ Ajouté read_jsonl
from rag_bench.filtering import filter_chunks
from rag_bench.benchmark_bm25 import run_bm25_benchmark
from rag_bench.benchmark_dense import run_dense_benchmark
from rag_bench.benchmark_hybrid_rrf import run_hybrid_rrf_benchmark
from rag_bench.evaluation.evaluator import evaluate_retrieval  # ✅ Nouveau nom
from rag_bench.common.cache_utils import resolve_cache_dir  # ✅ Depuis cache_utils
from rag_bench.paths import get_repo_root, resolve_path


_legitext_re = re.compile(r"(LEGITEXT\d{12,})")



def load_config(path: str) -> dict:
    """
    Charge un YAML en dict Python.
    Exposé volontairement (utile pour scripts de monitoring / mini-check).
    """
    return _load_yaml(path)


def _load_yaml(path: str) -> Dict[str, Any]:
    """Charge un fichier YAML en dict Python. => benchmark_*.yaml, queries_*.yaml et qrels_*.yaml
    via config.path"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _make_run_dir(output_dir: str, run_name: str) -> str:
    """Crée un dossier de run horodaté."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{ts}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _stable_hash(payload: Dict[str, Any]) -> str:
    """Empreinte stable (sert de tag pour le cache embeddings)."""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _load_queries(cfg, repo_root, config_path):
    """
    Charge les queries depuis le YAML.
    
    Supporte 2 structures :
    - Nouvelle : queries.file
    - Ancienne : bm25_benchmark.queries_file (rétrocompatibilité)
    """
    # Essayer nouvelle structure d'abord
    queries_file = cfg.get("queries", {}).get("file")
    
    # Fallback ancienne structure (rétrocompatibilité)
    if not queries_file:
        queries_file = cfg.get("bm25_benchmark", {}).get("queries_file")
    
    if not queries_file:
        raise ValueError(
            "Attendu : queries.file OU bm25_benchmark.queries_file dans la config YAML."
        )
    
    # Résoudre chemin (reste du code inchangé)
    queries_path = resolve_path(queries_file, repo_root=repo_root, config_path=config_path)  # ✅ CORRECTION ICI
    
    if not queries_path:
        raise ValueError(f"Impossible de résoudre le chemin queries_file : {queries_file}")
    
    queries_path = Path(queries_path)
    
    if not queries_path.exists():
        raise FileNotFoundError(f"Fichier queries introuvable : {queries_path}")
    
    with open(queries_path, "r", encoding="utf-8") as f:
        queries_data = yaml.safe_load(f)
    
    queries = queries_data.get("queries", [])
    
    if not queries:
        raise ValueError(f"Aucune query trouvée dans {queries_path}")
    
    return queries




def extract_corpus_juridique_from_source_path(source_path: str) -> str | None:
    """Extrait LEGITEXT... depuis source_path (fallback si absent du Parquet)."""
    if not source_path:
        return None
    m = _legitext_re.search(source_path)
    return m.group(1) if m else None


def _validate_io_and_filter(cfg: Dict[str, Any], *, corpus_jsonl_path: str, logger) -> List[Dict[str, Any]]:
    """
    Charge le corpus (Parquet OU JSONL) et applique les filtres.
    
    Détection automatique :
    - Si chemin finit par .parquet OU est un dossier → Parquet
    - Sinon → JSONL
    """
    data_cfg = cfg.get("data") or {}
    filters_cfg = cfg.get("filters") or {}

    # Détection du format
    corpus_path = Path(corpus_jsonl_path)
    is_parquet = corpus_jsonl_path.endswith(".parquet") or corpus_path.is_dir()
    
    if is_parquet:
        logger.info("📦 Détection format Parquet")
        logger.info("Chemin corpus: %s", corpus_jsonl_path)
        
        # Chercher source_map (optionnel mais recommandé)
        # utile pour associer chaque chunk à une source canonique (document / fichier / acte / version),
        # fournir un source_id stable pour des exports, audits, dédup, stats, ou UI (ex: regrouper les chunks par document).
        source_map_path = None
        
        # Stratégie 1 : gold/chunks/ → remonter à gold/ puis chercher source_map/
        if "gold/chunks" in str(corpus_path):
            gold_dir = str(corpus_path).split("gold/chunks")[0] + "gold"
            potential_source_map = Path(gold_dir) / "source_map" / "source_map.parquet"
            if potential_source_map.exists():
                source_map_path = str(potential_source_map)
                logger.info("✅ Source map trouvée: %s", source_map_path)
            else:
                logger.info("⚠️ Source map non trouvée (attendue: %s)", potential_source_map)
        
        # Charger avec io_parquet
        chunks = read_chunks_from_parquet(
            parquet_path=corpus_jsonl_path,
            source_map_path=source_map_path,
            limit=data_cfg.get("limit"),
            logger=logger,
            use_pandas=True  # Plus rapide pour benchmark
        )
    else:
        logger.info("📄 Détection format JSONL")
        logger.info("Chemin corpus: %s", corpus_jsonl_path)
        chunks = read_jsonl(path=corpus_jsonl_path, limit=data_cfg.get("limit"))

    # Enrichissement des métadonnées pour rendre les filtres robustes sur Parquet/JSONL.
    # Objectifs :
    # 1) Normaliser l'accès aux champs : certains loaders mettent doc_key/source_path/version_key au niveau racine,
    #    d'autres dans meta. On unifie en remplissant meta + en alimentant doc_id/doc_type.
    # 2) Extraire corpus_juridique=LEGITEXT... depuis source_path (fallback si absent)
    # 3) Inférer doc_type depuis doc_key (LEGIARTI... => article)

    for c in chunks:
        # meta peut être absent, None, ou déjà un dict
        meta = c.get("meta")
        meta = meta if isinstance(meta, dict) else {}

        # Certains loaders Parquet mettent les champs au niveau racine.
        # On normalise : si présent à la racine, on le copie dans meta.
        root_source_path = c.get("source_path")
        root_doc_key = c.get("doc_key")
        root_version_key = c.get("version_key")

        if not meta.get("source_path") and isinstance(root_source_path, str):
            meta["source_path"] = root_source_path
        if not meta.get("doc_key") and isinstance(root_doc_key, str):
            meta["doc_key"] = root_doc_key
        if not meta.get("version_key") and root_version_key is not None:
            meta["version_key"] = root_version_key

        # --- Récupération robuste des champs, meta OU racine ---

        # source_path désormais robuste (meta OU racine)
        source_path = (meta.get("source_path") or root_source_path or "")

        doc_key = meta.get("doc_key") or c.get("doc_key")
        version_key = meta.get("version_key") or c.get("version_key")
        chunk_text = meta.get("chunk_text") or c.get("chunk_text")

        # --- On recopie dans meta pour avoir un point d’accès unique ---
        if source_path:
            meta["source_path"] = source_path
            # Fallback: si corpus_juridique absent, extraire depuis source_path
            if not meta.get("corpus_juridique") and not c.get("corpus_juridique"):
                meta["corpus_juridique"] = extract_corpus_juridique_from_source_path(source_path)

            # doc_id absent en Parquet gold : on le remplit avec source_path pour l'affichage/debug
            if not c.get("doc_id"):
                c["doc_id"] = source_path

        if isinstance(doc_key, str) and doc_key:
            meta["doc_key"] = doc_key

            # doc_type est absent côté Parquet : on l’infère depuis doc_key pour que doc_types=["article"] marche.
            if not c.get("doc_type"):
                if doc_key.startswith("LEGIARTI"):
                    c["doc_type"] = "article"
                elif doc_key.startswith("LEGITEXT"):
                    c["doc_type"] = "text"

        if isinstance(version_key, str) and version_key:
            meta["version_key"] = version_key

        # --- Normalisation du champ texte ---
        # Les composants retrieval/embeddings peuvent attendre "text" selon ta config.
        # On remplit "text" si absent, en prenant chunk_text (racine ou meta).
        if not c.get("text") and isinstance(chunk_text, str) and chunk_text:
            c["text"] = chunk_text

        # --- Fallback temporel V3 : dériver valid_from/valid_to/status depuis version_key si besoin ---
        # Note: io_parquet.py charge déjà ces champs depuis le Parquet V3, ce fallback est pour anciens formats
        vk = meta.get("version_key") or root_version_key or c.get("version_key")
        if vk and (c.get("valid_from") is None or c.get("valid_to") is None or c.get("status") is None):
            parts = {}
            for item in str(vk).split("|"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    parts[k.strip()] = v.strip()

            # Écrire sous les nouveaux noms V3 au root du chunk
            c.setdefault("valid_from", parts.get("vf"))
            c.setdefault("valid_to", parts.get("vt"))
            c.setdefault("status", parts.get("st"))
            # Compatibilité : aussi dans meta pour anciens scripts
            meta.setdefault("valid_from", parts.get("vf"))
            meta.setdefault("valid_to", parts.get("vt"))
            meta.setdefault("status", parts.get("st"))

        c["meta"] = meta


    logger.info("Nombre total de chunks chargés : %d", len(chunks))

    # Filtrage (identique pour Parquet et JSONL)
    filter_stats = {}

    # Corpus juridique (ex: "LEGITEXT000006072050" pour Code du Travail)
    corpus_juridique = filters_cfg.get("corpus_juridique")

    # status_in peut être une string "VIGUEUR" ou une liste YAML ["VIGUEUR","MODIFIE"]
    raw_status_in = filters_cfg.get("status_in")
    status_in = None
    if isinstance(raw_status_in, str) and raw_status_in.strip():
        status_in = [raw_status_in.strip()]
    elif isinstance(raw_status_in, list) and raw_status_in:
        status_in = [str(s).strip() for s in raw_status_in if str(s).strip()]

    filtered = filter_chunks(
        chunks,
        doc_types=filters_cfg.get("doc_types"),
        code_title_contains=filters_cfg.get("code_title_contains"),
        corpus_juridique=corpus_juridique,
        status_in=status_in,
        as_of=filters_cfg.get("as_of"),
        strict_temporal=bool(filters_cfg.get("strict_temporal", False)),
        stats=filter_stats,
        logger=logger,
    )

    # Si tu veux un run rapide mais cohérent, on limite APRÈS filtrage.
    limit_after = data_cfg.get("limit_after_filter")
    if isinstance(limit_after, int) and limit_after > 0:
        filtered = filtered[:limit_after]

    logger.info("Stats filtrage : %s", filter_stats)
    logger.info("Nombre de chunks après filtrage : %d", len(filtered))

    for i, c in enumerate(filtered[:3]):
        meta = c.get("meta") or {}
        logger.info(
            "Exemple %d | doc_type=%s | titre=%s | doc_id=%s",
            i + 1, c.get("doc_type"), meta.get("titre"), c.get("doc_id")
        )

    return filtered


def _evaluate_and_write_metrics(
    *,
    run_dir: str,
    eval_cfg: Dict[str, Any],
    qrels_path: str,
    results_file: str,
    output_name: str,
    logger,
    cache_tag: Optional[str] = None,
) -> None:
    """Évalue un fichier de résultats JSONL via qrels."""
    from rag_bench.evaluation.evaluator import evaluate_retrieval
    
    # Sécurité : ne jamais crasher si le fichier de résultats n'existe pas
    if not os.path.exists(results_file):
        logger.warning("Évaluation %s ignorée : fichier absent (%s)", output_name, results_file)
        return

    # Récupérer config évaluation
    metrics = eval_cfg.get("metrics", ["recall", "mrr", "ndcg"])
    k_values = eval_cfg.get("k_values", [1, 3, 5, 10, 20])
    
    # Fichier de sortie métriques
    out_path = os.path.join(run_dir, f"metrics_{output_name}.json")
    
    logger.info("Évaluation %s: qrels=%s, k_values=%s", output_name, qrels_path, k_values)

    # Évaluation avec nouvelle signature
    report = evaluate_retrieval(
        results_file=results_file,
        qrels_file=qrels_path,
        metrics=metrics,
        k_values=k_values,
        output_file=out_path,
        logger=logger
    )

    # Ajouter cache_tag au rapport si fourni
    if cache_tag:
        report["cache_tag"] = cache_tag
        
        # Ré-écrire le fichier avec cache_tag
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("✅ Métriques %s écrites : %s", output_name, out_path)
    
    # Log métriques principales (depuis average)
    avg = report.get("average", {})
    if isinstance(avg, dict):
        for metric_name, value in avg.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric_name}: {value:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--config", required=True, help="Chemin vers un fichier YAML de configuration.")
    return p.parse_args()


def main() -> None:   # on charge d'abord les requetes puis les données ici ? à inverser si oui
    args = parse_args()
    
    t_setup0 = time.perf_counter()

    repo_root = get_repo_root()
    config_path = Path(args.config).expanduser().resolve()

    cfg = _load_yaml(str(config_path))

    # Résolution robuste du dossier de sortie du run : si le YAML contient "runs",
    # on l’ancre sur la racine du projet au lieu de dépendre du dossier courant.
    output_dir_raw = cfg["run"]["output_dir"]
    output_dir = resolve_path(output_dir_raw, repo_root=repo_root, config_path=config_path)
    if not output_dir:
        raise ValueError("run.output_dir manquant ou invalide après résolution.")

    run_name = cfg["run"]["run_name"]
    run_dir = _make_run_dir(output_dir, run_name)

    # Sections YAML
    bm25_cfg = cfg.get("bm25_benchmark") or {}
    retrieval_cfg = cfg.get("retrieval") or {}
    dense_cfg = cfg.get("dense") or {}
    output_cfg = cfg.get("output") or {}
    eval_cfg = cfg.get("evaluation") or {}
    data_cfg = cfg.get("data") or {}
    filters_cfg = cfg.get("filters") or {}
    # Date de référence juridique (audit + propagation dans les résultats)
    as_of = filters_cfg.get("as_of")


    # Paramètres retrieval
    # Le champ texte doit venir de retrieval.text_field (car dense/hybrid en dépendent),
    # et non de bm25_benchmark (qui est spécifique BM25).
    # Fallback: bm25_benchmark.text_field puis "text".
    text_field = str(retrieval_cfg.get("text_field", bm25_cfg.get("text_field", "text")))

    k = int(retrieval_cfg.get("k", bm25_cfg.get("k", 10)))
    retriever = str(retrieval_cfg.get("retriever", "bm25")).lower().strip()
    if retriever not in {"bm25", "dense", "hybrid", "elasticsearch"}:
        raise ValueError(f"retrieval.retriever invalide: {retriever} (attendu: bm25 | dense | hybrid | elasticsearch)")

    # Device dense (cpu par défaut)
    dense_device = str(dense_cfg.get("device", "cpu")).lower().strip()

    # Logger
    log_cfg = cfg.get("logging", {}) or {}
    logger = setup_logging(run_dir=run_dir, level=log_cfg.get("level", "INFO"))

    logger.info("Run démarré | run_dir=%s", run_dir)
    logger.info("Timing | 0) setup avant logger | %.3fs", time.perf_counter() - t_setup0)
    logger.info("Config chargée depuis : %s", str(config_path))
    
    # lancement du timer
    timer = TimingCollector(logger=logger, run_dir=run_dir, enabled=True)

    try:
        # Sauvegarde config brute (audit)
        with timer.timed("1) config_used.yaml"):
            cfg_path_copy = os.path.join(run_dir, "config_used.yaml")
            with open(cfg_path_copy, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            logger.info("Config sauvegardée : %s", cfg_path_copy)

        # Copie profonde de la config pour produire config_resolved.yaml (audit/reproductibilité)
        # On évite de modifier cfg (la config brute) et on prépare une version "résolue".
        cfg_resolved = copy.deepcopy(cfg)

        # Résolution des chemins (corpus, qrels, cache) pour rendre le run rejouable indépendamment du CWD
        corpus_jsonl = resolve_path(data_cfg.get("corpus_jsonl"), repo_root=repo_root, config_path=config_path)
        qrels_path = resolve_path(eval_cfg.get("qrels_file"), repo_root=repo_root, config_path=config_path)

        cache_dir_raw = dense_cfg.get("cache_dir")
        cache_dir_resolved = (
            resolve_path(cache_dir_raw, repo_root=repo_root, config_path=config_path) if cache_dir_raw else None
        )

        if not corpus_jsonl:
            raise ValueError("data.corpus_jsonl manquant ou invalide après résolution.")

        # Cache tag stable (même tag si corpus+filtres+limit+text_field identiques) / inclure le modèle (et le device) dans le cache_tag
        cache_tag = _stable_hash(
            {
                "corpus_jsonl": str(corpus_jsonl),
                "limit": data_cfg.get("limit"),
                "filters": filters_cfg,
                "text_field": text_field,
                # Le cache d'embeddings dépend du modèle : sinon on peut charger un .npy incompatible.
                "dense_model": dense_cfg.get("model_name") or dense_cfg.get("embedding_model") or dense_cfg.get("model"),
                # Le device / mode GPU peut aussi changer la reproductibilité (optionnel mais sain).
                "dense_device": dense_cfg.get("device"),
                "gpu_optimized": dense_cfg.get("use_gpu_optimized"),
            }
        )


        # Archive des entrées résolues (en STRING) pour sérialisation YAML sans erreur
        if qrels_path:
            cfg_resolved.setdefault("evaluation", {})["qrels_file"] = str(qrels_path)

        if cache_dir_raw:
            cfg_resolved.setdefault("dense", {})["cache_dir"] = str(cache_dir_resolved)

        cfg_resolved.setdefault("data", {})["corpus_jsonl"] = str(corpus_jsonl)
        cfg_resolved.setdefault("dense", {})["cache_tag"] = cache_tag

        # Cache dir dense réellement utilisé :
        # - si dense.cache_dir est fourni -> réutilisable entre runs
        # - sinon fallback dans runs/<run>/cache_dense
        resolved_cache_dir = resolve_cache_dir(cache_dir_resolved, run_dir=run_dir, logger=logger)

        # On trace le cache effectif réellement utilisé en STRING (resolve_cache_dir renvoie un Path)
        cfg_resolved.setdefault("dense", {})["resolved_cache_dir"] = str(resolved_cache_dir)

        # On archive aussi le chemin des requêtes résolu (utile si YAML contient un relatif)
        queries_file_raw = (cfg.get("bm25_benchmark") or {}).get("queries_file")
        queries_file_resolved = (
            resolve_path(queries_file_raw, repo_root=repo_root, config_path=config_path) if queries_file_raw else None
        )
        if queries_file_resolved:
            cfg_resolved.setdefault("bm25_benchmark", {})["queries_file"] = str(queries_file_resolved)

        # Maintenant que tous les chemins critiques sont résolus, on écrit UNE seule fois la config réellement exécutée
        config_resolved_path = os.path.join(run_dir, "config_resolved.yaml")
        with open(config_resolved_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_resolved, f, sort_keys=False, allow_unicode=True)
        logger.info("Config résolue sauvegardée : %s", config_resolved_path)

        # Chargement des requêtes (nécessaire pour BM25/Dense/Hybride)
        with timer.timed("2) chargement requêtes"):
            queries = _load_queries(cfg, repo_root=repo_root, config_path=config_path)

        # Chargement corpus + filtrage (Parquet ou JSONL auto-détecté)
        # corpus_jsonl est déjà résolu ci-dessus (string/Path), on passe une string.
        with timer.timed("3) chargement + filtrage"):
            filtered_chunks = _validate_io_and_filter(cfg, corpus_jsonl_path=str(corpus_jsonl), logger=logger)

        logger.info(
            "Préparation terminée | chunks_filtrés=%d | requêtes=%d | retriever=%s | k=%d",
            len(filtered_chunks),
            len(queries),
            retriever,
            k,
        )

        # ----------------------------
        # Backend Elasticsearch (orchestration)
        # ----------------------------
        if retriever == "elasticsearch":
            # Le backend = elasticsearch, mais l'algo = bm25|dense|hybrid
            es_mode = str(retrieval_cfg.get("mode", "hybrid")).lower().strip()
            if es_mode not in {"bm25", "dense", "hybrid"}:
                raise ValueError(f"retrieval.mode invalide: {es_mode} (attendu: bm25 | dense | hybrid)")

            es_cfg = retrieval_cfg.get("elasticsearch") or {}
            es_host = str(es_cfg.get("host", "http://localhost:9200")).strip()
            es_index = str(es_cfg.get("index", f"rag_bench_{run_name}".lower())).strip()
            recreate_index = bool(es_cfg.get("recreate_index", False))

            logger.info("Backend Elasticsearch | host=%s | index=%s | mode=%s | recreate=%s", es_host, es_index, es_mode, recreate_index)
            logger.info("Pré-requis : le conteneur Elasticsearch doit être démarré avant ce run.")

            cmd = [
                sys.executable, "-m", "rag_bench.benchmark_elasticsearch",
                "--config", str(config_path),
                "--es-host", es_host,
                "--es-index", es_index,
                "--run-dir", run_dir,
                "--mode", es_mode,
                "--skip-metrics",
            ]
            if recreate_index:
                cmd.append("--recreate-index")

            with timer.timed("4) Elasticsearch benchmark"):
                logger.info("Exec: %s", " ".join(cmd))
                proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
                if proc.returncode != 0:
                    logger.error("ES stdout:\n%s", proc.stdout)
                    logger.error("ES stderr:\n%s", proc.stderr)
                    raise RuntimeError(f"Benchmark Elasticsearch a échoué (code={proc.returncode}).")
                if proc.stdout.strip():
                    logger.info("ES stdout:\n%s", proc.stdout.strip())

            # Fichier résultats attendu (contrat identique à tes autres backends)
            results_file = os.path.join(run_dir, "hybrid_results.jsonl" if es_mode == "hybrid" else f"{es_mode}_results.jsonl")
            if not os.path.exists(results_file):
                raise FileNotFoundError(f"Résultats Elasticsearch introuvables: {results_file}")

            # Évaluation via les qrels (mêmes métriques que run.py)
            if eval_cfg and qrels_path:
                with timer.timed("5) évaluation qrels"):
                    _evaluate_and_write_metrics(
                        run_dir=run_dir,
                        eval_cfg=eval_cfg,
                        qrels_path=qrels_path,
                        results_file=results_file,
                        output_name=("hybrid" if es_mode == "hybrid" else es_mode),
                        logger=logger,
                        cache_tag=cache_tag,
                    )

            result = {
                "status": "OK",
                "run_dir": run_dir,
                "retriever": "elasticsearch",
                "mode": es_mode,
                "elasticsearch": {"host": es_host, "index": es_index, "recreate_index": recreate_index},
                "cache_tag": cache_tag,  # ✅ AJOUTÉ
                "cache_dir_used": str(resolved_cache_dir),  # ✅ AJOUTÉ                
                "outputs": {
                    "results": os.path.basename(results_file),
                    "metrics": f"metrics_{'hybrid' if es_mode == 'hybrid' else es_mode}.json" if (eval_cfg and qrels_path) else None,
                },
            }
            with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)


            logger.info(json.dumps(result, ensure_ascii=False, indent=2))
            return

        # BM25
        # On ne lance BM25 que si c'est demandé (bm25/hybrid) ou si on est en backend elasticsearch.
        # Pour un run dense pur, ça évite de tokeniser/indexer 2,3M chunks et de geler la machine.
        if retriever in {"bm25", "hybrid"}:
            bm25_k = int(bm25_cfg.get("k", 10))
            logger.info("BM25 benchmark multi-queries (%d requêtes) | k=%d", len(queries), bm25_k)
            with timer.timed("4) BM25 benchmark"):
                run_bm25_benchmark(
                    chunks=filtered_chunks,
                    queries=queries,
                    run_dir=run_dir,
                    k=bm25_k,
                    text_field=text_field,
                    as_of=as_of,
                    cache_tag=None,  # ✅ BM25 n'utilise pas de cache
                )
            logger.info("BM25 benchmark terminé")
        else:
            logger.info("BM25 ignoré (retriever=%s)", retriever)

        # Dense
        if retriever in {"dense", "hybrid"}:
            embedding_model = dense_cfg.get("embedding_model")
            if not embedding_model:
                raise ValueError("dense.embedding_model manquant dans la config YAML.")

            # Détection version GPU optimisée
            use_gpu_optimized = dense_cfg.get("use_gpu_optimized", False)
            batch_size = dense_cfg.get("batch_size", 64 if dense_device == "cuda" else 32)
            
            logger.info(
                "Dense benchmark (%d requêtes) | k=%d | model=%s | cache_tag=%s | device=%s | gpu_optimized=%s | batch_size=%d",
                len(queries), k, embedding_model, cache_tag, dense_device, use_gpu_optimized, batch_size
            )
            
            with timer.timed("5) Dense benchmark"):
                if use_gpu_optimized:
                    # Version GPU optimisée avec timer détaillé
                    run_dense_benchmark(
                        chunks=filtered_chunks,
                        queries=queries,
                        run_dir=run_dir,
                        k=k,
                        text_field=text_field,
                        embedding_model=embedding_model,
                        cache_dir=resolved_cache_dir,
                        cache_tag=cache_tag,
                        device=dense_device,
                        batch_size=batch_size,
                        show_verbatim=bool(output_cfg.get("show_verbatim", False)),
                        preview_chars=int(output_cfg.get("preview_chars", 450)),
                        logger=logger,
                        as_of=as_of,
                    )
                else:
                    # Version CPU classique
                    run_dense_benchmark(
                        chunks=filtered_chunks,
                        queries=queries,
                        run_dir=run_dir,
                        k=k,
                        text_field=text_field,
                        embedding_model=embedding_model,
                        cache_dir=resolved_cache_dir,
                        cache_tag=cache_tag,
                        device=dense_device,
                        batch_size=batch_size,
                        show_verbatim=bool(output_cfg.get("show_verbatim", False)),
                        preview_chars=int(output_cfg.get("preview_chars", 450)),
                        logger=logger,
                        as_of=as_of,
                    )
            logger.info("Dense benchmark terminé")

        # Hybride
        if retriever == "hybrid":
            bm25_shortlist = int(retrieval_cfg.get("bm25_shortlist", 200))
            rrf_k = int(retrieval_cfg.get("rrf_k", 60))
            embedding_model = dense_cfg.get("embedding_model")

            logger.info(
                "Hybride RRF benchmark (%d requêtes) | k=%d | bm25_shortlist=%d | rrf_k=%d",
                len(queries), k, bm25_shortlist, rrf_k
            )
            with timer.timed("6) Hybride RRF benchmark"):
                run_hybrid_rrf_benchmark(
                    chunks=filtered_chunks,
                    queries=queries,
                    run_dir=run_dir,
                    k=k,
                    text_field=text_field,
                    embedding_model=embedding_model,
                    cache_dir=resolved_cache_dir,
                    cache_tag=cache_tag,
                    device=dense_device,
                    bm25_shortlist=bm25_shortlist,
                    rrf_k=rrf_k,
                    show_verbatim=bool(output_cfg.get("show_verbatim", False)),
                    preview_chars=int(output_cfg.get("preview_chars", 450)),
                    logger=logger,
                    as_of=as_of,
                )
            logger.info("Hybride RRF benchmark terminé")

        # Évaluation
        if eval_cfg and qrels_path:
            with timer.timed("7) évaluation qrels"):
                bm25_path = os.path.join(run_dir, "bm25_results.jsonl")
                dense_path = os.path.join(run_dir, "dense_results.jsonl")
                hybrid_path = os.path.join(run_dir, "hybrid_results.jsonl")

                # BM25 : seulement si on l'a exécuté (bm25 ou hybrid) ET si le fichier existe
                if retriever in {"bm25", "hybrid"}:
                    if os.path.exists(bm25_path):
                        _evaluate_and_write_metrics(
                            run_dir=run_dir,
                            eval_cfg=eval_cfg,
                            qrels_path=qrels_path,
                            results_file=bm25_path,
                            output_name="bm25",
                            logger=logger,
                            cache_tag=None,  # ✅ BM25 n'utilise pas de cache
                        )
                    else:
                        logger.warning("Évaluation bm25 ignorée : fichier absent (%s)", bm25_path)

                # Dense : seulement si on l'a exécuté (dense ou hybrid) ET si le fichier existe
                if retriever in {"dense", "hybrid"}:
                    if os.path.exists(dense_path):
                        _evaluate_and_write_metrics(
                            run_dir=run_dir,
                            eval_cfg=eval_cfg,
                            qrels_path=qrels_path,
                            results_file=dense_path,
                            output_name="dense",
                            logger=logger,
                            cache_tag=cache_tag,  # ✅ AJOUTÉ
                        )
                    else:
                        logger.warning("Évaluation dense ignorée : fichier absent (%s)", dense_path)

                # Hybrid : seulement si on l'a exécuté ET si le fichier existe
                if retriever == "hybrid":
                    if os.path.exists(hybrid_path):
                        _evaluate_and_write_metrics(
                            run_dir=run_dir,
                            eval_cfg=eval_cfg,
                            qrels_path=qrels_path,
                            results_file=hybrid_path,
                            output_name="hybrid",
                            logger=logger,
                            cache_tag=cache_tag,  # ✅ AJOUTÉ
                        )
                    else:
                        logger.warning("Évaluation hybrid ignorée : fichier absent (%s)", hybrid_path)

        # Résumé des sorties produites (toujours écrit, même si pas de qrels)
        # On déclare les outputs uniquement si les fichiers existent réellement sur disque.
        bm25_results_path = os.path.join(run_dir, "bm25_results.jsonl")
        dense_results_path = os.path.join(run_dir, "dense_results.jsonl")
        hybrid_results_path = os.path.join(run_dir, "hybrid_results.jsonl")

        outputs = {
            "bm25": os.path.basename(bm25_results_path) if os.path.exists(bm25_results_path) else None,
            "dense": os.path.basename(dense_results_path) if os.path.exists(dense_results_path) else None,
            "hybrid": os.path.basename(hybrid_results_path) if os.path.exists(hybrid_results_path) else None,
        }

        result = {
            "status": "OK",
            "run_dir": run_dir,
            "retriever": retriever,
            "outputs": outputs,
            "cache_tag": cache_tag,  # ✅ AJOUT pour traçabilité
            "cache_dir_used": str(resolved_cache_dir),  # ✅ AJOUT (optionnel mais utile)
        }

        with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(json.dumps(result, ensure_ascii=False, indent=2))


    finally:
        # Toujours écrire les timings, même si une étape a crashé
        timer.write_json("timings.json")



if __name__ == "__main__":
    main()
