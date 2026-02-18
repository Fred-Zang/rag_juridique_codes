#!/usr/bin/env python3
"""
Script d'analyse complète des résultats BM25/Dense/Hybrid

Usage:
    python tools/analyze_results.py runs/20260128_225746_cdtravail_hybrid/bm25_results.jsonl

    ou bien

    python tools/analyze_results.py   "$LAST_RUN/dense_results.jsonl"   --corpus ~/montage1/"-- Projet RAG Avocats --"/data_main/data/datalake_legifrance_v1/gold/chunks   --qrels configs/qrels_cdtravail_v2.yaml

Affiche pour chaque query:
- Top-k résultats récupérés
- Tous les champs (doc_key, dates validité, status, corpus_juridique, etc.)
- Texte complet formaté
- Comparaison avec qrels

Sauvegarde:
- Rapport HTML interactif dans le même dossier
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import os
import re
import html
from urllib.parse import quote

_LEGAL_ID_RE = re.compile(r"\b(?:LEGIARTI|LEGITEXT|LEGISCTA|JORFTEXT)\d{12,}\b")

def build_legifrance_url(identifier: str) -> str:
    """
    Construit une URL Legifrance raisonnable à partir d'un identifiant.
    - Pour LEGIARTI, on pointe directement la page article (souvent la plus pratique).
    - Pour le reste, on retombe sur une recherche (robuste).
    """
    if not isinstance(identifier, str) or not identifier:
        return "https://www.legifrance.gouv.fr"
    if identifier.startswith("LEGIARTI"):
        return f"https://www.legifrance.gouv.fr/codes/article_lc/{identifier}"
    return f"https://www.legifrance.gouv.fr/search/all?query={quote(identifier)}"


def build_xml_file_url(source_path) -> str | None:
    """
    Transforme un chemin local (Linux) en URL file://… ouvrable dans Chromium.
    On ignore explicitement les chemins Windows (D:\\...), qui sont de la métadonnée d'audit.
    """
    if not source_path:
        return None
    sp = str(source_path).strip()
    if not sp:
        return None
    # Chemins Windows -> on ne tente pas d'ouvrir (non portable)
    if re.match(r"^[A-Za-z]:\\", sp):
        return None
    try:
        p = Path(sp)
        if p.exists():
            return p.resolve().as_uri()
    except Exception:
        return None
    return None


def _get_row_for_doc_key(df_corpus_idx: pd.DataFrame, doc_key: str):
    """Récupère une ligne "doc_data" robuste même si l'index a des doublons."""
    row = df_corpus_idx.loc[doc_key]
    return row.iloc[0] if isinstance(row, pd.DataFrame) else row


def linkify_legal_ids(text_raw: str, df_corpus_idx: pd.DataFrame) -> str:
    """
    Échappe le texte, puis rend cliquables les IDs juridiques détectés.
    - Si l'ID existe comme doc_key dans le corpus et possède un source_path valide -> lien file:// vers le XML
    - Sinon -> lien Legifrance (recherche)
    """
    if not isinstance(text_raw, str):
        text_raw = "" if text_raw is None else str(text_raw)

    safe = html.escape(text_raw)

    def _repl(m: re.Match) -> str:
        ident = m.group(0)
        xml_url = None
        # On ne fait un lookup corpus que si l'ID ressemble à un doc_key (LEGIARTI...) et est présent
        try:
            if ident in df_corpus_idx.index:
                dd = _get_row_for_doc_key(df_corpus_idx, ident)
                xml_url = build_xml_file_url(dd.get("source_path"))
        except Exception:
            xml_url = None
        href = xml_url or build_legifrance_url(ident)
        # Style inline pour éviter de dépendre du CSS existant
        return (
            f'<a href="{href}" target="_blank" rel="noopener" '
            f'style="color: inherit; text-decoration: underline;">{ident}</a>'
        )

    safe = _LEGAL_ID_RE.sub(_repl, safe)
    return safe.replace("\n", "<br>")


def load_run_config(run_dir):
    """
    Charge config_resolved.yaml depuis le dossier du run.
    
    Args:
        run_dir: Dossier du run (ex: runs/20260122_120329_cdtravail_dense_gpu)
        
    Returns:
        dict: Config complète ou None si non trouvé
    """
    config_file = Path(run_dir) / "config_resolved.yaml"
    if not config_file.exists():
        print(f"⚠️ config_resolved.yaml non trouvé dans {run_dir}")
        return None
    
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        print(f"✅ Config chargée depuis {config_file}")
        # Debug: afficher les clés principales
        if config:
            print(f"   Clés config: {list(config.keys())}")
            if 'dense' in config:
                print(f"   Clés dense: {list(config['dense'].keys())}")
        return config
    except Exception as e:
        print(f"⚠️ Erreur lecture config_resolved.yaml: {e}")
        return None

def load_cache_metadata(cache_tag, cache_dir=".dense_cache", run_config=None, result_json=None):
    """
    Charge les métadonnées du cache dense depuis config_resolved.yaml et .dense_cache/<tag>.meta.json
    
    Priorité:
    1. config_resolved.yaml du run (source la plus fiable)
    2. .dense_cache/<tag>.meta.json (backup)
    3. .dense_cache/<tag>.npy (infos shape/taille)
    
    Args:
        cache_tag: Hash du cache (ex: "ec9d2607951f0adc")
        cache_dir: Dossier du cache
        run_config: Config du run (depuis config_resolved.yaml)
        
    Returns:
        dict: Métadonnées complètes du cache
    """
    if not cache_tag:
        # Sans cache_tag, on ne peut pas relier le run à un fichier .npy/.meta.json
        return None
    
    cache_info = {
        "cache_tag": cache_tag,
        "cache_dir": str(Path(cache_dir).absolute()) if cache_dir else "N/A",
    }
    
    # 1. Extraire infos depuis config_resolved.yaml (priorité haute)
    if run_config:
        print(f"   📋 Extraction depuis config_resolved.yaml...")
        dense_config = run_config.get("dense", {})
        data_config = run_config.get("data", {})
        filters_config = run_config.get("filters", {})
        
        # Infos du modèle
        cache_info["embedding_model"] = dense_config.get("embedding_model", "N/A")
        cache_info["device"] = dense_config.get("device", "N/A")
        cache_info["use_gpu_optimized"] = dense_config.get("use_gpu_optimized")
        cache_info["batch_size"] = dense_config.get("batch_size")
        
        # Infos du corpus
        cache_info["text_field"] = data_config.get("text_field", "text")
        cache_info["limit"] = data_config.get("limit")
        cache_info["limit_after_filter"] = data_config.get("limit_after_filter")
        
        # Filtres
        cache_info["filters"] = filters_config
        cache_info["doc_types"] = filters_config.get("doc_types", [])
        
         # Corpus source
         # Ici, ton YAML utilise "corpus_jsonl" (et parfois d'autres variantes selon les pipelines).
         # On tente plusieurs clés pour être robuste et éviter les "N/A" inutiles dans le rapport.
        corpus_any = (
            data_config.get("corpus_path")
            or data_config.get("corpus_parquet")
            or data_config.get("corpus_jsonl")
            or data_config.get("corpus")
        )
        cache_info["corpus_source"] = Path(corpus_any).name if corpus_any else "N/A"
          
        
        # Debug
        print(f"      Modèle: {cache_info['embedding_model']}")
        print(f"      Device: {cache_info['device']}")
        print(f"      Text field: {cache_info['text_field']}")
    else:
        print(f"   ⚠️ run_config est None, pas d'extraction depuis config_resolved.yaml")
    
    # 1b. Essayer result.json comme source alternative pour modèle/device
    if result_json and cache_info.get("embedding_model") == "N/A":
        print(f"   📋 Tentative extraction depuis result.json...")
        dense_result = result_json.get("dense", {})
        if dense_result:
            if cache_info.get("embedding_model") == "N/A":
                cache_info["embedding_model"] = dense_result.get("embedding_model", "N/A")
            if cache_info.get("device") == "N/A":
                cache_info["device"] = dense_result.get("device", "N/A")
            if cache_info.get("batch_size") is None:
                cache_info["batch_size"] = dense_result.get("batch_size")
            print(f"      Modèle (result.json): {cache_info['embedding_model']}")
            print(f"      Device (result.json): {cache_info['device']}")
    
    # 2. Lire métadonnées depuis .dense_cache/<tag>.meta.json (backup)
    if cache_dir:
        cache_dir_path = Path(cache_dir)
        if cache_dir_path.exists():
            meta_file = cache_dir_path / f"{cache_tag}.meta.json"
            npy_file = cache_dir_path / f"{cache_tag}.npy"
            
            cache_info["meta_file_exists"] = meta_file.exists()
            cache_info["npy_file_exists"] = npy_file.exists()
            
            # Lire .meta.json si pas déjà dans run_config
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                    
                    # Compléter avec .meta.json si manquant
                    if cache_info.get("embedding_model") == "N/A":
                        cache_info["embedding_model"] = meta.get("embedding_model", "N/A")
                    if cache_info.get("device") == "N/A":
                        cache_info["device"] = meta.get("device", "N/A")
                    if cache_info.get("text_field") == "N/A":
                        cache_info["text_field"] = meta.get("text_field", "N/A")
                        
                except Exception as e:
                    cache_info["meta_error"] = str(e)
            
            # 3. Lire infos du fichier NPY (shape, taille, dates)
            if npy_file.exists():
                try:
                    # Taille du fichier
                    file_size = npy_file.stat().st_size
                    cache_info["file_size_bytes"] = file_size
                    cache_info["file_size_mb"] = file_size / (1024 * 1024)
                    cache_info["file_size_gb"] = file_size / (1024 * 1024 * 1024)
                    
                    # Date de création/modification
                    cache_info["created"] = datetime.fromtimestamp(npy_file.stat().st_ctime).isoformat()
                    cache_info["modified"] = datetime.fromtimestamp(npy_file.stat().st_mtime).isoformat()
                    
                    # Shape des embeddings
                    # mmap_mode='r' : évite de charger l'intégralité des embeddings en RAM
                    embeddings = np.load(npy_file, mmap_mode="r")
                    cache_info["embeddings_shape"] = embeddings.shape
                    cache_info["num_embeddings"] = embeddings.shape[0]
                    cache_info["embedding_dim"] = embeddings.shape[1] if len(embeddings.shape) > 1 else None
                    
                except Exception as e:
                    cache_info["npy_error"] = str(e)
    
    # Normalisation : ne jamais retourner de "None" silencieux pour des champs attendus
    cache_info.setdefault("embedding_model", "N/A")
    cache_info.setdefault("device", "N/A")
    cache_info.setdefault("text_field", "text")
    return cache_info

def load_corpus(corpus_path):
    """Charge le corpus Parquet en DataFrame."""
    print(f"📦 Chargement corpus: {corpus_path}")
    table = pq.read_table(corpus_path)
    df = table.to_pandas()
    print(f"✅ {len(df):,} chunks chargés")
    return df

def load_results(results_file):
    """Charge les résultats JSONL."""
    print(f"📋 Chargement résultats: {results_file}")
    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))
    print(f"✅ {len(results)} résultats chargés")
    return results

def load_qrels(qrels_file):
    """Charge les qrels YAML (qid -> {doc_key: rel_score})."""
    if not qrels_file or not Path(qrels_file).exists():
        return {}
    
    print(f"📋 Chargement qrels: {qrels_file}")
    with open(qrels_file, "r", encoding="utf-8") as f:
        qrels = yaml.safe_load(f) or {}

    # Conserver les scores de pertinence (utile pour afficher QREL=... dans le HTML)
    qrels_map = {}
    for qid, docs in qrels.items():
        if isinstance(docs, dict):
            qrels_map[qid] = docs

    print(f"✅ {len(qrels_map)} queries avec qrels")
    return qrels_map

def load_run_metrics(run_dir: Path, results_file: Path):
    """
    Charge les métriques réelles du run (metrics_*.json) SANS recalcul.

    Convention de nommage:
    - bm25_results.jsonl   -> metrics_bm25.json
    - dense_results.jsonl  -> metrics_dense.json
    - hybrid_results.jsonl -> metrics_hybrid.json
    """
    stem = results_file.stem  # ex: "bm25_results"
    retriever = stem.split("_")[0]  # "bm25" | "dense" | "hybrid"

    metrics_path = run_dir / f"metrics_{retriever}.json"

    # Logs diagnostiques détaillés
    print("\n" + "=" * 80)
    print("🔎 DEBUG METRICS (source de vérité)")
    print(f"   run_dir        : {run_dir}")
    print(f"   results_file   : {results_file}")
    print(f"   results.stem   : {stem}")
    print(f"   retriever      : {retriever}")
    print(f"   metrics_path   : {metrics_path}")
    print(f"   metrics_abs    : {metrics_path.resolve()}")
    print(f"   metrics_exists : {metrics_path.exists()}")
    print("=" * 80)

    if not metrics_path.exists():
        print(f"⚠️ metrics non trouvé: {metrics_path}")
        return None, retriever, str(metrics_path)

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        # Logs sur la structure du JSON
        if isinstance(report, dict):
            print(f"✅ Metrics chargées. Clés top-level: {list(report.keys())}")
            maybe_metrics = report.get("metrics")
            if isinstance(maybe_metrics, dict):
                print(f"   Clés report['metrics']: {list(maybe_metrics.keys())}")
            else:
                print("   report['metrics'] absent ou non-dict (métriques peut-être à la racine).")
        else:
            print(f"⚠️ Metrics chargées mais format inattendu: type={type(report)}")

        return report, retriever, str(metrics_path)

    except Exception as e:
        print(f"⚠️ Erreur lecture metrics: {e}")
        return None, retriever, str(metrics_path)



def format_date(date_str):
    """Formate une date ou retourne N/A."""
    if not date_str or pd.isna(date_str) or date_str == '':
        return "N/A"
    return str(date_str)

def analyze_results(results_file, corpus_path, qrels_file=None, top_k=10):
    """
    Analyse complète des résultats avec enrichissement corpus.
    
    Args:
        results_file: Chemin vers bm25_results.jsonl / dense_results.jsonl / hybrid_results.jsonl
        corpus_path: Chemin vers gold/chunks (Parquet)
        qrels_file: Chemin vers qrels.yaml (optionnel)
        top_k: Nombre de résultats à afficher par query
    """
    # Charger données
    df_corpus = load_corpus(corpus_path)
    results = load_results(results_file)

    # Charger config_resolved.yaml du run pour rester cohérent avec l'évaluation du benchmark.
    # Objectif: surligner dans le HTML les docs qui matchent *le même qrels* que celui utilisé pour calculer metrics_*.json.
    run_dir = Path(results_file).parent
    run_config = load_run_config(run_dir)

    qrels_from_config = None
    if isinstance(run_config, dict):
        qrels_from_config = run_config.get("evaluation", {}).get("qrels_file")

    if qrels_from_config and Path(qrels_from_config).exists():
        if qrels_file and str(qrels_file) != str(qrels_from_config):
            print(f"⚠️ Qrels fourni en argument différent de celui du run. On utilise celui du run pour l'affichage HTML.")
            print(f"   - qrels arg : {qrels_file}")
            print(f"   - qrels run : {qrels_from_config}")
        qrels_file = qrels_from_config
        print(f"✅ Qrels (depuis config_resolved.yaml): {qrels_file}")
    else:
        if qrels_file:
            print(f"ℹ️ Qrels (argument): {qrels_file}")
        else:
            print("ℹ️ Aucun qrels fourni et aucun qrels trouvé dans config_resolved.yaml (pas de surlignage des pertinents).")

    qrels = load_qrels(qrels_file) if qrels_file else {}
    
    # Créer index doc_key pour recherche rapide
    df_corpus_idx = df_corpus.set_index('doc_key')
    
    # Grouper résultats par query
    queries = {}
    for r in results:
        qid = r.get('query_id')
        if qid not in queries:
            queries[qid] = []
        queries[qid].append(r)
    
    # Extraire cache_tag depuis les résultats (premier résultat qui en a un)
    cache_tag = None
    for r in results:
        if r.get('cache_tag'):
            cache_tag = r.get('cache_tag')
            break
    
    # Charger config du run
    run_dir = Path(results_file).parent
    run_config = load_run_config(run_dir)

    # Charger métriques réelles du run (sans recalcul)
    metrics_report, retriever_name, metrics_path_str = load_run_metrics(run_dir, Path(results_file))

    # Extraire les métriques par query (sans recalcul)
    per_query_metrics = {}
    if metrics_report and isinstance(metrics_report, dict):
        pq = metrics_report.get("per_query", {})
        if isinstance(pq, dict):
            per_query_metrics = pq
    
    # Charger aussi result.json pour infos additionnelles
    result_json = None
    result_json_path = run_dir / "result.json"
    if result_json_path.exists():
        try:
            with open(result_json_path) as f:
                result_json = json.load(f)
            print(f"✅ result.json chargé depuis {result_json_path}")
        except Exception as e:
            print(f"⚠️ Erreur lecture result.json: {e}")
    
    # Charger métadonnées du cache si disponible
    cache_info = None
    if cache_tag:
        # Résolution robuste du répertoire de cache :
        # 1) si config_resolved.yaml fournit dense.resolved_cache_dir / dense.cache_dir, on l'utilise (source la plus fiable)
        # 2) sinon on retombe sur <project_root>/.dense_cache puis ./.dense_cache
        cache_dir = None
        if run_config:
            dense_cfg = run_config.get("dense", {})
            cache_dir_candidate = dense_cfg.get("resolved_cache_dir") or dense_cfg.get("cache_dir")
            if cache_dir_candidate:
                cache_dir = Path(cache_dir_candidate)
        
        if cache_dir is None:
            project_root = run_dir.parent.parent if run_dir.parent.name == "runs" else run_dir.parent
            candidate = project_root / ".dense_cache"
            cache_dir = candidate if candidate.exists() else Path(".dense_cache")
        
        # Important : on n'appelle la fonction QU'UNE seule fois,
        # sinon on écrase cache_info et on perd les infos extraites du YAML.
        cache_info = load_cache_metadata(
            cache_tag,
            cache_dir=cache_dir,
            run_config=run_config,
            result_json=result_json
        )
    
    # Préparer rapport
    output_dir = Path(results_file).parent
    output_file = output_dir / f"analyse_{Path(results_file).stem}.html"
    
    # Générer rapport HTML
    html_parts = []
    
    # Header
    html_parts.append(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Analyse Résultats - {Path(results_file).stem}</title>
        <style>
            :root {{
                --bg: #0f1115;
                --panel: #161a22;
                --panel-2: #111520;
                --panel-3: #0c0f16;

                --text: #e7eaf0;
                --muted: #aab2c0;

                --border: #2a3140;
                --shadow: rgba(0, 0, 0, 0.45);

                --accent: #8b93ff;
                --accent-2: #b07cff;

                --ok: #38d070;
                --info: #3aa7ff;
                --warn: #ffcc66;

                --highlight: rgba(255, 204, 102, 0.20);
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: var(--bg);
                color: var(--text);
            }}

            .header {{
                background: linear-gradient(135deg, #2b2f6a 0%, #3a2457 100%);
                color: white;
                padding: 30px;
                border-radius: 12px;
                margin-bottom: 30px;
                border: 1px solid rgba(255,255,255,0.08);
                box-shadow: 0 10px 30px var(--shadow);
            }}

            .query-section {{
                background: var(--panel);
                padding: 25px;
                margin-bottom: 30px;
                border-radius: 12px;
                border: 1px solid var(--border);
                box-shadow: 0 8px 26px var(--shadow);
            }}

            .query-title {{
                font-size: 24px;
                color: var(--accent);
                margin-bottom: 10px;
                border-bottom: 3px solid rgba(139,147,255,0.75);
                padding-bottom: 10px;
            }}

            .query-text {{
                font-size: 16px;
                color: var(--muted);
                margin-bottom: 20px;
                padding: 15px;
                background: var(--panel-2);
                border-left: 4px solid var(--accent);
                border-radius: 10px;
            }}
            .query-hits {{
                font-size: 13px;
                color: var(--muted);
                margin: -10px 0 18px 0;
                padding-left: 15px;
                opacity: 0.9;               
            }}
            .query-metrics {{
                margin: 10px 0 18px 0;
                padding: 12px 14px;
                background: rgba(58,167,255,0.10);
                border: 1px solid rgba(58,167,255,0.20);
                border-radius: 10px;
                color: var(--text);
                font-family: 'Courier New', monospace;
                font-size: 13px;
                line-height: 1.6;
            }}
            .result {{
                border: 1px solid var(--border);
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 12px;
                background: var(--panel-2);
                transition: all 0.2s ease;
            }}

            .result:hover {{
                box-shadow: 0 10px 28px var(--shadow);
                transform: translateY(-2px);
                border-color: rgba(139,147,255,0.6);
            }}

            .result.relevant {{
                border-color: rgba(56,208,112,0.8);
                background: rgba(56,208,112,0.08);
            }}

            .result-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid rgba(42,49,64,0.8);
            }}

            .rank {{
                font-size: 28px;
                font-weight: bold;
                color: var(--accent);
                min-width: 50px;
            }}

            .doc-key {{
                font-family: 'Courier New', monospace;
                font-size: 14px;
                color: var(--text);
                background: var(--panel-3);
                padding: 6px 10px;
                border-radius: 10px;
                border: 1px solid rgba(42,49,64,0.9);
            }}

            .link-out {{
                color: #ff9f1a;       /* orange lisible sur fond sombre */
                font-weight: 600;
                text-decoration: none;
            }}
            .link-out:hover {{
                text-decoration: underline;
            }}

            .score {{
                font-size: 18px;
                font-weight: bold;
                color: var(--ok);
            }}

            .badge {{
                display: inline-block;
                padding: 5px 12px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 700;
                margin-left: 10px;
                border: 1px solid rgba(255,255,255,0.10);
            }}

            .badge-relevant {{
                background: rgba(56,208,112,0.20);
                color: var(--ok);
                border-color: rgba(56,208,112,0.35);
            }}

            .badge-status {{
                background: rgba(58,167,255,0.16);
                color: #bfe2ff;
                border-color: rgba(58,167,255,0.30);
            }}

            .metadata {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 15px;
                padding: 15px;
                background: var(--panel-3);
                border-radius: 12px;
                border: 1px solid rgba(42,49,64,0.9);
            }}

            .meta-item {{
                display: flex;
                flex-direction: column;
                gap: 4px;
            }}

            .meta-label {{
                font-size: 11px;
                color: var(--muted);
                text-transform: uppercase;
                font-weight: 700;
                letter-spacing: 0.04em;
                margin-bottom: 2px;
            }}

            .meta-value {{
                font-size: 14px;
                color: var(--text);
                font-family: 'Courier New', monospace;
            }}

            .text-content {{
                padding: 15px;
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(42,49,64,0.9);
                border-radius: 12px;
                line-height: 1.8;
                color: var(--text);
                white-space: pre-wrap;
                font-size: 14px;
            }}

            .stats {{
                background: rgba(139,147,255,0.10);
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 20px;
                border: 1px solid rgba(139,147,255,0.22);
            }}

            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
            }}

            .stat-item {{
                text-align: center;
            }}

            .stat-value {{
                font-size: 24px;
                font-weight: 800;
                color: var(--accent);
            }}

            .stat-label {{
                font-size: 12px;
                color: var(--muted);
            }}

            .highlight {{
                background: var(--highlight);
                padding: 2px 6px;
                border-radius: 6px;
                border: 1px solid rgba(255, 204, 102, 0.25);
            }}
            
            .cache-info {{
                background: rgba(176,124,255,0.10);
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
                border: 1px solid rgba(176,124,255,0.22);
            }}
            
            .cache-info h2 {{
                color: var(--accent-2);
                margin-top: 0;
                margin-bottom: 15px;
                font-size: 20px;
                border-bottom: 2px solid rgba(176,124,255,0.35);
                padding-bottom: 8px;
            }}
            
            .cache-details {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 12px;
            }}
            
            .cache-detail {{
                background: var(--panel-3);
                padding: 12px;
                border-radius: 8px;
                border: 1px solid rgba(42,49,64,0.8);
            }}
            
            .cache-detail-label {{
                font-size: 11px;
                color: var(--muted);
                text-transform: uppercase;
                font-weight: 700;
                letter-spacing: 0.04em;
                margin-bottom: 6px;
            }}
            
            .cache-detail-value {{
                font-size: 15px;
                color: var(--text);
                font-family: 'Courier New', monospace;
            }}
            
            .cache-warning {{
                background: rgba(255,204,102,0.15);
                color: var(--warn);
                padding: 12px 15px;
                border-radius: 8px;
                border: 1px solid rgba(255,204,102,0.3);
                margin-top: 12px;
                font-size: 13px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Analyse Résultats Retrieval</h1>
            <p>Fichier: {Path(results_file).name}</p>
            <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """)

    # Section Cache Info (si disponible)
    if cache_info:
        # Extraire infos (priorité: config_resolved.yaml > .meta.json)
        model_name = cache_info.get('embedding_model', 'N/A')
        device = cache_info.get('device', 'N/A')
        use_gpu_optimized = cache_info.get('use_gpu_optimized')
        batch_size = cache_info.get('batch_size', 'N/A')
        
        num_embeddings = cache_info.get('num_embeddings', 'N/A')
        embedding_dim = cache_info.get('embedding_dim', 'N/A')
        file_size_mb = cache_info.get('file_size_mb', 0)
        file_size_gb = cache_info.get('file_size_gb', 0)
        created = cache_info.get('created', 'N/A')
        
        text_field = cache_info.get('text_field', 'N/A')
        
        # Config corpus et filtres
        corpus_source = cache_info.get('corpus_source', 'N/A')
        limit = cache_info.get('limit')
        limit_after_filter = cache_info.get('limit_after_filter')
        filters = cache_info.get('filters', {})
        doc_types = cache_info.get('doc_types', [])
        
        # Format taille
        if file_size_gb >= 1.0:
            size_display = f"{file_size_gb:.2f} GB"
        elif file_size_mb > 0:
            size_display = f"{file_size_mb:.1f} MB"
        else:
            size_display = "N/A"
        
        # Format GPU optimized
        gpu_opt_display = "✅ Oui" if use_gpu_optimized else "❌ Non" if use_gpu_optimized is not None else "N/A"
        
        html_parts.append(f"""
    <div class="cache-info">
        <h2>🗄️ Informations Cache Dense</h2>
        <div class="cache-details">
            <div class="cache-detail">
                <div class="cache-detail-label">Cache Tag</div>
                <div class="cache-detail-value">{cache_tag}</div>
            </div>
            <div class="cache-detail">
                <div class="cache-detail-label">Modèle Embedding</div>
                <div class="cache-detail-value" style="font-size: 12px;">{model_name}</div>
            </div>
            <div class="cache-detail">
                <div class="cache-detail-label">Device</div>
                <div class="cache-detail-value">{device}</div>
            </div>
            <div class="cache-detail">
                <div class="cache-detail-label">GPU Optimisé</div>
                <div class="cache-detail-value">{gpu_opt_display}</div>
            </div>
            <div class="cache-detail">
                <div class="cache-detail-label">Batch Size</div>
                <div class="cache-detail-value">{batch_size}</div>
            </div>
            <div class="cache-detail">
                <div class="cache-detail-label">Nombre d'embeddings</div>
                <div class="cache-detail-value">{num_embeddings:,} chunks</div>
            </div>
            <div class="cache-detail">
                <div class="cache-detail-label">Dimension</div>
                <div class="cache-detail-value">{embedding_dim}</div>
            </div>
            <div class="cache-detail">
                <div class="cache-detail-label">Taille fichier</div>
                <div class="cache-detail-value">{size_display}</div>
            </div>
            <div class="cache-detail">
                <div class="cache-detail-label">Champ texte</div>
                <div class="cache-detail-value">{text_field}</div>
            </div>
            <div class="cache-detail">
                <div class="cache-detail-label">Date création</div>
                <div class="cache-detail-value">{created[:10] if created != 'N/A' else 'N/A'}</div>
            </div>
        </div>
        """)
        
        # Section corpus et limites
        html_parts.append(f"""
        <div class="cache-details" style="margin-top: 15px;">
            <div class="cache-detail">
                <div class="cache-detail-label">Corpus source</div>
                <div class="cache-detail-value" style="font-size: 12px; word-break: break-all;">
                    {corpus_source}
                </div>
            </div>
            <div class="cache-detail">
                <div class="cache-detail-label">Limite corpus (avant filtres)</div>
                <div class="cache-detail-value">{limit if limit else 'Aucune'}</div>
            </div>
            <div class="cache-detail">
                <div class="cache-detail-label">Limite après filtres</div>
                <div class="cache-detail-value">{limit_after_filter if limit_after_filter else 'Aucune'}</div>
            </div>
        </div>
            """)
        
        # Afficher filtres et doc_types
        if filters or doc_types:
            filters_parts = []
            if doc_types:
                doc_types_str = ", ".join(doc_types) if isinstance(doc_types, list) else str(doc_types)
                filters_parts.append(f"• <strong>doc_types</strong>: {doc_types_str}")
            for k, v in filters.items():
                if k != 'doc_types':  # Éviter doublon
                    filters_parts.append(f"• <strong>{k}</strong>: {v}")
            
            if filters_parts:
                filters_html = "<br>".join(filters_parts)
                html_parts.append(f"""
        <div class="cache-warning">
            <strong>⚠️ Filtres appliqués :</strong><br>
            {filters_html}
        </div>
                """)
        
        html_parts.append("</div>\n")
    
    elif cache_tag:
        # Cache tag présent mais métadonnées non trouvées
        html_parts.append(f"""
    <div class="cache-info">
        <h2>🗄️ Informations Cache Dense</h2>
        <div class="cache-details">
            <div class="cache-detail">
                <div class="cache-detail-label">Cache Tag</div>
                <div class="cache-detail-value">{cache_tag}</div>
            </div>
        </div>
        <div class="cache-warning">
            ⚠️ Métadonnées du cache non trouvées dans .dense_cache/
        </div>
    </div>
    """)


    # Statistiques globales (métriques réelles du run, sans recalcul)
    total_results = len(results)
    total_queries = len(queries)

    recall_k = "N/A"
    mrr = "N/A"
    ndcg_k = "N/A"

    if metrics_report and isinstance(metrics_report, dict):
        # Format : {"average": {...}, "per_query": {...}, "cache_tag": "..."}
        avg = metrics_report.get("average", {})
        
        if isinstance(avg, dict) and avg:
            recall_k = avg.get(f"Recall@{top_k}", "N/A")
            mrr = avg.get("MRR", "N/A")
            ndcg_k = avg.get(f"nDCG@{top_k}", "N/A")
            
            # Formatage si float  ← 4 espaces (OK)
            if isinstance(recall_k, float):
                recall_k = f"{recall_k:.4f}"
            if isinstance(mrr, float):
                mrr = f"{mrr:.4f}"
            if isinstance(ndcg_k, float):
                ndcg_k = f"{ndcg_k:.4f}"
            
            # Logs
            print("\n" + "=" * 80)
            print("🧾 DEBUG METRICS (valeurs affichées HTML)")
            print(f"   metrics_path : {metrics_path_str}")
            print(f"   avg_keys     : {list(avg.keys())}")  # ← Corriger : avg au lieu de metrics
            print(f"   Recall@{top_k}: {recall_k}")
            print(f"   MRR          : {mrr}")
            print(f"   nDCG@{top_k}  : {ndcg_k}")
            print("=" * 80 + "\n")
        else:
            print("⚠️ Aucune métrique trouvée dans metrics_*.json")


    html_parts.append(f"""
    <div class="stats">
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{total_queries}</div>
                <div class="stat-label">Queries (dans le fichier résultats)</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{total_results}</div>
                <div class="stat-label">Résultats (lignes JSONL)</div>
            </div>

            <div class="stat-item">
                <div class="stat-value">{recall_k}</div>
                <div class="stat-label">Recall@{top_k} (metrics_*.json)</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{mrr}</div>
                <div class="stat-label">MRR (metrics_*.json)</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{ndcg_k}</div>
                <div class="stat-label">nDCG@{top_k} (metrics_*.json)</div>
            </div>
        </div>

        <div style="margin-top: 10px; color: var(--muted); font-size: 12px;">
            Metrics source: {metrics_path_str}
        </div>
    </div>
""")

    
    # Pour chaque query
    for qid in sorted(queries.keys()):
        query_results = queries[qid][:top_k]
        query_text = query_results[0].get('query_text', 'N/A')
        query_qrels_map = qrels.get(qid, {})
        query_qrels_keys = set(query_qrels_map.keys())
        
        # Stats query
        found_relevant = sum(1 for r in query_results if r['doc_key'] in query_qrels_keys)
        hits = [r['doc_key'] for r in query_results if r['doc_key'] in query_qrels_keys]

        # Ligne de synthèse: permet de voir immédiatement quels résultats expliquent Recall/MRR/nDCG
        query_hits_html = ""
        if query_qrels_keys:
            query_hits_html = f'<div class="query-hits">Hits qrels dans top{top_k}: {", ".join(hits) if hits else "Aucun"}</div>'
  
        
        html_parts.append(f"""
    <div class="query-section">
        <div class="query-title">
            {qid}
            {f'<span class="badge badge-status">{found_relevant}/{len(query_qrels_keys)} pertinents</span>' if query_qrels_keys else ''}
        </div>
        <div class="query-text">{query_text}</div>
        {query_hits_html}
        
""")
        # Mètres par query (source: metrics_*.json -> per_query), sans recalcul
        qm = per_query_metrics.get(qid, {})
        if isinstance(qm, dict) and qm:
            q_recall = qm.get(f"Recall@{top_k}", qm.get("Recall@10", "N/A"))
            q_mrr = qm.get("MRR", "N/A")
            q_ndcg = qm.get(f"nDCG@{top_k}", qm.get("nDCG@10", "N/A"))

            q_num_rel = qm.get("num_relevant", "N/A")
            q_num_ret = qm.get("num_retrieved", "N/A")

            # Format propre si floats
            if isinstance(q_recall, float):
                q_recall = f"{q_recall:.4f}"
            if isinstance(q_mrr, float):
                q_mrr = f"{q_mrr:.4f}"
            if isinstance(q_ndcg, float):
                q_ndcg = f"{q_ndcg:.4f}"

            query_metrics_html = (
                f'<div class="query-metrics">'
                f'<span class="label">Metrics query</span>'
                f'Recall@{top_k}={q_recall} | MRR={q_mrr} | nDCG@{top_k}={q_ndcg}'
                f' &nbsp;&nbsp; (num_relevant={q_num_rel} / num_retrieved={q_num_ret})'
                f'</div>'
            )
        else:
            query_metrics_html = (
                f'<div class="query-metrics">'
                f'<span class="label">Metrics query</span>'
                f'Indisponibles dans metrics_*.json pour {qid}'
                f'</div>'
            )

        html_parts.append(query_metrics_html)
        
        # Pour chaque résultat
        for i, result in enumerate(query_results, 1):
            doc_key = result['doc_key']
            score = result.get('score', 0)
            qrel_score = query_qrels_map.get(doc_key)
            is_relevant = qrel_score is not None
            
            # Récupérer métadonnées depuis corpus
            if doc_key in df_corpus_idx.index:
                doc_data = df_corpus_idx.loc[doc_key].iloc[0] if isinstance(df_corpus_idx.loc[doc_key], pd.DataFrame) else df_corpus_idx.loc[doc_key]
                
                # Texte : on le rend "safe" et on linkify les IDs juridiques
                raw_text = doc_data.get('chunk_text', '[Texte non disponible]')
                text_full = linkify_legal_ids(raw_text, df_corpus_idx)
                valid_from = format_date(doc_data.get('valid_from'))
                valid_to = format_date(doc_data.get('valid_to'))
                status = doc_data.get('status', 'N/A')
                corpus_juridique = doc_data.get('corpus_juridique', 'N/A')
                doc_type = doc_data.get('doc_type', 'N/A')
                article_num = doc_data.get('article_num', 'N/A')
                version_key = doc_data.get('version_key', 'N/A')
                source_path = doc_data.get('source_path')
            else:
                text_full = linkify_legal_ids('[Document non trouvé dans corpus]', df_corpus_idx)
                valid_from = valid_to = status = corpus_juridique = doc_type = article_num = version_key = 'N/A'
                source_path = None
            
            # Déterminer si article en vigueur
            en_vigueur = "✅ EN VIGUEUR" if valid_to in ['2999-01-01', '9999-01-01'] else f"⚠️ Abrogé ({valid_to})"

            # Liens cliquables (XML local si possible, sinon Legifrance)
            xml_url = build_xml_file_url(source_path)
            legi_url = build_legifrance_url(doc_key)
            doc_href = xml_url or legi_url
            doc_key_html = (
                f'<a class="doc-key" href="{doc_href}" target="_blank" rel="noopener" '
                f'style="text-decoration: none;">{doc_key}</a>'
                + (
                    f' <a class="link-out" href="{xml_url}" target="_blank" rel="noopener">[XML]</a>'
                    if xml_url else ''
                )
                + f' <a class="link-out" href="{legi_url}" target="_blank" rel="noopener">[Legifrance]</a>'
            )
            
            html_parts.append(f"""
            <div class="result {'relevant' if is_relevant else ''}" id="doc-{doc_key}">
            <div class="result-header">
                <div>
                    <span class="rank">#{i}</span>
                    {doc_key_html}
                    {f'<span class="badge badge-relevant">✅ QREL={qrel_score}</span>' if is_relevant else ''}

                </div>
                <span class="score">Score: {score:.6f}</span>
            </div>
            
            <div class="metadata">
                <div class="meta-item">
                    <span class="meta-label">📅 Valid From</span>
                    <span class="meta-value">{valid_from}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">📅 Valid To</span>
                    <span class="meta-value">{valid_to}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">📊 Status</span>
                    <span class="meta-value">{status if status != 'N/A' else en_vigueur}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">📋 Corpus Juridique</span>
                    <span class="meta-value">{corpus_juridique if corpus_juridique else 'N/A'}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">📄 Doc Type</span>
                    <span class="meta-value">{doc_type}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">🔢 Article #</span>
                    <span class="meta-value">{article_num if article_num else 'N/A'}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">🔑 Version Key</span>
                    <span class="meta-value" style="font-size: 11px;">{version_key[:30]}...</span>
                </div>
            </div>
            
            <div class="text-content">
{text_full}
            </div>
        </div>
""")
        
        html_parts.append("    </div>")  # Close query-section
    
    # Footer
    html_parts.append("""
</body>
</html>
""")
    
    # Écrire fichier HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    
    print(f"\n✅ Rapport généré: {output_file}")
    print(f"📂 Ouvrez-le dans votre navigateur pour analyse interactive !")
    
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/analyze_results.py <results_file.jsonl>")
        print("\nExemples:")
        print("  python tools/analyze_results.py runs/20260121_XXXXXX/bm25_results.jsonl")
        print("  python tools/analyze_results.py runs/20260121_XXXXXX/dense_results.jsonl")
        print("  python tools/analyze_results.py runs/20260121_XXXXXX/hybrid_results.jsonl")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    # Chemins par défaut
    corpus_path = "/home/fred/montage1/-- Projet RAG Avocats --/data_main/data/datalake_legifrance_v1/gold/chunks"
    qrels_file = "/home/fred/montage1/-- Projet RAG Avocats --/rag_bench/configs/qrels_cdtravail_v2.yaml"
    
    # Analyser
    analyze_results(results_file, corpus_path, qrels_file, top_k=10)