#!/usr/bin/env python3
"""
Script pour associer les caches aux runs
Phase 3 - Exploration des caches

Usage:
    python tools/match_cache_to_runs.py [--runs-dir runs] [--cache-dir .dense_cache]

Affiche:
- Quel cache a été utilisé par quels runs
- Historique d'utilisation de chaque cache
- Caches orphelins (non utilisés)
- Analyse de performance par cache

📝 IMPORTANT : le script sauvegarde l'intégralité des logs dans un fichier Markdown
   dans le dossier logs/ (nom_de_fichier_horodatage.md).
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
import argparse
from collections import defaultdict
import logging


class MarkdownLogFormatter(logging.Formatter):
    """Formatter orienté Markdown (une ligne = une puce)."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        return f"- [{ts}] **{record.levelname}** — {record.getMessage()}"


def setup_logging(script_path: Path, level: str = "INFO"):
    """
    Configure un logger :
    - console (stdout)
    - fichier Markdown dans logs/ : <nom_script>_<horodatage>.md
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = Path(script_path).stem
    md_log_path = logs_dir / f"{base_name}_{ts}.md"

    header = [
        f"# Logs — {base_name}",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Dossier courant: `{Path.cwd().absolute()}`",
        "",
        "---",
        "",
    ]
    md_log_path.write_text("\n".join(header), encoding="utf-8")

    logger = logging.getLogger(base_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Nettoyage si relance dans le même process (ex: notebook)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    fh = logging.FileHandler(md_log_path, mode="a", encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(MarkdownLogFormatter())
    logger.addHandler(fh)

    return logger, md_log_path


def extract_cache_tag_from_run(run_dir):
    """
    Extrait le cache_tag d'un run depuis result.json / config_resolved.yaml / dense_results.jsonl.
    """
    result_file = Path(run_dir) / "result.json"
    if result_file.exists():
        try:
            with result_file.open("r", encoding="utf-8") as f:
                result = json.load(f)
            cache_tag = result.get("cache_tag")
            if cache_tag:
                return cache_tag
            dense = result.get("dense", {})
            if dense.get("cache_tag"):
                return dense["cache_tag"]
        except Exception:
            pass

    config_file = Path(run_dir) / "config_resolved.yaml"
    if config_file.exists():
        try:
            with config_file.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            dense = config.get("dense", {})
            if dense.get("cache_tag"):
                return dense["cache_tag"]
        except Exception:
            pass

    dense_results = Path(run_dir) / "dense_results.jsonl"
    if dense_results.exists():
        try:
            with dense_results.open("r", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line:
                    data = json.loads(first_line)
                    if data.get("cache_tag"):
                        return data["cache_tag"]
        except Exception:
            pass

    return None


def extract_metrics_from_run(run_dir):
    """Extrait les métriques du run."""
    metrics = {}
    metrics_file = Path(run_dir) / "metrics_dense.json"
    if metrics_file.exists():
        try:
            with metrics_file.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            pass
    return metrics


def list_runs(runs_dir):
    """Liste tous les runs dans le dossier."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    runs = [d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return runs


def main():
    parser = argparse.ArgumentParser(description="Associer caches aux runs")
    parser.add_argument("--runs-dir", default="runs", help="Dossier des runs")
    parser.add_argument("--cache-dir", default=".dense_cache", help="Dossier des caches")
    parser.add_argument("--limit", type=int, help="Limiter au N runs les plus récents")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de logs",
    )
    args = parser.parse_args()

    logger, md_log_path = setup_logging(Path(__file__), level=args.log_level)

    runs_dir = Path(args.runs_dir)
    cache_dir = Path(args.cache_dir)

    logger.info("=" * 80)
    logger.info("🔗 ASSOCIATION CACHES ↔ RUNS")
    logger.info("=" * 80)
    logger.info("")

    logger.info(f"📁 Runs: {runs_dir.absolute()}")
    logger.info(f"📝 Logs sauvegardés dans: {md_log_path.absolute()}")
    runs = list_runs(runs_dir)

    if not runs:
        logger.error("❌ Aucun run trouvé")
        return

    if args.limit:
        runs = runs[:args.limit]

    logger.info(f"✅ {len(runs)} run(s) trouvé(s)")
    logger.info("")

    cache_to_runs = defaultdict(list)
    runs_without_cache = []

    logger.info("🔍 Analyse des runs...")
    for run_dir in runs:
        cache_tag = extract_cache_tag_from_run(run_dir)
        metrics = extract_metrics_from_run(run_dir)

        run_info = {
            "run_dir": str(run_dir.name),
            "full_path": str(run_dir),
            "date": datetime.fromtimestamp(run_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            "metrics": metrics,
        }

        if cache_tag:
            cache_to_runs[cache_tag].append(run_info)
        else:
            runs_without_cache.append(run_info)

    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 RÉSULTATS")
    logger.info("=" * 80)
    logger.info("")

    if cache_to_runs:
        logger.info(f"🗄️ {len(cache_to_runs)} cache(s) utilisé(s)")
        logger.info("")
        for cache_tag, runs_list in sorted(cache_to_runs.items(), key=lambda x: len(x[1]), reverse=True):
            logger.info("─" * 80)
            logger.info(f"📦 Cache: {cache_tag}")
            logger.info(f"   Utilisé par {len(runs_list)} run(s)")
            logger.info("")
            for idx, run_info in enumerate(runs_list, 1):
                logger.info(f"   #{idx}. {run_info['run_dir']}")
                logger.info(f"       Date    : {run_info['date']}")
                metrics = run_info['metrics']
                if metrics:
                    recall = metrics.get('recall@10')
                    if recall is not None:
                        logger.info(f"       Recall@10: {recall:.3f}")
                    mrr = metrics.get('MRR')
                    if mrr is not None:
                        logger.info(f"       MRR     : {mrr:.3f}")
                logger.info("")
    else:
        logger.error("❌ Aucun cache trouvé dans les runs")

    if runs_without_cache:
        logger.info("=" * 80)
        logger.info(f"⚠️ {len(runs_without_cache)} run(s) sans cache_tag")
        logger.info("=" * 80)
        logger.info("")
        for run_info in runs_without_cache[:10]:
            logger.info(f"   - {run_info['run_dir']} ({run_info['date']})")
        if len(runs_without_cache) > 10:
            logger.info(f"   ... et {len(runs_without_cache) - 10} autres")
        logger.info("")

    if cache_dir.exists():
        all_cache_files = list(cache_dir.glob("*.npy"))
        all_cache_tags = {f.stem for f in all_cache_files}
        used_cache_tags = set(cache_to_runs.keys())
        orphan_caches = all_cache_tags - used_cache_tags

        if orphan_caches:
            logger.info("=" * 80)
            logger.info(f"🗑️ {len(orphan_caches)} cache(s) orphelin(s) (non utilisé dans les runs récents)")
            logger.info("=" * 80)
            logger.info("")
            for cache_tag in sorted(orphan_caches):
                cache_file = cache_dir / f"{cache_tag}.npy"
                size_mb = cache_file.stat().st_size / (1024 * 1024)
                date = datetime.fromtimestamp(cache_file.stat().st_mtime).strftime('%Y-%m-%d')
                logger.info(f"   - {cache_tag}")
                logger.info(f"     Taille: {size_mb:.1f} MB, Modifié: {date}")
            logger.info("")
            logger.info("   💡 Ces caches peuvent être supprimés si plus nécessaires")
            logger.info("")

    if cache_to_runs:
        logger.info("=" * 80)
        logger.info("📈 PERFORMANCE PAR CACHE")
        logger.info("=" * 80)
        logger.info("")
        cache_perf = []
        for cache_tag, runs_list in cache_to_runs.items():
            recalls = [r['metrics'].get('recall@10') for r in runs_list if r['metrics'].get('recall@10') is not None]
            mrrs = [r['metrics'].get('MRR') for r in runs_list if r['metrics'].get('MRR') is not None]

            perf = {
                "cache_tag": cache_tag,
                "num_runs": len(runs_list),
                "avg_recall": sum(recalls) / len(recalls) if recalls else None,
                "avg_mrr": sum(mrrs) / len(mrrs) if mrrs else None,
            }
            cache_perf.append(perf)

        cache_perf.sort(key=lambda x: x['avg_recall'] or 0, reverse=True)

        for idx, perf in enumerate(cache_perf, 1):
            logger.info(f"#{idx}. {perf['cache_tag']}")
            logger.info(f"    Runs      : {perf['num_runs']}")
            if perf['avg_recall'] is not None:
                logger.info(f"    Recall@10 : {perf['avg_recall']:.3f}")
            if perf['avg_mrr'] is not None:
                logger.info(f"    MRR       : {perf['avg_mrr']:.3f}")
            logger.info("")

        if cache_perf and cache_perf[0].get('avg_recall') is not None:
            best = cache_perf[0]
            logger.info("💡 RECOMMANDATION")
            logger.info("─" * 80)
            logger.info(f"   Cache optimal : {best['cache_tag']}")
            logger.info(f"   Recall@10     : {best['avg_recall']:.3f}")
            logger.info(f"   Basé sur      : {best['num_runs']} run(s)")
            logger.info("")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
