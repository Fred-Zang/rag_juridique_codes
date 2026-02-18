#!/usr/bin/env python3
"""
Script d'inspection des caches denses
Phase 3 - Exploration des caches

Usage:
    python tools/inspect_dense_caches.py [--cache-dir .dense_cache]

Ce script :
- Liste tous les caches (*.npy) du dossier indiqué
- Charge les métadonnées (*.meta.json) quand disponibles
- Affiche taille, dates, modèle, device, champ texte, shape/dimension
- Produit un résumé (taille totale, total embeddings, caches anciens, etc.)

📝 IMPORTANT : le script sauvegarde également l'intégralité des logs dans un fichier .md  
   directement dans le dossier des caches analysés (cache_dir). => sav plutot à faire dans /logs/
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import logging


def format_size(bytes_size: int) -> str:
    """Formate une taille en bytes vers KB/MB/GB."""
    if bytes_size >= 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024 * 1024):.2f} GB"
    if bytes_size >= 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.1f} MB"
    if bytes_size >= 1024:
        return f"{bytes_size / 1024:.1f} KB"
    return f"{bytes_size} B"


def format_date(timestamp: float) -> str:
    """Formate un timestamp (epoch) vers date lisible."""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


class MarkdownLogFormatter(logging.Formatter):
    """Formatter orienté Markdown (une ligne = une puce)."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        msg = record.getMessage()
        return f"- [{ts}] **{record.levelname}** — {msg}"


def setup_logging(cache_dir: Path, level: str = "INFO") -> tuple[logging.Logger, Path]:
    """
    Configure un logger :
    - console (stdout)
    - fichier Markdown dans cache_dir
    - fichier Markdown dans logs/ : <nom_script>_<horodatage>.md

    Returns:
        (logger, md_log_path)
    """
    cache_dir = Path(cache_dir)
    logs_dir = Path("logs")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_log_path = logs_dir / f"{Path(__file__).stem}_{ts}.md"       # sav  dans /logs/

    # Écrire un en-tête markdown avant de brancher le FileHandler
    header = [
        f"# Inspection des caches denses",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Dossier analysé: `{cache_dir.absolute()}`",
        f"- Dossier logs: `{logs_dir.absolute()}`",
        "",
        "---",
        "",
    ]
    md_log_path.write_text("\n".join(header), encoding="utf-8")

    logger = logging.getLogger("inspect_dense_caches")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Nettoyage si relance dans le même process (ex: notebook)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Console handler (lisible)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    # Fichier markdown handler
    fh = logging.FileHandler(md_log_path, mode="a", encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(MarkdownLogFormatter())
    logger.addHandler(fh)

    return logger, md_log_path


def inspect_cache(cache_tag: str, cache_dir: Path, logger: logging.Logger) -> dict:
    """
    Inspecte un cache donné et retourne ses métadonnées.

    Args:
        cache_tag: Hash du cache
        cache_dir: Dossier .dense_cache
        logger: Logger configuré

    Returns:
        dict: Métadonnées complètes du cache
    """
    cache_dir = Path(cache_dir)
    meta_file = cache_dir / f"{cache_tag}.meta.json"
    npy_file = cache_dir / f"{cache_tag}.npy"

    cache_info = {
        "cache_tag": cache_tag,
        "meta_exists": meta_file.exists(),
        "npy_exists": npy_file.exists(),
    }

    # Lire métadonnées JSON
    if meta_file.exists():
        try:
            with meta_file.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            cache_info["meta"] = meta
            cache_info["embedding_model"] = meta.get("embedding_model", "N/A")
            cache_info["device"] = meta.get("device", "N/A")
            cache_info["text_field"] = meta.get("text_field", "N/A")
            cache_info["timestamp"] = meta.get("timestamp", "N/A")
        except Exception as e:
            cache_info["meta_error"] = str(e)
            logger.warning(f"⚠️ Erreur lecture meta pour {cache_tag}: {e}")
    else:
        cache_info["embedding_model"] = "N/A"
        cache_info["device"] = "N/A"
        cache_info["text_field"] = "N/A"

    # Lire infos NPY
    if npy_file.exists():
        try:
            file_stat = npy_file.stat()
            cache_info["file_size"] = file_stat.st_size
            cache_info["file_size_str"] = format_size(file_stat.st_size)

            cache_info["created_ts"] = file_stat.st_ctime
            cache_info["modified_ts"] = file_stat.st_mtime
            cache_info["created"] = format_date(file_stat.st_ctime)
            cache_info["modified"] = format_date(file_stat.st_mtime)

            # mmap_mode='r' évite de charger l'intégralité du tableau en RAM
            embeddings = np.load(npy_file, mmap_mode="r")
            cache_info["shape"] = tuple(embeddings.shape)
            cache_info["num_embeddings"] = int(embeddings.shape[0]) if len(embeddings.shape) >= 1 else 0
            cache_info["embedding_dim"] = int(embeddings.shape[1]) if len(embeddings.shape) > 1 else None

        except Exception as e:
            cache_info["npy_error"] = str(e)
            logger.warning(f"⚠️ Erreur lecture npy pour {cache_tag}: {e}")
    else:
        cache_info["file_size"] = 0

    return cache_info


def list_caches(cache_dir: Path, logger: logging.Logger) -> list[str]:
    """Liste tous les caches (basés sur les *.npy) dans le dossier."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        logger.error(f"❌ Dossier {cache_dir} n'existe pas")
        return []

    npy_files = list(cache_dir.glob("*.npy"))
    return [f.stem for f in npy_files]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspecter les caches denses")
    parser.add_argument("--cache-dir", default=".dense_cache", help="Dossier des caches")
    parser.add_argument(
        "--sort-by",
        choices=["size", "date", "embeddings"],
        default="size",
        help="Trier par taille, date ou nombre d'embeddings",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de logs",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    logger, md_log_path = setup_logging(cache_dir, level=args.log_level)

    logger.info("=" * 80)
    logger.info("🔍 INSPECTION DES CACHES DENSES")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"📁 Dossier: {cache_dir.absolute()}")
    logger.info(f"📝 Logs sauvegardés dans: {md_log_path.absolute()}")
    logger.info("")

    cache_tags = list_caches(cache_dir, logger)

    if not cache_tags:
        logger.error("❌ Aucun cache trouvé")
        sys.exit(1)

    logger.info(f"✅ {len(cache_tags)} cache(s) trouvé(s)")
    logger.info("")

    caches_info: list[dict] = []
    for tag in cache_tags:
        caches_info.append(inspect_cache(tag, cache_dir, logger))

    # Trier selon choix
    if args.sort_by == "size":
        caches_info.sort(key=lambda x: x.get("file_size", 0), reverse=True)
    elif args.sort_by == "date":
        caches_info.sort(key=lambda x: x.get("created_ts", 0), reverse=True)
    elif args.sort_by == "embeddings":
        caches_info.sort(key=lambda x: x.get("num_embeddings", 0), reverse=True)

    # Afficher tableau
    logger.info("─" * 80)
    for idx, cache in enumerate(caches_info, 1):
        logger.info(f"\n📦 CACHE #{idx}: {cache['cache_tag']}")
        logger.info("─" * 80)

        if cache.get("meta"):
            logger.info(f"   Modèle          : {cache.get('embedding_model', 'N/A')}")
            logger.info(f"   Device          : {cache.get('device', 'N/A')}")
            logger.info(f"   Champ texte     : {cache.get('text_field', 'N/A')}")
        else:
            logger.info("   (pas de fichier meta.json)")

        if cache.get("npy_exists"):
            logger.info(f"   Taille          : {cache.get('file_size_str', 'N/A')}")
            num = cache.get("num_embeddings", "N/A")
            if isinstance(num, int):
                logger.info(f"   Embeddings      : {num:,} chunks")
            else:
                logger.info(f"   Embeddings      : {num} chunks")
            logger.info(f"   Dimension       : {cache.get('embedding_dim', 'N/A')}")
            logger.info(f"   Créé le         : {cache.get('created', 'N/A')}")
            logger.info(f"   Modifié le      : {cache.get('modified', 'N/A')}")
        else:
            logger.info("   (pas de fichier .npy)")

        if cache.get("meta_error"):
            logger.warning(f"   ⚠️ Erreur meta   : {cache['meta_error']}")
        if cache.get("npy_error"):
            logger.warning(f"   ⚠️ Erreur npy    : {cache['npy_error']}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 RÉSUMÉ")
    logger.info("=" * 80)

    total_size = sum(int(c.get("file_size", 0) or 0) for c in caches_info)
    total_embeddings = sum(int(c.get("num_embeddings", 0) or 0) for c in caches_info if isinstance(c.get("num_embeddings", 0), int))

    logger.info(f"   Total caches    : {len(caches_info)}")
    logger.info(f"   Taille totale   : {format_size(total_size)}")
    logger.info(f"   Total embeddings: {total_embeddings:,} chunks")
    logger.info("")

    logger.info("💡 RECOMMANDATIONS")
    logger.info("─" * 80)

    if caches_info:
        biggest = max(caches_info, key=lambda x: x.get("file_size", 0))
        logger.info(f"   Cache le plus volumineux : {biggest['cache_tag']}")
        logger.info(f"      Taille : {biggest.get('file_size_str', 'N/A')}")
        logger.info(f"      Chunks : {int(biggest.get('num_embeddings', 0) or 0):,}")
        if biggest.get('file_size', 0) > 500 * 1024 * 1024:
            logger.warning("      ⚠️ Cache très volumineux (>500 MB) - vérifier si nécessaire")
        logger.info("")

        recent = max(caches_info, key=lambda x: x.get("created_ts", 0))
        logger.info(f"   Cache le plus récent : {recent['cache_tag']}")
        logger.info(f"      Date : {recent.get('created', 'N/A')}")
        logger.info("")

        now = datetime.now()
        old_threshold = now - timedelta(days=30)
        old_caches: list[tuple[str, str]] = []
        for cache in caches_info:
            created_ts = cache.get("created_ts")
            if isinstance(created_ts, (int, float)) and created_ts > 0:
                created_dt = datetime.fromtimestamp(created_ts)
                if created_dt < old_threshold:
                    old_caches.append((cache["cache_tag"], cache.get("created", "N/A")))

        if old_caches:
            logger.warning(f"   ⚠️ Caches anciens (>30 jours) : {len(old_caches)}")
            for tag, date in old_caches:
                logger.warning(f"      - {tag} (créé: {date})")
            logger.info("      💡 Utiliser match_cache_to_runs.py pour vérifier si encore utilisés")
        else:
            logger.info("   ✅ Tous les caches sont récents (<30 jours)")

    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
