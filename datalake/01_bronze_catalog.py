#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_bronze_catalog.py

Construit un catalogue BRONZE (Parquet) à partir de l'inventaire gz.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, types as T

from datalake.utils import load_yaml, human_readable_size


# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(config_path: str) -> None:
    """
    Construit le catalogue BRONZE depuis l'inventaire.
    
    Args:
        config_path: Chemin vers datalake_pipeline_v1.yaml
    """
    logger.info("=" * 80)
    logger.info("DÉMARRAGE: 01_bronze_catalog.py")
    logger.info("=" * 80)
    
    cfg = load_yaml(config_path)

    root_corpus = cfg["paths"]["root_corpus"].rstrip("/") + "/"
    inv_gz = cfg["paths"]["audit_inventory_gz"]
    work_dir = cfg["paths"]["work_dir"]
    snapshot_dt = cfg["project"]["snapshot_dt"]

    logger.info("Configuration:")
    logger.info("  root_corpus: %s", root_corpus)
    logger.info("  inventory: %s", inv_gz)
    logger.info("  work_dir: %s", work_dir)
    logger.info("  snapshot_dt: %s", snapshot_dt)

    bronze_dir = Path(work_dir) / cfg["bronze"]["output_subdir"]
    catalog_dir = Path(work_dir) / cfg["bronze"]["catalog_subdir"]
    catalog_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Dossier catalogue créé: %s", catalog_dir)

    logger.info("Création session Spark...")
    spark = (
        SparkSession.builder
        .appName("datalake_bronze_catalog")
        .master(cfg["spark"]["master"])
        .config("spark.sql.shuffle.partitions", str(cfg["spark"]["shuffle_partitions"]))
        .config("spark.default.parallelism", str(cfg["spark"]["default_parallelism"]))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("✅ Session Spark créée")

    try:
        # Lecture TSV gz : 2 colonnes (path, size)
        schema = T.StructType([
            T.StructField("path", T.StringType(), False),
            T.StructField("size_bytes", T.LongType(), False),
        ])

        logger.info("Lecture inventaire TSV.gz: %s", inv_gz)
        df = (
            spark.read
            .option("sep", "\t")
            .schema(schema)
            .csv(inv_gz)
        )
        
        total_before_filter = df.count()
        logger.info("Lignes lues: %d", total_before_filter)

        # Filtrer strictement sur ROOT_CORPUS
        logger.info("Filtrage sur root_corpus: %s", root_corpus)
        df = df.filter(F.col("path").startswith(root_corpus))
        total_after_filter = df.count()
        logger.info("Lignes après filtrage: %d (exclus: %d)", 
                    total_after_filter, total_before_filter - total_after_filter)

        # Colonnes dérivées
        logger.info("Calcul colonnes dérivées...")
        df = df.withColumn("basename", F.element_at(F.split(F.col("path"), "/"), -1))

        # Règle générale: versions.xml sous eli/
        df = df.withColumn(
            "is_versions_xml",
            (F.col("basename") == F.lit("versions.xml")) & (F.col("path").contains("/eli/"))
        )

        # Family par patterns de chemin
        df = df.withColumn(
            "family",
            F.when(F.col("path").contains("/LEGI/ARTI/"), F.lit("LEGI_ARTI"))
             .when(F.col("path").contains("/LEGI/TEXT/"), F.lit("LEGI_TEXT"))
             .when(F.col("path").contains("/LEGI/SCTA/"), F.lit("LEGI_SCTA"))
             .otherwise(F.lit("OTHER"))
        )

        # Subtree
        df = df.withColumn(
            "subtree",
            F.when(F.col("path").contains("/code_en_vigueur/"), F.lit("code_en_vigueur"))
             .when(F.col("path").contains("/TNC_en_vigueur/"), F.lit("TNC_en_vigueur"))
             .otherwise(F.lit("unknown"))
        )

        # State et snapshot_dt
        df = df.withColumn("state", F.lit("en_vigueur"))
        df = df.withColumn("snapshot_dt", F.lit(snapshot_dt))

        # Écriture Parquet
        out_parquet = str(catalog_dir / "bronze_files.parquet")
        logger.info("Écriture Parquet: %s", out_parquet)
        df.write.mode("overwrite").parquet(out_parquet)
        logger.info("✅ Parquet écrit")

        # Stats simples (JSON) pour validation rapide
        logger.info("Calcul statistiques...")
        row_count = df.count()
        total_bytes = df.agg(F.sum("size_bytes")).collect()[0][0]
        by_family = {r["family"]: int(r["count"]) for r in df.groupBy("family").count().collect()}
        by_subtree = {r["subtree"]: int(r["count"]) for r in df.groupBy("subtree").count().collect()}
        
        stats = {
            "snapshot_dt": snapshot_dt,
            "root_corpus": root_corpus,
            "rows": row_count,
            "total_bytes": total_bytes,
            "total_size_human": human_readable_size(total_bytes),
            "by_family": by_family,
            "by_subtree": by_subtree,
        }

        stats_file = catalog_dir / "bronze_stats.json"
        logger.info("Écriture stats: %s", stats_file)
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info("=" * 80)
        logger.info("STATISTIQUES BRONZE")
        logger.info("=" * 80)
        logger.info("Lignes: %d", row_count)
        logger.info("Taille totale: %s", human_readable_size(total_bytes))
        logger.info("Par famille: %s", by_family)
        logger.info("Par subtree: %s", by_subtree)
        logger.info("=" * 80)
        
        logger.info("✅ SUCCÈS - BRONZE catalog écrit: %s", out_parquet)
    
    except Exception as e:
        logger.error("❌ ERREUR lors de la création du catalogue BRONZE: %s", e, exc_info=True)
        raise
    
    finally:
        spark.stop()
        logger.info("Session Spark fermée")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Chemin vers datalake_pipeline_v1.yaml")
    args = ap.parse_args()

    main(args.config)