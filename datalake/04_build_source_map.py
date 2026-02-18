#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_build_source_map.py

Construit une table de correspondance SOURCE:
- source_id (hash SHA-256 du source_path)
- source_path (chemin complet local)
- doc_key, version_key (pour traçabilité)
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from datalake.utils import load_yaml


def main(config_path: str) -> None:
    """
    Construit la source_map depuis gold/chunks.
    
    Args:
        config_path: Chemin vers datalake_pipeline_v1.yaml
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("DÉMARRAGE: 04_build_source_map.py")
    logger.info("=" * 80)
    
    cfg = load_yaml(config_path)
    work_dir = Path(cfg["paths"]["work_dir"])

    gold_chunks_dir = work_dir / cfg["gold"]["output_chunks"]
    out_dir = work_dir / "gold/source_map"

    logger.info("Configuration:")
    logger.info("  gold_chunks: %s", gold_chunks_dir)
    logger.info("  output: %s", out_dir)

    logger.info("Création session Spark...")
    spark = (
        SparkSession.builder
        .appName("datalake_build_source_map")
        .master(cfg["spark"]["master"])
        .config("spark.sql.shuffle.partitions", str(cfg["spark"]["shuffle_partitions"]))
        .config("spark.default_parallelism", str(cfg["spark"]["default_parallelism"]))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("✅ Session Spark créée")

    try:
        logger.info("Lecture GOLD chunks: %s", gold_chunks_dir)
        df = spark.read.parquet(str(gold_chunks_dir))
        total_chunks = df.count()
        logger.info("Chunks chargés: %d", total_chunks)

        # Calcul source_id = SHA256(source_path)
        logger.info("Calcul source_map (distinct source_path → source_id)...")
        df_map = (
            df.select(
                F.sha2(F.col("source_path"), 256).alias("source_id"),
                F.col("source_path"),
                F.col("doc_key"),
                F.col("version_key"),
            )
            .dropna(subset=["source_path"])
            .dropDuplicates(["source_id"])
        )

        source_map_count = df_map.count()
        logger.info("Source map entries: %d", source_map_count)

        # Vérifier collisions (même source_id pour différents source_path)
        logger.info("Vérification des collisions SHA256...")
        collision_check = (
            df.select("source_path")
            .dropna()
            .distinct()
            .withColumn("source_id", F.sha2(F.col("source_path"), 256))
            .groupBy("source_id")
            .count()
            .filter(F.col("count") > 1)
        )
        
        collision_count = collision_check.count()
        if collision_count > 0:
            logger.warning("⚠️ COLLISIONS DÉTECTÉES: %d source_id ont plusieurs source_path", collision_count)
            logger.warning("Affichage des 10 premières collisions:")
            collision_check.show(10, truncate=False)
        else:
            logger.info("✅ Aucune collision détectée")

        # Écriture
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "source_map.parquet"
        
        logger.info("Écriture source_map: %s", out_path)
        df_map.write.mode("overwrite").parquet(str(out_path))

        logger.info("=" * 80)
        logger.info("✅ SUCCÈS - Source map créée")
        logger.info("  fichier: %s", out_path)
        logger.info("  lignes: %d", source_map_count)
        logger.info("  collisions: %d", collision_count)
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error("❌ ERREUR lors de la création source_map: %s", e, exc_info=True)
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