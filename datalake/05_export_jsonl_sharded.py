#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_export_jsonl_sharded.py

Export JSONL shardé gzip depuis gold/chunks avec source_id au lieu de source_path.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from datalake.utils import load_yaml


def main(config_path: str) -> None:
    """
    Exporte gold/chunks en JSONL shardé gzip.
    
    Args:
        config_path: Chemin vers datalake_pipeline_v1.yaml
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("DÉMARRAGE: 05_export_jsonl_sharded.py")
    logger.info("=" * 80)
    
    cfg = load_yaml(config_path)
    work_dir = Path(cfg["paths"]["work_dir"])

    gold_chunks_dir = work_dir / cfg["gold"]["output_chunks"]
    source_map_path = work_dir / "gold/source_map/source_map.parquet"
    export_dir = work_dir / cfg["gold"]["export_jsonl_dir"] / "corpus_chunks_v1_jsonl_shards_gz"

    logger.info("Configuration:")
    logger.info("  gold_chunks: %s", gold_chunks_dir)
    logger.info("  source_map: %s", source_map_path)
    logger.info("  output: %s", export_dir)

    logger.info("Création session Spark...")
    spark = (
        SparkSession.builder
        .appName("datalake_export_jsonl_sharded")
        .master(cfg["spark"]["master"])
        .config("spark.sql.shuffle.partitions", "256")  # 256 shards
        .config("spark.default.parallelism", "256")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("✅ Session Spark créée")

    try:
        logger.info("Lecture GOLD chunks...")
        chunks = spark.read.parquet(str(gold_chunks_dir))
        chunks_count = chunks.count()
        logger.info("Chunks chargés: %d", chunks_count)

        logger.info("Lecture source_map...")
        source_map = spark.read.parquet(str(source_map_path))
        source_map_count = source_map.count()
        logger.info("Source map chargée: %d entrées", source_map_count)

        # Join chunks + source_map pour remplacer source_path par source_id
        logger.info("Join chunks + source_map...")
        chunks_with_source_id = chunks.join(
            source_map.select("source_path", "source_id"),
            on="source_path",
            how="left"
        )

        # Vérifier les chunks sans source_id (anomalie)
        missing_source_id = chunks_with_source_id.filter(F.col("source_id").isNull()).count()
        if missing_source_id > 0:
            logger.warning("⚠️ %d chunks sans source_id (source_path non trouvé dans source_map)", missing_source_id)

        # Créer JSONL avec source_id
        logger.info("Construction JSONL (format: chunk_id, text, meta)...")
        export_df = (
            chunks_with_source_id
            .select(
                F.to_json(
                    F.struct(
                        F.col("chunk_id").alias("chunk_id"),
                        F.col("chunk_text").alias("text"),
                        F.struct(
                            F.col("doc_key").alias("doc_key"),
                            F.col("version_key").alias("version_key"),
                            F.col("unit_id").alias("unit_id"),
                            F.col("chunk_index").alias("chunk_index"),
                            F.col("source_id").alias("source_id"),  # source_id au lieu de source_path
                        ).alias("meta"),
                    )
                ).alias("value")
            )
        )

        # Écriture shardée + gzip
        export_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Écriture JSONL shardé gzip (256 shards)...")
        logger.info("Destination: %s", export_dir)
        
        (export_df
         .repartition(256)  # 256 shards
         .write
         .mode("overwrite")
         .option("compression", "gzip")
         .text(str(export_dir))
        )

        # Compter les shards créés
        shard_files = list(export_dir.glob("part-*.gz"))
        shard_count = len(shard_files)

        # Calculer taille totale
        total_size = sum(f.stat().st_size for f in shard_files)
        size_mb = total_size / (1024 * 1024)

        logger.info("=" * 80)
        logger.info("✅ SUCCÈS - Export JSONL shardé créé")
        logger.info("  dossier: %s", export_dir)
        logger.info("  shards: %d fichiers", shard_count)
        logger.info("  taille totale: %.2f Mo", size_mb)
        logger.info("  chunks exportés: %d", chunks_count)
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error("❌ ERREUR lors de l'export JSONL: %s", e, exc_info=True)
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