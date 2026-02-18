#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_gold_build.py (V3 - ENRICHISSEMENT COMPLET)

Construit GOLD enrichi a partir de SILVER:
- gold/units : avec temporalite + metadonnees structurelles + nouveaux champs V3
- gold/chunks : avec tous les champs enrichis

Nouveaux champs V3 propages:
- nature: Type de document ("Article", "CODE", "SECTION")
- code_titre: Nom lisible du code ("Code du travail")
- liens: Relations juridiques en JSON
- struct_sections: Sections enfants directes en JSON

Usage:
    python src/datalake/03_gold_build.py --config configs/datalake_pipeline_v1.yaml

Sortie:
    gold/units/   - Parquet avec tous les champs
    gold/chunks/  - Parquet chunke avec metadonnees
    gold/exports/ - JSONL pour Elasticsearch
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from pyspark.sql import SparkSession, functions as F, types as T

from datalake.utils import load_yaml


def stable_hash(s: str) -> str:
    """Hash stable court pour IDs deterministes."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def normalize_ws(text: str) -> str:
    """Normalisation legere des espaces."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join([" ".join(line.split()) for line in text.split("\n")])
    return text.strip()


def chunk_text_charwise(
    text: str,
    max_chars: int,
    overlap: int,
    min_chars: int,
    sentence_aware: bool = True
) -> List[Tuple[int, int, str]]:
    """
    Chunking deterministe base sur caracteres, avec option sentence-aware.

    Args:
        text: Texte a decouper
        max_chars: Taille max d'un chunk
        overlap: Chevauchement entre chunks
        min_chars: Taille min d'un chunk
        sentence_aware: Si True, essaie de couper aux limites de phrases

    Returns:
        Liste de (start, end, chunk_text)
    """
    if not text:
        return []

    chunks = []
    n = len(text)
    start = 0

    while start < n:
        end = min(n, start + max_chars)

        if sentence_aware and end < n:
            search_start = start + int(max_chars * 0.8)
            search_zone = text[search_start:end]

            last_period = search_zone.rfind(".")
            last_exclaim = search_zone.rfind("!")
            last_question = search_zone.rfind("?")
            last_newline = search_zone.rfind("\n")

            break_point = max(last_period, last_exclaim, last_question, last_newline)

            if break_point > 0:
                end = search_start + break_point + 1

        chunk = text[start:end].strip()

        if chunk and len(chunk) < min_chars and chunks:
            prev_start, prev_end, prev_txt = chunks[-1]
            merged_txt = (prev_txt + "\n" + chunk).strip()
            chunks[-1] = (prev_start, end, merged_txt)
            break

        if chunk:
            chunks.append((start, end, chunk))

        if end >= n:
            break

        start = max(0, end - overlap)

    return chunks


def chunks_partition(
    rows: Iterable[Tuple[str, str, str, str, str, str, str,
                         str, str, str, str, str, str,
                         str, str, str, str]]
) -> Iterator[Dict]:
    """
    Signature enrichie V3.

    rows: (unit_id, doc_key, version_key, family, title, article_num, text,
           source_path, valid_from, valid_to, status, corpus_juridique, doc_type,
           nature, code_titre, liens, struct_sections)

    Genere les lignes de chunks enrichis.
    """
    for (unit_id, doc_key, version_key, family, title, article_num, text,
         source_path, valid_from, valid_to, status, corpus_juridique, doc_type,
         nature, code_titre, liens, struct_sections) in rows:

        text_norm = normalize_ws(text) if text else ""
        if not text_norm:
            continue

        max_chars = chunks_partition.max_chars
        overlap = chunks_partition.overlap_chars
        min_chars = chunks_partition.min_chars

        parts = chunk_text_charwise(
            text_norm,
            max_chars=max_chars,
            overlap=overlap,
            min_chars=min_chars,
            sentence_aware=True
        )

        for idx, (start, end, chunk_txt) in enumerate(parts):
            chunk_id = stable_hash(f"{unit_id}|{idx}|max={max_chars}|ov={overlap}|min={min_chars}")

            yield {
                # Identifiants
                "chunk_id": chunk_id,
                "unit_id": unit_id,
                "doc_key": doc_key,
                "version_key": version_key,
                "chunk_index": idx,

                # Position
                "start_char": int(start),
                "end_char": int(end),

                # Contenu
                "chunk_text": chunk_txt,

                # Metadonnees temporelles
                "valid_from": valid_from,
                "valid_to": valid_to,
                "status": status,

                # Metadonnees structurelles
                "corpus_juridique": corpus_juridique,
                "doc_type": doc_type,
                "article_num": article_num,

                # ⭐ NOUVEAU V3: Metadonnees enrichies
                "nature": nature,
                "code_titre": code_titre,
                "liens": liens,
                "struct_sections": struct_sections,

                # Audit trail
                "source_path": source_path,
            }


def main(config_path: str) -> None:
    """
    Construit GOLD enrichi V3 depuis SILVER.

    Args:
        config_path: Chemin vers datalake_pipeline_v1.yaml
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("DEMARRAGE: 03_gold_build.py (V3 - ENRICHISSEMENT COMPLET)")
    logger.info("=" * 80)

    cfg = load_yaml(config_path)

    work_dir = Path(cfg["paths"]["work_dir"])
    silver_docs_dir = work_dir / cfg["silver"]["output_docs"]
    silver_versions_dir = work_dir / cfg["silver"]["output_versions"]
    gold_units_dir = work_dir / cfg["gold"]["output_units"]
    gold_chunks_dir = work_dir / cfg["gold"]["output_chunks"]
    export_dir = work_dir / cfg["gold"]["export_jsonl_dir"]
    export_filename = cfg["gold"]["export_jsonl_filename"]

    chunk_cfg = cfg["chunking"]
    max_chars = int(chunk_cfg["max_chars"])
    overlap_chars = int(chunk_cfg["overlap_chars"])
    min_chars = int(chunk_cfg["min_chars"])

    logger.info("Configuration chunking:")
    logger.info("  max_chars: %d", max_chars)
    logger.info("  overlap_chars: %d", overlap_chars)
    logger.info("  min_chars: %d", min_chars)

    logger.info("Creation session Spark...")
    spark = (
        SparkSession.builder
        .appName("datalake_gold_build_v3")
        .master(cfg["spark"]["master"])
        .config("spark.sql.shuffle.partitions", str(cfg["spark"]["shuffle_partitions"]))
        .config("spark.default.parallelism", str(cfg["spark"]["default_parallelism"]))
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.parquet.columnarReaderBatchSize", "2048")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Session Spark creee")

    try:
        logger.info("Lecture SILVER docs: %s", silver_docs_dir)
        docs = spark.read.parquet(str(silver_docs_dir))
        docs_count = docs.count()
        logger.info("Docs charges: %d", docs_count)

        logger.info("Lecture SILVER versions: %s", silver_versions_dir)
        versions = spark.read.parquet(str(silver_versions_dir))
        versions_count = versions.count()
        logger.info("Versions chargees: %d", versions_count)

        # Agregation versions enrichie
        logger.info("Agregation versions par doc_key (avec temporalite)...")
        vers1 = (
            versions
            .groupBy("doc_key")
            .agg(
                F.first("version_key", ignorenulls=True).alias("version_key"),
                F.first("valid_from", ignorenulls=True).alias("valid_from"),
                F.first("valid_to", ignorenulls=True).alias("valid_to"),
                F.first("status", ignorenulls=True).alias("status"),
            )
        )

        # Construction units enrichie V3
        logger.info("Construction des units (join docs + versions + enrichissement V3)...")
        units = (
            docs.join(vers1, on="doc_key", how="left")
            .withColumn("version_key", F.coalesce(F.col("version_key"), F.lit("default")))
            .withColumn("unit_id", F.sha1(F.concat_ws("|", F.col("doc_key"), F.col("version_key"))))
            .withColumn("text", F.col("body_text"))

            # corpus_juridique : priorite a la valeur SILVER (XML), fallback sur source_path
            .withColumn("corpus_juridique",
                F.coalesce(
                    F.when(F.col("corpus_juridique").isNotNull() & (F.col("corpus_juridique") != ""),
                           F.col("corpus_juridique")),
                    F.regexp_extract(F.col("source_path"), r'(LEGITEXT\d+)', 1)
                )
            )

            # Inference doc_type depuis doc_key
            .withColumn("doc_type",
                F.when(F.col("doc_key").startswith("LEGIARTI"), F.lit("article"))
                 .when(F.col("doc_key").startswith("LEGISCTA"), F.lit("section"))
                 .when(F.col("doc_key").startswith("LEGITEXT"), F.lit("code"))
                 .when(F.col("doc_key").startswith("JORFTEXT"), F.lit("jorf_text"))
                 .when(F.col("doc_key").startswith("JORFARTI"), F.lit("jorf_article"))
                 .otherwise(F.lit("unknown"))
            )

            # Select enrichi V3
            .select(
                # Identifiants
                "unit_id", "doc_key", "version_key", "family",

                # Metadonnees structurelles
                "title", "article_num", "corpus_juridique", "doc_type",

                # ⭐ NOUVEAU V3
                "nature", "code_titre", "liens", "struct_sections",

                # Contenu
                "text",

                # Audit
                "source_path", "state", "subtree", "ingest_dt",

                # Temporalite
                "valid_from", "valid_to", "status",
            )
        )

        logger.info("Ecriture GOLD units: %s", gold_units_dir)
        units.write.mode("overwrite").parquet(str(gold_units_dir))
        units_count = units.count()
        logger.info("GOLD units ecrit: %d lignes (enrichi V3)", units_count)

        # Chunking distribue
        logger.info("Lancement chunking distribue (enrichi V3)...")
        chunks_partition.max_chars = max_chars
        chunks_partition.overlap_chars = overlap_chars
        chunks_partition.min_chars = min_chars

        # Select enrichi pour chunking V3
        sel = units.select(
            "unit_id", "doc_key", "version_key", "family", "title", "article_num",
            "text", "source_path",
            "valid_from", "valid_to", "status",
            "corpus_juridique", "doc_type",
            # ⭐ NOUVEAU V3
            "nature", "code_titre", "liens", "struct_sections"
        )
        rdd = sel.rdd.mapPartitions(chunks_partition)

        # Schema enrichi V3
        schema = T.StructType([
            # Identifiants
            T.StructField("chunk_id", T.StringType(), False),
            T.StructField("unit_id", T.StringType(), False),
            T.StructField("doc_key", T.StringType(), False),
            T.StructField("version_key", T.StringType(), False),
            T.StructField("chunk_index", T.IntegerType(), False),

            # Position
            T.StructField("start_char", T.IntegerType(), False),
            T.StructField("end_char", T.IntegerType(), False),

            # Contenu
            T.StructField("chunk_text", T.StringType(), False),

            # Metadonnees temporelles
            T.StructField("valid_from", T.StringType(), True),
            T.StructField("valid_to", T.StringType(), True),
            T.StructField("status", T.StringType(), True),

            # Metadonnees structurelles
            T.StructField("corpus_juridique", T.StringType(), True),
            T.StructField("doc_type", T.StringType(), True),
            T.StructField("article_num", T.StringType(), True),

            # ⭐ NOUVEAU V3
            T.StructField("nature", T.StringType(), True),
            T.StructField("code_titre", T.StringType(), True),
            T.StructField("liens", T.StringType(), True),           # JSON string
            T.StructField("struct_sections", T.StringType(), True), # JSON string

            # Audit trail
            T.StructField("source_path", T.StringType(), True),
        ])

        chunks_df = spark.createDataFrame(rdd, schema=schema)

        logger.info("Ecriture GOLD chunks: %s", gold_chunks_dir)
        chunks_df.write.mode("overwrite").parquet(str(gold_chunks_dir))
        chunks_count = chunks_df.count()
        logger.info("GOLD chunks ecrit: %d lignes (enrichi V3)", chunks_count)

        # Statistiques sur enrichissement V3
        logger.info("Statistiques enrichissement V3:")
        corpus_juridique_present = chunks_df.filter(F.col("corpus_juridique").isNotNull()).count()
        valid_from_present = chunks_df.filter(F.col("valid_from").isNotNull()).count()
        article_num_present = chunks_df.filter(F.col("article_num").isNotNull()).count()
        nature_present = chunks_df.filter(F.col("nature").isNotNull()).count()
        code_titre_present = chunks_df.filter(F.col("code_titre").isNotNull()).count()
        liens_present = chunks_df.filter(F.col("liens").isNotNull()).count()
        struct_present = chunks_df.filter(F.col("struct_sections").isNotNull()).count()

        logger.info("  corpus_juridique present: %d / %d (%.1f%%)",
                    corpus_juridique_present, chunks_count, 100.0 * corpus_juridique_present / max(chunks_count, 1))
        logger.info("  valid_from present: %d / %d (%.1f%%)",
                    valid_from_present, chunks_count, 100.0 * valid_from_present / max(chunks_count, 1))
        logger.info("  article_num present: %d / %d (%.1f%%)",
                    article_num_present, chunks_count, 100.0 * article_num_present / max(chunks_count, 1))
        logger.info("  nature present: %d / %d (%.1f%%)",
                    nature_present, chunks_count, 100.0 * nature_present / max(chunks_count, 1))
        logger.info("  code_titre present: %d / %d (%.1f%%)",
                    code_titre_present, chunks_count, 100.0 * code_titre_present / max(chunks_count, 1))
        logger.info("  liens present: %d / %d (%.1f%%)",
                    liens_present, chunks_count, 100.0 * liens_present / max(chunks_count, 1))
        logger.info("  struct_sections present: %d / %d (%.1f%%)",
                    struct_present, chunks_count, 100.0 * struct_present / max(chunks_count, 1))

        # Export JSONL enrichi V3
        logger.info("Export JSONL: %s", export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / export_filename

        export_df = (
            chunks_df
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
                            F.col("source_path").alias("source_path"),
                            # Metadonnees enrichies
                            F.col("corpus_juridique").alias("corpus_juridique"),
                            F.col("doc_type").alias("doc_type"),
                            F.col("article_num").alias("article_num"),
                            F.col("valid_from").alias("valid_from"),
                            F.col("valid_to").alias("valid_to"),
                            F.col("status").alias("status"),
                            # ⭐ NOUVEAU V3
                            F.col("nature").alias("nature"),
                            F.col("code_titre").alias("code_titre"),
                            F.col("liens").alias("liens"),
                            F.col("struct_sections").alias("struct_sections"),
                        ).alias("meta"),
                    )
                ).alias("value")
            )
        )

        tmp_dir = export_dir / (export_filename + ".tmp_dir")
        (export_df
         .coalesce(1)
         .write
         .mode("overwrite")
         .text(str(tmp_dir))
        )

        part_files = list(Path(tmp_dir).glob("part-*"))
        if part_files:
            if export_path.exists():
                export_path.unlink()
            part_files[0].rename(export_path)

        # Nettoyage
        for f in Path(tmp_dir).glob("*"):
            if f.is_file():
                f.unlink()
        Path(tmp_dir).rmdir()

        logger.info("=" * 80)
        logger.info("SUCCES - GOLD V3 cree (ENRICHISSEMENT COMPLET)")
        logger.info("  units: %s (%d lignes)", gold_units_dir, units_count)
        logger.info("  chunks: %s (%d lignes)", gold_chunks_dir, chunks_count)
        logger.info("  export JSONL: %s", export_path)
        logger.info("")
        logger.info("ENRICHISSEMENTS V3:")
        logger.info("  - Metadonnees temporelles: valid_from, valid_to, status")
        logger.info("  - Metadonnees structurelles: corpus_juridique, doc_type, article_num")
        logger.info("  - NOUVEAU: nature, code_titre, liens (JSON), struct_sections (JSON)")
        logger.info("  - Filtrage temporel: maintenant possible via as_of")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("ERREUR lors de la creation GOLD: %s", e, exc_info=True)
        raise

    finally:
        spark.stop()
        logger.info("Session Spark fermee")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Chemin vers datalake_pipeline_v1.yaml")
    args = ap.parse_args()

    main(args.config)
