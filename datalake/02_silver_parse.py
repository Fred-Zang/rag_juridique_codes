#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_silver_parse.py (V3 - ENRICHISSEMENT COMPLET)

Parse les XML sélectionnés depuis le catalogue BRONZE et produit SILVER enrichi.

Nouveaux champs V3:
- nature: Type de document ("Article", "CODE", "SECTION")
- code_titre: Nom lisible du code ("Code du travail")
- liens: Relations juridiques en JSON
- struct_sections: Sections enfants directes en JSON

Usage:
    python src/datalake/02_silver_parse.py --config configs/datalake_pipeline_v1.yaml

Sortie:
    silver/docs/         - Parquet avec tous les champs enrichis
    silver/doc_versions/ - Parquet temporalité
    silver/quality/      - Parquet qualité parsing
    
### ℹ️ Note sur `version_key` (alias `vf/vt/st`) vs champs “réels” `valid_from/valid_to/status`

- Les sous-champs **`vf`**, **`vt`**, **`st`** présents dans `version_key` sont des **alias** construits à partir des champs “source de vérité” :
  - `vf` ⇔ `valid_from`
  - `vt` ⇔ `valid_to`
  - `st` ⇔ `status`
- Les **valeurs** sont donc identiques en contenu, à une nuance près : `version_key` est une **représentation stringifiée** et peut remplacer l’absence par une chaîne (`"null"`) alors que les champs réels peuvent être `None`/vides.

**Pourquoi dupliquer dans `version_key` ?**
- `version_key` sert de **clé composite compacte** (un seul champ) pour identifier une version juridique :
  - utile pour logs/debug, regroupements (`groupby`), jointures, et traçabilité
  - facilite la stabilité d’identifiants `(doc_key, version_key)` à travers différents formats (Parquet/JSONL) et étapes du pipeline
  - peut servir de **fingerprint** pour caches / contrôles de non-régression

✅ Règle : `valid_from/valid_to/status` restent les champs de référence ; `version_key` est une clé technique dérivée pour simplifier l’indexation et l’audit.

"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

from pyspark.sql import SparkSession, functions as F, types as T

from datalake.utils import load_yaml
from datalake.utils_xml import parse_xml_bytes


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_partition(rows: Iterable[Tuple[str, str, str, str, int, str]]) -> Iterator[Dict]:
    """
    Parse une partition de fichiers XML.

    Args:
        rows: tuples (path, family, subtree, state, size_bytes, ingest_dt)

    Yields:
        Dicts typés (doc/version/quality) avec champ 'row_type'

    Champs doc enrichis V3:
        - nature, code_titre, liens, struct_sections
    """
    for (path, family, subtree, state, size_bytes, ingest_dt) in rows:
        try:
            with open(path, "rb") as f:
                xml_bytes = f.read()

            # Parse avec extraction enrichie V3
            parsed = parse_xml_bytes(xml_bytes, source_path=path)

            doc_key = parsed.doc_key
            title = parsed.title
            body_text = parsed.body_text

            # Ligne "docs" enrichie V3
            yield {
                "row_type": "doc",
                "doc_key": doc_key,
                "family": family,
                "subtree": subtree,
                "state": state,
                "title": title,
                "article_num": parsed.article_num,
                "corpus_juridique": parsed.corpus_juridique,
                # ⭐ NOUVEAU V3
                "nature": parsed.nature,
                "code_titre": parsed.code_titre,
                "liens": parsed.liens,
                "struct_sections": parsed.struct_sections,
                # Contenu et audit
                "body_text": body_text,
                "source_path": path,
                "size_bytes": int(size_bytes),
                "ingest_dt": ingest_dt,
            }

            # Ligne "doc_versions" (inchangée)
            version_key = "default"
            if parsed.valid_from or parsed.valid_to or parsed.status:
                # vf = valide_from,  vt = valide_to  et st = status
                version_key = f"vf={parsed.valid_from or 'null'}|vt={parsed.valid_to or 'null'}|st={parsed.status or 'null'}"

            yield {
                "row_type": "version",
                "doc_key": doc_key,
                "version_key": version_key,
                "valid_from": parsed.valid_from,
                "valid_to": parsed.valid_to,
                "status": parsed.status,
                "ingest_dt": ingest_dt,
            }

            # Ligne "quality" (inchangée)
            yield {
                "row_type": "quality",
                "doc_key": doc_key,
                "source_path": path,
                "parse_ok": True,
                "error_type": None,
                "error_msg": None,
                "text_len": len(body_text or ""),
                "ingest_dt": ingest_dt,
            }

        except (IOError, ValueError, OSError) as e:
            err_msg = f"{type(e).__name__}: {e}"
            logger.warning("Erreur parsing %s: %s", path, err_msg)
            yield {
                "row_type": "quality",
                "doc_key": None,
                "source_path": path,
                "parse_ok": False,
                "error_type": type(e).__name__,
                "error_msg": err_msg,
                "text_len": 0,
                "ingest_dt": ingest_dt,
            }

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            logger.error("Erreur inattendue parsing %s: %s", path, err_msg)
            logger.error(traceback.format_exc())
            yield {
                "row_type": "quality",
                "doc_key": None,
                "source_path": path,
                "parse_ok": False,
                "error_type": type(e).__name__,
                "error_msg": err_msg,
                "text_len": 0,
                "ingest_dt": ingest_dt,
            }


def main(config_path: str) -> None:
    """
    Parse les fichiers XML et produit SILVER enrichi V3.

    Args:
        config_path: Chemin vers datalake_pipeline_v1.yaml
    """
    logger.info("=" * 80)
    logger.info("DEMARRAGE: 02_silver_parse.py (V3 - ENRICHISSEMENT COMPLET)")
    logger.info("=" * 80)

    cfg = load_yaml(config_path)

    work_dir = Path(cfg["paths"]["work_dir"])
    ingest_dt = cfg["project"]["ingest_dt"]

    catalog_path = work_dir / cfg["bronze"]["catalog_subdir"] / "bronze_files.parquet"
    silver_docs_dir = work_dir / cfg["silver"]["output_docs"]
    silver_versions_dir = work_dir / cfg["silver"]["output_versions"]
    silver_quality_dir = work_dir / cfg["silver"]["output_quality"]

    logger.info("Configuration:")
    logger.info("  catalog: %s", catalog_path)
    logger.info("  ingest_dt: %s", ingest_dt)
    logger.info("  output_docs: %s", silver_docs_dir)

    logger.info("Creation session Spark...")
    spark = (
        SparkSession.builder
        .appName("datalake_silver_parse_v3")
        .master(cfg["spark"]["master"])
        .config("spark.sql.shuffle.partitions", str(cfg["spark"]["shuffle_partitions"]))
        .config("spark.default.parallelism", str(cfg["spark"]["default_parallelism"]))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Session Spark creee")

    try:
        logger.info("Lecture catalogue BRONZE: %s", catalog_path)
        bronze = spark.read.parquet(str(catalog_path))
        total_bronze = bronze.count()
        logger.info("Fichiers dans BRONZE: %d", total_bronze)

        families = cfg["silver"]["parse_families"]
        logger.info("Filtrage sur families: %s", families)
        bronze = bronze.filter(F.col("family").isin(families))
        after_family_filter = bronze.count()
        logger.info("Apres filtrage families: %d (exclus: %d)",
                    after_family_filter, total_bronze - after_family_filter)

        # Exclusion versions.xml
        if cfg["silver"].get("exclude_versions_xml", True):
            logger.info("Exclusion des versions.xml")
            bronze = bronze.filter(~F.col("is_versions_xml"))
            after_versions_filter = bronze.count()
            logger.info("Apres exclusion versions.xml: %d (exclus: %d)",
                        after_versions_filter, after_family_filter - after_versions_filter)

        # Garder uniquement XML
        bronze = bronze.filter(F.col("path").endswith(".xml"))
        final_count = bronze.count()
        logger.info("Fichiers XML a parser: %d", final_count)

        # Colonne ingest_dt
        bronze = bronze.withColumn("ingest_dt", F.lit(ingest_dt))

        # Selection des champs necessaires
        sel = bronze.select("path", "family", "subtree", "state", "size_bytes", "ingest_dt")

        # Parse distribue
        logger.info("Lancement parsing distribue...")
        rdd = sel.rdd.mapPartitions(parse_partition)

        # Schema enrichi V3
        schema = T.StructType([
            T.StructField("row_type", T.StringType(), False),
            T.StructField("doc_key", T.StringType(), True),
            T.StructField("family", T.StringType(), True),
            T.StructField("subtree", T.StringType(), True),
            T.StructField("state", T.StringType(), True),
            T.StructField("title", T.StringType(), True),
            T.StructField("article_num", T.StringType(), True),
            T.StructField("corpus_juridique", T.StringType(), True),
            # ⭐ NOUVEAU V3
            T.StructField("nature", T.StringType(), True),
            T.StructField("code_titre", T.StringType(), True),
            T.StructField("liens", T.StringType(), True),           # JSON string
            T.StructField("struct_sections", T.StringType(), True), # JSON string
            # Contenu et audit
            T.StructField("body_text", T.StringType(), True),
            T.StructField("source_path", T.StringType(), True),
            T.StructField("size_bytes", T.LongType(), True),
            # Versions
            T.StructField("version_key", T.StringType(), True),
            T.StructField("valid_from", T.StringType(), True),
            T.StructField("valid_to", T.StringType(), True),
            T.StructField("status", T.StringType(), True),
            # Quality
            T.StructField("parse_ok", T.BooleanType(), True),
            T.StructField("error_type", T.StringType(), True),
            T.StructField("error_msg", T.StringType(), True),
            T.StructField("text_len", T.IntegerType(), True),
            T.StructField("ingest_dt", T.StringType(), False),
        ])

        df = spark.createDataFrame(rdd, schema=schema).cache()

        counts_by_type = {r["row_type"]: int(r["count"]) for r in df.groupBy("row_type").count().collect()}
        logger.info("Lignes parsees par type: %s", counts_by_type)

        # 1) docs enrichi V3
        logger.info("Ecriture SILVER docs (enrichi V3)...")
        docs = df.filter(F.col("row_type") == F.lit("doc")).select(
            "doc_key", "family", "subtree", "state", "title",
            "article_num", "corpus_juridique",
            # ⭐ NOUVEAU V3
            "nature", "code_titre", "liens", "struct_sections",
            # Contenu et audit
            "body_text", "source_path", "size_bytes", "ingest_dt"
        )
        docs_count = docs.count()
        docs.write.mode("overwrite").parquet(str(silver_docs_dir))
        logger.info("SILVER docs ecrit: %d lignes (V3 enrichi)", docs_count)

        # Statistiques enrichissement V3
        nature_present = docs.filter(F.col("nature").isNotNull()).count()
        code_titre_present = docs.filter(F.col("code_titre").isNotNull()).count()
        liens_present = docs.filter(F.col("liens").isNotNull()).count()
        struct_present = docs.filter(F.col("struct_sections").isNotNull()).count()

        logger.info("Statistiques enrichissement V3:")
        logger.info("  nature present: %d / %d (%.1f%%)",
                    nature_present, docs_count, 100.0 * nature_present / max(docs_count, 1))
        logger.info("  code_titre present: %d / %d (%.1f%%)",
                    code_titre_present, docs_count, 100.0 * code_titre_present / max(docs_count, 1))
        logger.info("  liens present: %d / %d (%.1f%%)",
                    liens_present, docs_count, 100.0 * liens_present / max(docs_count, 1))
        logger.info("  struct_sections present: %d / %d (%.1f%%)",
                    struct_present, docs_count, 100.0 * struct_present / max(docs_count, 1))

        # 2) doc_versions (inchange)
        logger.info("Ecriture SILVER doc_versions...")
        versions = df.filter(F.col("row_type") == F.lit("version")).select(
            "doc_key", "version_key", "valid_from", "valid_to", "status", "ingest_dt"
        )
        versions_count = versions.count()
        versions.write.mode("overwrite").parquet(str(silver_versions_dir))
        logger.info("SILVER doc_versions ecrit: %d lignes", versions_count)

        # 3) quality (inchange)
        logger.info("Ecriture SILVER quality...")
        quality = df.filter(F.col("row_type") == F.lit("quality")).select(
            "doc_key", "source_path", "parse_ok", "error_type", "error_msg", "text_len", "ingest_dt"
        )
        quality_count = quality.count()

        parse_ok_count = quality.filter(F.col("parse_ok") == True).count()
        parse_ko_count = quality_count - parse_ok_count

        quality.write.mode("overwrite").parquet(str(silver_quality_dir))
        logger.info("SILVER quality ecrit: %d lignes (OK: %d, KO: %d)",
                    quality_count, parse_ok_count, parse_ko_count)

        logger.info("=" * 80)
        logger.info("SUCCES - SILVER V3 cree (ENRICHISSEMENT COMPLET)")
        logger.info("  docs: %s (%d lignes)", silver_docs_dir, docs_count)
        logger.info("  versions: %s (%d lignes)", silver_versions_dir, versions_count)
        logger.info("  quality: %s (%d lignes, %d erreurs)", silver_quality_dir, quality_count, parse_ko_count)
        logger.info("")
        logger.info("NOUVEAUX CHAMPS V3:")
        logger.info("  - nature: Type document (Article, CODE, SECTION)")
        logger.info("  - code_titre: Nom du code ('Code du travail')")
        logger.info("  - liens: Relations juridiques (JSON)")
        logger.info("  - struct_sections: Sections enfants (JSON)")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("ERREUR lors du parsing SILVER: %s", e, exc_info=True)
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
