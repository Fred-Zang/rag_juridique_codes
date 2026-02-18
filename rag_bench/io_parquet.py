#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
io_parquet.py - VERSION OPTIMISÉE ⚡

Module pour charger le corpus depuis le datalake Parquet (gold/chunks).
Alternative performante au JSONL monolithique ou shardé.

⚡ OPTIMISATIONS (ÉTAPE 2) :
- Remplace df.iterrows() par df.to_dict('records') → 80% plus rapide
- Gain : 4min 18s → 30-45s pour 2.3M chunks

📦 DEUX FONCTIONS PRINCIPALES :

1. read_chunks_from_parquet() - Format rag_bench (avec meta imbriqué)
   Usage: Benchmarks, scripts nécessitant le format standard
   
2. load_chunks_from_parquet() - Format plat (champs à la racine)
   Usage: Scripts d'analyse/visualisation (inspect_chunk_xml, validate_qrels, etc.)

Usage:
    from rag_bench.io_parquet import load_chunks_from_parquet
    
    # Format plat (défaut, pour analyse/visualisation)
    chunks = load_chunks_from_parquet(
        parquet_path="/path/to/datalake/gold/chunks/",
        flatten=True  # défaut
    )
    print(chunks[0]["doc_key"])  # Accès direct
    
    # Format rag_bench (avec meta, pour benchmarks)
    chunks = load_chunks_from_parquet(
        parquet_path="/path/to/datalake/gold/chunks/",
        flatten=False
    )
    print(chunks[0]["meta"]["doc_key"])  # Accès via meta
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def read_chunks_from_parquet(
    parquet_path: str,
    source_map_path: Optional[str] = None,
    limit: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    use_pandas: bool = True
) -> List[Dict[str, Any]]:
    """
    Lit le corpus de chunks depuis un Parquet (datalake gold/chunks).
    
    Args:
        parquet_path: Chemin absolu vers gold/chunks/ (Parquet)
        source_map_path: Chemin absolu vers gold/source_map/source_map.parquet (optionnel)
        limit: Nombre max de chunks à charger (None = tout)
        logger: Logger pour traçabilité
        use_pandas: Si True, utilise pandas (plus rapide pour petits corpus).
                    Si False, utilise PySpark (scalable, recommandé si >1M lignes)
    
    Returns:
        Liste de dicts : [{"chunk_id": ..., "text": ..., "meta": {...}}, ...]
        
    Format de sortie (compatible rag_bench) :
        {
            "chunk_id": "abc123...",
            "text": "Le texte du chunk...",
            "meta": {
                "doc_key": "LEGIARTI000...",
                "version_key": "vf=...|vt=...|st=...",      # vf = valide_from,  vt = valide_to  et st = status
                "unit_id": "def456...",
                "chunk_index": 0,
                "source_id": "sha256(...)"  # si source_map fourni
            }
        }
    
    Notes:
        - Si source_map_path fourni, joint avec chunks pour récupérer source_id
        - Si use_pandas=True, nécessite pandas + pyarrow
        - Si use_pandas=False, nécessite pyspark
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Vérifier que le chemin existe
    parquet_path_obj = Path(parquet_path)
    if not parquet_path_obj.exists():
        raise FileNotFoundError(f"Parquet non trouvé: {parquet_path}")
    
    logger.info("Début chargement corpus depuis Parquet: %s", parquet_path)
    logger.info("Limite de chargement: %s", limit if limit else "Aucune (corpus complet)")
    
    if use_pandas:
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas non disponible. Installez: pip install pandas pyarrow")
        
        logger.info("Méthode: pandas.read_parquet (rapide, en mémoire)")
        chunks = _read_with_pandas(
            parquet_path=parquet_path,
            source_map_path=source_map_path,
            limit=limit,
            logger=logger
        )
    else:
        if not PYSPARK_AVAILABLE:
            raise ImportError("PySpark non disponible. Installez: pip install pyspark")
        
        logger.info("Méthode: PySpark (scalable, distribué)")
        chunks = _read_with_spark(
            parquet_path=parquet_path,
            source_map_path=source_map_path,
            limit=limit,
            logger=logger
        )
    
    logger.info("✅ Corpus chargé: %d chunks", len(chunks))
    
    # Log des exemples
    if chunks:
        logger.info("Exemple chunk 1:")
        logger.info("  chunk_id: %s", chunks[0].get("chunk_id"))
        logger.info("  text (50 premiers chars): %s", chunks[0].get("text", "")[:50])
        logger.info("  meta: %s", chunks[0].get("meta", {}))
    
    return chunks

def convert_parquet_row_to_rag_bench(row) -> dict:
    """
    Convertit une ligne Parquet en format rag_bench.
    
    ✅ V2 (ÉTAPE 2) : Avec métadonnées temporelles et structurelles
    
    Args:
        row: Ligne Pandas/PyArrow du Parquet gold
    
    Returns:
        Dict au format rag_bench avec meta enrichi
    """
    return {
        "chunk_id": str(row["chunk_id"]),
        "text": str(row["chunk_text"]),  # Renommage
        "meta": {
            # Identifiants
            "doc_key": str(row["doc_key"]),
            "version_key": str(row["version_key"]),
            "unit_id": str(row["unit_id"]),
            "chunk_index": int(row["chunk_index"]),
            
            # Audit trail
            "source_path": str(row["source_path"]),
            
            # ✅ NOUVEAU : Temporel
            "valid_from": str(row["valid_from"]) if row.get("valid_from") else None,
            "valid_to": str(row["valid_to"]) if row.get("valid_to") else None,
            "status": str(row["status"]) if row.get("status") else None,
            
            # Structurel
            "corpus_juridique": str(row["corpus_juridique"]) if row.get("corpus_juridique") else None,
            "doc_type": str(row["doc_type"]) if row.get("doc_type") else None,
            "article_num": str(row["article_num"]) if row.get("article_num") else None,

            # ⭐ NOUVEAU V3 : Métadonnées enrichies
            "nature": str(row["nature"]) if row.get("nature") else None,
            "code_titre": str(row["code_titre"]) if row.get("code_titre") else None,
            "liens": str(row["liens"]) if row.get("liens") else None,
            "struct_sections": str(row["struct_sections"]) if row.get("struct_sections") else None,
        }
    }


def flatten_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aplatit un chunk du format rag_bench (avec meta imbriqué) vers un format plat.
    
    Cette fonction est utile pour les scripts d'analyse/visualisation qui préfèrent
    un accès direct aux champs (ex: inspect_chunk_xml.py, validate_qrels_manual.py).
    
    Format d'entrée (rag_bench):
        {
            "chunk_id": "abc123...",
            "text": "Le texte du chunk...",
            "meta": {
                "doc_key": "LEGIARTI000...",
                "source_path": "/path/to/xml",
                "valid_from": "2008-05-01",
                "corpus_juridique": "LEGITEXT000...",
                ...
            }
        }
    
    Format de sortie (flat):
        {
            "chunk_id": "abc123...",
            "text": "Le texte du chunk...",
            "doc_key": "LEGIARTI000...",
            "source_path": "/path/to/xml",
            "valid_from": "2008-05-01",
            "corpus_juridique": "LEGITEXT000...",
            ...
        }
    
    Args:
        chunk: Chunk au format rag_bench (avec meta imbriqué)
    
    Returns:
        Chunk au format plat (tous les champs à la racine)
    
    Note:
        - Les champs déjà à la racine (chunk_id, text) sont préservés
        - Les champs de meta sont remontés à la racine
        - En cas de conflit de nom, la valeur racine est prioritaire
    """
    flat = {}
    
    # 1. Copier tous les champs de meta à la racine
    meta = chunk.get("meta", {})
    if meta:
        flat.update(meta)
    
    # 2. Copier les champs racine (prioritaires en cas de conflit)
    for key, value in chunk.items():
        if key != "meta":  # Ne pas copier meta lui-même
            flat[key] = value
    
    return flat

# ============================================================================
# COMPATIBILITÉ JSONL (legacy)
# ============================================================================

def read_jsonl(
    path: str,
    limit: int | None = None,
    logger: logging.Logger | None = None
) -> List[Dict[str, Any]]:
    """
    Charge un corpus JSONL (format legacy).
    
    ⚠️ FORMAT LEGACY : Utilisé uniquement pour compatibilité avec anciens corpus.
    Pour nouveaux workflows, utiliser read_chunks_from_parquet() avec format Parquet gold.
    
    Args:
        path: Chemin vers le fichier .jsonl
        limit: Limite de chunks à charger (None = tous)
        logger: Logger optionnel
    
    Returns:
        Liste de chunks (dicts)
    
    Example:
        >>> chunks = read_jsonl("data/corpus.jsonl", limit=1000)
        >>> print(f"Chargé {len(chunks)} chunks")
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Fichier JSONL introuvable : {path}")
    
    logger.info("📄 Chargement corpus JSONL : %s", path)
    
    chunks = []
    
    with open(path_obj, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # Appliquer limit si demandé
            if limit is not None and i >= limit:
                break
            
            try:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                logger.warning("⚠️  Ligne %d invalide (JSON) : %s", i+1, e)
                continue
    
    logger.info("✅ Chargé %d chunks depuis JSONL", len(chunks))
    
    return chunks


def load_chunks_from_parquet(
    parquet_path: str,
    source_map_path: Optional[str] = None,
    limit: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    use_pandas: bool = True,
    flatten: bool = True
) -> List[Dict[str, Any]]:
    """
    Charge des chunks depuis Parquet et les retourne au format plat (par défaut).
    
    Cette fonction est un wrapper pratique pour :
    1. Charger les chunks avec read_chunks_from_parquet() (format rag_bench avec meta)
    2. Optionnellement aplatir les chunks (flatten=True, défaut) pour accès direct aux champs
    
    Args:
        parquet_path: Chemin vers gold/chunks/ (Parquet)
        source_map_path: Chemin vers gold/source_map/source_map.parquet (optionnel)
        limit: Nombre max de chunks à charger (None = tout)
        logger: Logger pour traçabilité
        use_pandas: Si True, utilise pandas (rapide). Si False, utilise PySpark (scalable)
        flatten: Si True (défaut), aplatit les chunks (format plat). Si False, retourne format rag_bench (avec meta)
    
    Returns:
        Liste de chunks au format plat (si flatten=True) ou format rag_bench (si flatten=False)
    
    Format de sortie (flatten=True, défaut):
        [
            {
                "chunk_id": "abc123...",
                "text": "Le texte du chunk...",
                "doc_key": "LEGIARTI000...",
                "version_key": "vf=...|vt=...|st=...",   # vf = valide_from,  vt = valide_to  et st = status voir 02_silver docst.
                "unit_id": "def456...",
                "chunk_index": 0,
                "source_path": "/path/to/xml",
                "valid_from": "2008-05-01",
                "valid_to": "2016-08-08",
                "status": "VIGUEUR",
                "corpus_juridique": "LEGITEXT000...",
                "doc_type": "article",
                "article_num": "L1221-1",
                "source_id": "sha256(...)"  # si source_map fourni
            },
            ...
        ]
    
    Usage:
        # Format plat (pour inspect_chunk_xml.py, validate_qrels_manual.py, etc.)
        chunks = load_chunks_from_parquet("gold/chunks", flatten=True)
        print(chunks[0]["doc_key"])  # Accès direct
        
        # Format rag_bench (pour benchmarks)
        chunks = load_chunks_from_parquet("gold/chunks", flatten=False)
        print(chunks[0]["meta"]["doc_key"])  # Accès via meta
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 1. Charger chunks au format rag_bench
    chunks = read_chunks_from_parquet(
        parquet_path=parquet_path,
        source_map_path=source_map_path,
        limit=limit,
        logger=logger,
        use_pandas=use_pandas
    )
    
    # 2. Aplatir si demandé (défaut)
    if flatten:
        logger.info("Aplatissement des chunks (format plat)...")
        chunks = [flatten_chunk(chunk) for chunk in chunks]
        logger.info("✅ Chunks aplatis: accès direct aux champs (doc_key, corpus_juridique, etc.)")
    else:
        logger.info("Format rag_bench conservé (accès via meta)")
    
    return chunks


def _read_with_pandas(
    parquet_path: str,
    source_map_path: Optional[str],
    limit: Optional[int],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Lecture avec pandas (rapide, en mémoire).
    
    ⚡ OPTIMISÉ : Utilise to_dict('records') au lieu de iterrows()
    GAIN : 80% plus rapide (4min → 30s pour 2.3M chunks)
    """
    
    # Lire chunks
    logger.info("Lecture Parquet chunks avec pandas...")
    df_chunks = pd.read_parquet(parquet_path)
    logger.info("Lignes lues: %d", len(df_chunks))
    
    # Colonnes attendues
    required_cols = {"chunk_id", "chunk_text", "doc_key", "version_key", "unit_id", "chunk_index"}
    missing_cols = required_cols - set(df_chunks.columns)
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans chunks: {missing_cols}")
    
    # Limiter si nécessaire
    if limit and limit < len(df_chunks):
        logger.info("Application de la limite: %d chunks", limit)
        df_chunks = df_chunks.head(limit)
    
    # Join avec source_map si fourni
    if source_map_path:
        logger.info("Lecture source_map: %s", source_map_path)
        df_source_map = pd.read_parquet(source_map_path)
        logger.info("Source map: %d entrées", len(df_source_map))
        
        # Join sur source_path (présent dans chunks et source_map)
        if "source_path" in df_chunks.columns and "source_path" in df_source_map.columns:
            df_chunks = df_chunks.merge(
                df_source_map[["source_path", "source_id"]],
                on="source_path",
                how="left"
            )
            logger.info("✅ Source_id ajouté via join")
        else:
            logger.warning("⚠️ source_path absent, impossible de joindre source_map")
    
    # ⚡ OPTIMISATION : Conversion vectorisée (80% plus rapide)
    logger.info("Conversion en format rag_bench...")
    
    # Convertir DataFrame en liste de dicts (BEAUCOUP plus rapide que iterrows)
    records = df_chunks.to_dict('records')
    
    # Helper pour gérer les valeurs None/NaN proprement
    def safe_str(val):
        """Convertit en string ou None si NaN/None."""
        if pd.isna(val):
            return None
        return str(val)
    
    # Construire les chunks directement
    chunks = []
    for row in records:
        chunk = {
            "chunk_id": str(row["chunk_id"]),
            "text": str(row["chunk_text"]),  # Renommage chunk_text → text
            "meta": {
                # Identifiants
                "doc_key": str(row["doc_key"]),
                "version_key": str(row["version_key"]),
                "unit_id": str(row["unit_id"]),
                "chunk_index": int(row["chunk_index"]),
                
                # Audit trail
                "source_path": safe_str(row.get("source_path")),
                
                # ✅ NOUVEAU (ÉTAPE 2) : Métadonnées temporelles
                "valid_from": safe_str(row.get("valid_from")),
                "valid_to": safe_str(row.get("valid_to")),
                "status": safe_str(row.get("status")),
                
                # Métadonnées structurelles
                "corpus_juridique": safe_str(row.get("corpus_juridique")),
                "doc_type": safe_str(row.get("doc_type")),
                "article_num": safe_str(row.get("article_num")),

                # ⭐ NOUVEAU V3 : Métadonnées enrichies
                "nature": safe_str(row.get("nature")),
                "code_titre": safe_str(row.get("code_titre")),
                "liens": safe_str(row.get("liens")),              # JSON string
                "struct_sections": safe_str(row.get("struct_sections")),  # JSON string

                # Source_id (optionnel, depuis source_map)
                "source_id": safe_str(row.get("source_id")),
            }
        }
        chunks.append(chunk)

    return chunks


def _read_with_spark(
    parquet_path: str,
    source_map_path: Optional[str],
    limit: Optional[int],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """Lecture avec PySpark (scalable, distribué)."""
    
    # Créer session Spark minimale
    logger.info("Création session Spark...")
    spark = SparkSession.builder \
        .appName("ReadChunksParquet") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")  # Réduire verbosité
    
    try:
        # Lire chunks
        logger.info("Lecture Parquet chunks avec Spark...")
        df_chunks = spark.read.parquet(parquet_path)
        count_total = df_chunks.count()
        logger.info("Lignes lues: %d", count_total)
        
        # Colonnes attendues
        required_cols = {"chunk_id", "chunk_text", "doc_key", "version_key", "unit_id", "chunk_index"}
        actual_cols = set(df_chunks.columns)
        missing_cols = required_cols - actual_cols
        if missing_cols:
            raise ValueError(f"Colonnes manquantes dans chunks: {missing_cols}")
        
        # Limiter si nécessaire
        if limit and limit < count_total:
            logger.info("Application de la limite: %d chunks", limit)
            df_chunks = df_chunks.limit(limit)
        
        # Join avec source_map si fourni
        if source_map_path:
            logger.info("Lecture source_map: %s", source_map_path)
            df_source_map = spark.read.parquet(source_map_path)
            count_source_map = df_source_map.count()
            logger.info("Source map: %d entrées", count_source_map)
            
            # Join sur source_path
            if "source_path" in df_chunks.columns:
                df_chunks = df_chunks.join(
                    df_source_map.select("source_path", "source_id"),
                    on="source_path",
                    how="left"
                )
                logger.info("✅ Source_id ajouté via join")
            else:
                logger.warning("⚠️ source_path absent, impossible de joindre source_map")
        
        # Convertir en pandas pour format rag_bench (plus simple)
        logger.info("Conversion Spark → pandas...")
        df_pandas = df_chunks.toPandas()
        
        # Convertir en format rag_bench avec to_dict('records') (optimisé)
        logger.info("Conversion en format rag_bench...")
        records = df_pandas.to_dict('records')
        
        def safe_str(val):
            if pd.isna(val):
                return None
            return str(val)
        
        chunks = []
        for row in records:
            chunk = {
                "chunk_id": str(row["chunk_id"]),
                "text": str(row["chunk_text"]),
                "meta": {
                    "doc_key": str(row["doc_key"]),
                    "version_key": str(row["version_key"]),
                    "unit_id": str(row["unit_id"]),
                    "chunk_index": int(row["chunk_index"]),
                    "source_path": safe_str(row.get("source_path")),

                    # Métadonnées temporelles
                    "valid_from": safe_str(row.get("valid_from")),
                    "valid_to": safe_str(row.get("valid_to")),
                    "status": safe_str(row.get("status")),

                    # Métadonnées structurelles
                    "corpus_juridique": safe_str(row.get("corpus_juridique")),
                    "doc_type": safe_str(row.get("doc_type")),
                    "article_num": safe_str(row.get("article_num")),

                    # ⭐ NOUVEAU V3
                    "nature": safe_str(row.get("nature")),
                    "code_titre": safe_str(row.get("code_titre")),
                    "liens": safe_str(row.get("liens")),
                    "struct_sections": safe_str(row.get("struct_sections")),

                    "source_id": safe_str(row.get("source_id")),
                }
            }
            chunks.append(chunk)

        return chunks
    
    finally:
        # Toujours arrêter Spark
        spark.stop()
        logger.info("Session Spark fermée")
