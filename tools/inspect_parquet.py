#!/usr/bin/env python3
"""
Script d'inspection de fichiers/dossiers Parquet.

LANCEMENT : python tools/inspect_parquet.py /home/fred/montage1/"-- Projet RAG Avocats --"/data_main/data/datalake_legifrance_v1/gold/chunks/

Usage:
    python tools/inspect_parquet.py <chemin_absolu_vers_parquet>

Exemple:
    python tools/inspect_parquet.py /home/fred/montage1/"-- Projet RAG Avocats --"/data_main/data/datalake_legifrance_v1/gold/chunks/

    python tools/inspect_parquet.py /home/fred/montage1/"-- Projet RAG Avocats --"/rag_bench/data_sample/gold/chunks_sample/  # pour chunks_code_route_filtered.parquet
Affiche:
- Schéma (colonnes + types)
- Premières lignes
- Statistiques (nombre de lignes, taille)
- Distribution de valeurs pour colonnes catégorielles

Sauvegarde:
- Fichier Markdown dans tools/inspections/<nom_table>_<timestamp>.md

```

---

## ✅ Améliorations apportées

1. **Sauvegarde automatique** dans `tools/inspections/<nom_table>_<timestamp>.md`
2. **Format Markdown** avec tables pour pandas (plus lisible)
3. **Timestamp** dans le nom du fichier pour historique
4. **Message de confirmation** avec le chemin du fichier sauvegardé
5. **Exigence chemin absolu** documentée dans l'usage

---

## 📝 Dossier créé automatiquement

Le script crée `tools/inspections/` automatiquement. Vos rapports seront là :
```
tools/
├── inspect_parquet.py
├── inspections/          ← Créé automatiquement
│   ├── chunks_20260119_143022.md
│   ├── docs_20260119_143045.md
│   └── ...
"""

import sys
import os
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
    from pyspark.sql import SparkSession
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Installez les dépendances: pip install pandas pyarrow pyspark")
    sys.exit(1)


def human_readable_size(size_bytes):
    """Convertit une taille en octets en format lisible (Ko, Mo, Go)."""
    for unit in ['o', 'Ko', 'Mo', 'Go', 'To']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} Po"


def get_directory_size(path):
    """Calcule la taille totale d'un dossier (récursif)."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


def inspect_parquet(parquet_path):
    """
    Inspecte un fichier ou dossier Parquet et affiche les informations clés.
    Sauvegarde le résultat en Markdown.
    
    Args:
        parquet_path: Chemin ABSOLU vers le fichier .parquet ou dossier contenant des Parquet
    """
    # Vérifier que le chemin existe
    path = Path(parquet_path)
    if not path.exists():
        print(f"❌ Erreur: Le chemin '{parquet_path}' n'existe pas")
        print("💡 Utilisez un chemin ABSOLU (ex: /home/fred/montage1/...)")
        sys.exit(1)
    
    # Préparer le fichier de sortie Markdown
    output_dir = Path(__file__).parent / "inspections"
    output_dir.mkdir(exist_ok=True)
    
    # Nom du fichier de sortie : <nom_table>_<timestamp>.md
    table_name = path.name if path.name else "parquet"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{table_name}_{timestamp}.md"
    
    # Buffer pour collecter tout le contenu (console + fichier)
    output_lines = []
    
    def log(message=""):
        """Affiche à la console ET collecte pour le fichier Markdown."""
        print(message)
        output_lines.append(message)
    
    log("=" * 80)
    log(f"🔍 INSPECTION PARQUET: {parquet_path}")
    log("=" * 80)
    
    # Calculer la taille du fichier/dossier
    if path.is_file():
        file_size = os.path.getsize(parquet_path)
        log(f"📦 Type: Fichier unique")
    else:
        file_size = get_directory_size(parquet_path)
        log(f"📦 Type: Dossier (Parquet shardé)")
    
    log(f"💾 Taille: {human_readable_size(file_size)}")
    log(f"🕐 Date inspection: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log()
    
    # Créer une session Spark locale (minimale)
    spark = SparkSession.builder \
        .appName("InspectParquet") \
        .master("local[1]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    # Désactiver les logs verbeux
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        # Lire le Parquet avec Spark
        df_spark = spark.read.parquet(str(parquet_path))
        
        # 1. SCHÉMA (colonnes + types)
        log("📋 SCHÉMA (colonnes et types)")
        log("-" * 80)
        
        # Capturer le schéma en string
        schema_lines = []
        for field in df_spark.schema.fields:
            schema_lines.append(f"- `{field.name}` : {field.dataType.simpleString()}")
        
        for line in schema_lines:
            log(line)
        log()
        
        # 2. STATISTIQUES
        log("📊 STATISTIQUES")
        log("-" * 80)
        row_count = df_spark.count()
        col_count = len(df_spark.columns)
        log(f"Nombre de lignes: {row_count:,}")
        log(f"Nombre de colonnes: {col_count}")
        log()
        
        # 3. PREMIÈRES LIGNES (pandas pour affichage plus lisible)
        log("👀 PREMIÈRES LIGNES (5 premières)")
        log("-" * 80)
        # Convertir en pandas pour affichage (limiter à 5 lignes)
        df_pandas = df_spark.limit(5).toPandas()
        
        # Afficher avec pandas (plus lisible que .show())
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        # Convertir en markdown table
        table_md = df_pandas.to_markdown(index=False)
        if table_md:
            log(table_md)
        else:
            # Fallback si to_markdown() échoue
            log(df_pandas.to_string(index=False))
        log()
        
        # 4. DISTRIBUTION DES VALEURS (colonnes catégorielles)
        log("📈 DISTRIBUTION DES VALEURS (colonnes catégorielles)")
        log("-" * 80)
        
        # Identifier les colonnes probablement catégorielles (type string, faible cardinalité)
        categorical_candidates = []
        for col_name, col_type in df_spark.dtypes:
            if col_type == "string":
                # Compter les valeurs distinctes (limiter à 100 pour être rapide)
                distinct_count = df_spark.select(col_name).distinct().count()
                if distinct_count <= 20:  # Seuil arbitraire pour "catégorielle"
                    categorical_candidates.append((col_name, distinct_count))
        
        if categorical_candidates:
            for col_name, distinct_count in categorical_candidates[:5]:  # Max 5 colonnes
                log(f"\n🏷️  Colonne: '{col_name}' ({distinct_count} valeurs distinctes)")
                
                # Récupérer les valeurs
                value_counts_df = df_spark.groupBy(col_name).count().orderBy("count", ascending=False).limit(10).toPandas()
                
                # Convertir en markdown table
                if not value_counts_df.empty:
                    table_md = value_counts_df.to_markdown(index=False)
                    if table_md:
                        log(table_md)
                    else:
                        log(value_counts_df.to_string(index=False))
        else:
            log("Aucune colonne catégorielle détectée (ou cardinalité trop élevée)")
        
        log()
        log("=" * 80)
        log("✅ Inspection terminée avec succès")
        log("=" * 80)
        
    except Exception as e:
        log(f"❌ Erreur lors de la lecture du Parquet: {e}")
        sys.exit(1)
    
    finally:
        # Arrêter la session Spark
        spark.stop()
    
    # Sauvegarder le résultat en Markdown
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    
    print()
    print(f"💾 Résultat sauvegardé dans: {output_file}")
    print()


if __name__ == "__main__":
    # Vérifier qu'un argument est fourni
    if len(sys.argv) != 2:
        print("❌ Usage: python tools/inspect_parquet.py <chemin_absolu_vers_parquet>")
        print("\nExemples:")
        print('  python tools/inspect_parquet.py /home/fred/montage1/"-- Projet RAG Avocats --"/data_main/data/datalake_legifrance_v1/gold/chunks/')
        print('  python tools/inspect_parquet.py /home/fred/montage1/"-- Projet RAG Avocats --"/data_main/data/datalake_legifrance_v1/silver/docs/')
        sys.exit(1)
    
    # Chemin du Parquet à inspecter
    parquet_path_to_inspect = sys.argv[1]
    
    # Lancer l'inspection
    inspect_parquet(parquet_path_to_inspect)
