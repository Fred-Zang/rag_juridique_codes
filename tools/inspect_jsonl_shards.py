#!/usr/bin/env python3
"""
Script d'inspection d'exports JSONL shardés (gzip).

lancment = python tools/inspect_jsonl_shards.py /home/fred/montage1/"-- Projet RAG Avocats --"/data_main/data/datalake_legifrance_v1/gold/exports/corpus_chunks_v1_jsonl_shards_gz/

Usage:
    python tools/inspect_jsonl_shards.py <chemin_absolu_vers_dossier_shards>

Exemple:
    python tools/inspect_jsonl_shards.py /home/fred/montage1/"-- Projet RAG Avocats --"/data_main/data/datalake_legifrance_v1/gold/exports/corpus_chunks_v1_jsonl_shards_gz/

Affiche:
- Nombre de shards (fichiers part-*.gz)
- Taille totale
- Schéma JSON (clés disponibles)
- Exemples de chunks
- Nombre total de lignes (approximatif)

Sauvegarde:
- Fichier Markdown dans tools/inspections/jsonl_shards_<timestamp>.md
"""

import sys
import os
import gzip
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

try:
    import pandas as pd
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Installez les dépendances: pip install pandas")
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


def inspect_jsonl_shards(shards_dir):
    """
    Inspecte un dossier contenant des shards JSONL gzip.
    Sauvegarde le résultat en Markdown.
    
    Args:
        shards_dir: Chemin ABSOLU vers le dossier contenant les part-*.gz
    """
    # Vérifier que le chemin existe
    path = Path(shards_dir)
    if not path.exists():
        print(f"❌ Erreur: Le chemin '{shards_dir}' n'existe pas")
        print("💡 Utilisez un chemin ABSOLU (ex: /home/fred/montage1/...)")
        sys.exit(1)
    
    if not path.is_dir():
        print(f"❌ Erreur: '{shards_dir}' n'est pas un dossier")
        sys.exit(1)
    
    # Préparer le fichier de sortie Markdown
    output_dir = Path(__file__).parent / "inspections"
    output_dir.mkdir(exist_ok=True)
    
    # Nom du fichier de sortie
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"jsonl_shards_{timestamp}.md"
    
    # Buffer pour collecter tout le contenu (console + fichier)
    output_lines = []
    
    def log(message=""):
        """Affiche à la console ET collecte pour le fichier Markdown."""
        print(message)
        output_lines.append(message)
    
    log("=" * 80)
    log(f"🔍 INSPECTION JSONL SHARDS: {shards_dir}")
    log("=" * 80)
    
    # Lister les fichiers .gz dans le dossier
    shard_files = sorted(path.glob("part-*.gz"))
    
    if not shard_files:
        log("❌ Aucun fichier part-*.gz trouvé dans ce dossier")
        sys.exit(1)
    
    # Calculer la taille totale
    total_size = get_directory_size(shards_dir)
    
    log(f"📦 Nombre de shards: {len(shard_files)}")
    log(f"💾 Taille totale: {human_readable_size(total_size)}")
    log(f"🕐 Date inspection: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log()
    
    # 1. SCHÉMA JSON (lire le premier shard pour détecter les clés)
    log("📋 SCHÉMA JSON (clés disponibles)")
    log("-" * 80)
    
    first_shard = shard_files[0]
    all_keys = set()
    sample_chunks = []
    
    try:
        with gzip.open(first_shard, "rt", encoding="utf-8") as f:
            # Lire les 10 premières lignes pour détecter le schéma
            for i, line in enumerate(f):
                if i >= 10:
                    break
                try:
                    chunk = json.loads(line)
                    all_keys.update(chunk.keys())
                    
                    # Garder 3 exemples
                    if i < 3:
                        sample_chunks.append(chunk)
                except json.JSONDecodeError:
                    continue
        
        # Afficher les clés détectées
        for key in sorted(all_keys):
            log(f"- `{key}`")
        log()
        
    except Exception as e:
        log(f"❌ Erreur lors de la lecture du premier shard: {e}")
        sys.exit(1)
    
    # 2. EXEMPLES DE CHUNKS (3 premiers)
    log("👀 EXEMPLES DE CHUNKS (3 premiers)")
    log("-" * 80)
    
    for i, chunk in enumerate(sample_chunks, 1):
        log(f"\n**Chunk {i}:**")
        log("```json")
        log(json.dumps(chunk, indent=2, ensure_ascii=False))
        log("```")
    log()
    
    # 3. STATISTIQUES PAR SHARD
    log("📊 STATISTIQUES PAR SHARD (échantillon)")
    log("-" * 80)
    
    shard_stats = []
    
    # Analyser les 5 premiers et 5 derniers shards (pour éviter de tout lire)
    sample_shards = list(shard_files[:5]) + list(shard_files[-5:]) if len(shard_files) > 10 else shard_files
    
    for shard_file in sample_shards:
        line_count = 0
        file_size = os.path.getsize(shard_file)
        
        try:
            with gzip.open(shard_file, "rt", encoding="utf-8") as f:
                for line in f:
                    line_count += 1
        except Exception:
            line_count = -1  # Erreur de lecture
        
        shard_stats.append({
            "Fichier": shard_file.name,
            "Lignes": line_count if line_count >= 0 else "Erreur",
            "Taille": human_readable_size(file_size)
        })
    
    # Afficher sous forme de table
    df_stats = pd.DataFrame(shard_stats)
    try:
        table_md = df_stats.to_markdown(index=False)
        log(table_md)
    except Exception:
        log(df_stats.to_string(index=False))
    
    log()
    
    # 4. ESTIMATION DU NOMBRE TOTAL DE LIGNES
    log("📈 ESTIMATION DU NOMBRE TOTAL DE LIGNES")
    log("-" * 80)
    
    # Calculer la moyenne de lignes par shard (sur échantillon)
    valid_counts = [s["Lignes"] for s in shard_stats if isinstance(s["Lignes"], int)]
    
    if valid_counts:
        avg_lines_per_shard = sum(valid_counts) / len(valid_counts)
        estimated_total = int(avg_lines_per_shard * len(shard_files))
        
        log(f"Moyenne de lignes par shard (échantillon): {avg_lines_per_shard:,.0f}")
        log(f"Estimation du nombre total de lignes: {estimated_total:,}")
    else:
        log("Impossible d'estimer (erreur de lecture)")
    
    log()
    log("=" * 80)
    log("✅ Inspection terminée avec succès")
    log("=" * 80)
    
    # Sauvegarder le résultat en Markdown
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    
    print()
    print(f"💾 Résultat sauvegardé dans: {output_file}")
    print()


if __name__ == "__main__":
    # Vérifier qu'un argument est fourni
    if len(sys.argv) != 2:
        print("❌ Usage: python tools/inspect_jsonl_shards.py <chemin_absolu_vers_dossier_shards>")
        print("\nExemples:")
        print('  python tools/inspect_jsonl_shards.py /home/fred/montage1/"-- Projet RAG Avocats --"/data_main/data/datalake_legifrance_v1/gold/exports/corpus_chunks_v1_jsonl_shards_gz/')
        sys.exit(1)
    
    # Chemin du dossier de shards à inspecter
    shards_directory = sys.argv[1]
    
    # Lancer l'inspection
    inspect_jsonl_shards(shards_directory)