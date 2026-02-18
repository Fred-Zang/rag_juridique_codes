#!/usr/bin/env python3
"""
Auto-analyse améliorée avec recherche intelligente des qrels
Adapté selon analyze_results.py fonctionnel

Usage:
    python tools/auto_analyze.py runs/20260122_180321_cdtravail_dense_gpu
    python tools/auto_analyze.py runs/20260122_180321_cdtravail_dense_gpu --corpus /path/to/chunks
"""

import sys
import json
import yaml
from pathlib import Path
import argparse


def find_qrels_file(run_dir: Path) -> Path:
    """
    Cherche intelligemment le fichier qrels
    1. Dans config_resolved.yaml (evaluation.qrels_file)
    2. Dans configs/
    3. Demande à l'utilisateur
    """
    # 1. Essayer de lire config_resolved.yaml dans le run_dir
    config_resolved = run_dir / "config_resolved.yaml"
    if config_resolved.exists():
        try:
            with open(config_resolved, 'r') as f:
                config = yaml.safe_load(f)
            
            # Chercher evaluation.qrels_file
            qrels_file = config.get('evaluation', {}).get('qrels_file')
            if qrels_file:
                qrels_path = Path(qrels_file)
                if qrels_path.exists():
                    print(f"✅ Qrels trouvé depuis config: {qrels_path}")
                    return qrels_path
        except:
            pass
    
    # 2. Essayer les chemins courants
    common_paths = [
        Path("configs/qrels_cdtravail_v2.yaml"),
        Path("configs/qrels_cdtravail_v3_final.yaml"),
        Path("configs/qrels_cdtravail.yaml"),
        Path("configs/qrels.yaml"),
    ]
    
    for path in common_paths:
        if path.exists():
            print(f"✅ Qrels trouvé: {path}")
            return path
    
    # 3. Chercher dans le dossier configs
    configs_dir = Path("configs")
    if configs_dir.exists():
        qrels_files = list(configs_dir.glob("qrels*.yaml"))
        if qrels_files:
            print(f"✅ Qrels trouvé: {qrels_files[0]}")
            return qrels_files[0]
    
    return None


def find_corpus_path(run_dir: Path) -> Path:
    """
    Cherche le chemin du corpus
    1. Dans config_resolved.yaml (data.corpus_jsonl)
    2. Chemin par défaut hardcodé
    """
    # 1. Essayer de lire config_resolved.yaml
    config_resolved = run_dir / "config_resolved.yaml"
    if config_resolved.exists():
        try:
            with open(config_resolved, 'r') as f:
                config = yaml.safe_load(f)
            
            corpus_path = config.get('data', {}).get('corpus_jsonl')
            if corpus_path:
                corpus_path = Path(corpus_path)
                if corpus_path.exists():
                    print(f"✅ Corpus trouvé depuis config: {corpus_path}")
                    return corpus_path
        except:
            pass
    
    # 2. Chemin par défaut
    default_corpus = Path("/home/fred/montage1/-- Projet RAG Avocats --/data_main/data/datalake_legifrance_v1/gold/chunks")
    if default_corpus.exists():
        print(f"✅ Corpus trouvé (défaut): {default_corpus}")
        return default_corpus
    
    return None


def auto_analyze_run(run_dir: Path, corpus_path: Path = None, qrels_file: Path = None):
    """
    Analyse automatiquement tous les résultats d'un run
    """
    run_dir = Path(run_dir)
    selected_result_file = None

    # Accepter aussi un chemin direct vers un fichier *_results.jsonl
    if run_dir.is_file():
        if run_dir.name.endswith("_results.jsonl"):
            selected_result_file = run_dir
            run_dir = run_dir.parent
        else:
            print(f"❌ Chemin fichier non supporté: {run_dir}")
            print("   Attendu: un dossier de run OU un fichier *_results.jsonl")
            return False    
    
    if not run_dir.exists():
        print(f"❌ Dossier non trouvé: {run_dir}")
        return False
    
    print(f"\n{'='*80}")
    print(f"🔍 AUTO-ANALYSE DU RUN: {run_dir.name}")
    print(f"{'='*80}\n")
    
    # Trouver corpus si non fourni
    if corpus_path is None:
        corpus_path = find_corpus_path(run_dir)
        if corpus_path is None:
            print(f"\n❌ ERREUR: Corpus introuvable !")
            print(f"Merci de spécifier avec: --corpus /path/to/chunks\n")
            return False
    else:
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            print(f"❌ Corpus non trouvé: {corpus_path}")
            return False
    
    # Trouver qrels si non fourni
    if qrels_file is None:
        qrels_file = find_qrels_file(run_dir)
        if qrels_file is None:
            print(f"\n⚠️ AVERTISSEMENT: Fichier qrels introuvable !")
            print(f"L'analyse continuera mais sans comparaison qrels.")
            print(f"Spécifiez avec: --qrels /path/to/qrels.yaml\n")
    else:
        qrels_file = Path(qrels_file)
        if not qrels_file.exists():
            print(f"⚠️ Qrels non trouvé: {qrels_file}")
            qrels_file = None
    
    # Chercher tous les fichiers *_results.jsonl (ou analyser uniquement celui fourni)
    if selected_result_file is not None:
        result_files = [selected_result_file]
    else:
        result_files = list(run_dir.glob("*_results.jsonl"))
    
    if not result_files:
        print(f"❌ Aucun fichier *_results.jsonl trouvé dans {run_dir}")
        return False
    
    print(f"📋 Fichiers détectés: {len(result_files)}")
    for f in result_files:
        print(f"   - {f.name}")
    print()
    
    # Import de analyze_results depuis analyze_results.py
    try:
        # Essayer d'importer depuis tools/
        sys.path.insert(0, str(Path(__file__).parent))
        from analyze_results import analyze_results
    except ImportError:
        print("❌ Impossible d'importer analyze_results.py")
        print("Assurez-vous que analyze_results.py est dans tools/")
        return False
    
    # Analyser chaque fichier
    success_count = 0
    for result_file in result_files:
        print(f"\n{'─'*80}")
        print(f"📊 Analyse de {result_file.name}")
        print(f"{'─'*80}")
        
        try:
            analyze_results(
                results_file=str(result_file),
                corpus_path=str(corpus_path),
                qrels_file=str(qrels_file) if qrels_file else None,
                top_k=10
            )
            success_count += 1
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse de {result_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ Analyse terminée: {success_count}/{len(result_files)} fichiers traités")
    print(f"{'='*80}\n")
    
    # Lister les fichiers générés
    print("📁 Fichiers générés:")
    for pattern in ["metrics_*.json", "analyse_*.html"]:
        for f in sorted(run_dir.glob(pattern)):
            print(f"   - {f.name}")
    print()
    
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Auto-analyse tous les résultats d'un run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Auto-détection corpus + qrels
  python tools/auto_analyze.py runs/20260122_180321_cdtravail_dense_gpu
  
  # Spécifier corpus
  python tools/auto_analyze.py runs/20260122_180321_cdtravail_dense_gpu \\
      --corpus ~/montage1/"-- Projet RAG Avocats --"/data_main/data/datalake_legifrance_v1/gold/chunks
  
  # Spécifier corpus + qrels
  python tools/auto_analyze.py runs/20260122_180321_cdtravail_dense_gpu \\
      --corpus ~/path/to/chunks \\
      --qrels configs/qrels_cdtravail_v3_final.yaml

Le script cherche automatiquement:
  1. Corpus dans config_resolved.yaml (data.corpus_jsonl)
  2. Qrels dans config_resolved.yaml (evaluation.qrels_file)
  3. Sinon chemins par défaut dans configs/
"""
    )
    
    parser.add_argument("run_dir", type=str, help="Dossier du run à analyser")
    parser.add_argument("--corpus", type=str, help="Chemin vers corpus (Parquet chunks)")
    parser.add_argument("--qrels", type=str, help="Chemin vers qrels.yaml")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    corpus_path = Path(args.corpus) if args.corpus else None
    qrels_file = Path(args.qrels) if args.qrels else None
    
    success = auto_analyze_run(run_dir, corpus_path, qrels_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
