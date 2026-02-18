#!/usr/bin/env python3
"""
Script de comparaison des benchmarks BM25 vs Dense vs Hybrid
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

def load_metrics(run_dir: Path) -> Dict[str, Any]:
    """Charge les métriques depuis metrics.json"""
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return {}
    
    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)

def find_latest_runs() -> Dict[str, Path]:
    """Trouve les derniers runs pour chaque configuration"""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return {}
    
    runs = {
        "bm25": None,
        "dense": None,
        "hybrid": None
    }
    
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        
        name = run_dir.name.lower()
        if "bm25" in name and runs["bm25"] is None:
            runs["bm25"] = run_dir
        elif "dense" in name and "hybrid" not in name and runs["dense"] is None:
            runs["dense"] = run_dir
        elif "hybrid" in name and runs["hybrid"] is None:
            runs["hybrid"] = run_dir
    
    return {k: v for k, v in runs.items() if v is not None}

def print_comparison(metrics_by_method: Dict[str, Dict[str, Any]]):
    """Affiche une comparaison tabulaire des métriques"""
    
    print("\n" + "="*80)
    print("COMPARAISON DES MÉTRIQUES - CODE DU TRAVAIL")
    print("="*80 + "\n")
    
    # Métriques principales
    main_metrics = ["recall", "mrr", "ndcg"]
    k_values = [1, 3, 5, 10, 20]
    
    for metric in main_metrics:
        print(f"\n📊 {metric.upper()}")
        print("-" * 80)
        print(f"{'k':<6} {'BM25':<15} {'Dense':<15} {'Hybrid':<15} {'Meilleur':<15}")
        print("-" * 80)
        
        for k in k_values:
            key = f"{metric}@{k}"
            values = {}
            
            for method, data in metrics_by_method.items():
                if key in data.get("aggregated", {}):
                    values[method] = data["aggregated"][key]
            
            if not values:
                continue
            
            # Trouver le meilleur
            best_method = max(values.items(), key=lambda x: x[1])[0]
            
            bm25_val = values.get("bm25", 0.0)
            dense_val = values.get("dense", 0.0)
            hybrid_val = values.get("hybrid", 0.0)
            
            # Formater avec * pour le meilleur
            bm25_str = f"{bm25_val:.4f}{'*' if best_method == 'bm25' else ' '}"
            dense_str = f"{dense_val:.4f}{'*' if best_method == 'dense' else ' '}"
            hybrid_str = f"{hybrid_val:.4f}{'*' if best_method == 'hybrid' else ' '}"
            
            print(f"@{k:<5} {bm25_str:<15} {dense_str:<15} {hybrid_str:<15} {best_method:<15}")
    
    print("\n" + "="*80)
    print("* = Meilleur score pour ce k")
    print("="*80 + "\n")

def main():
    runs = find_latest_runs()
    
    if not runs:
        print("❌ Aucun run trouvé dans ./runs/")
        sys.exit(1)
    
    print("\n📂 Runs détectés:")
    for method, run_dir in runs.items():
        print(f"  - {method.upper()}: {run_dir.name}")
    
    metrics_by_method = {}
    for method, run_dir in runs.items():
        metrics = load_metrics(run_dir)
        if metrics:
            metrics_by_method[method] = metrics
    
    if not metrics_by_method:
        print("❌ Aucune métrique trouvée")
        sys.exit(1)
    
    print_comparison(metrics_by_method)
    
    # Résumé
    print("\n📈 RÉSUMÉ:")
    print("-" * 80)
    
    # Compter les victoires
    wins = {"bm25": 0, "dense": 0, "hybrid": 0}
    main_metrics = ["recall", "mrr", "ndcg"]
    k_values = [1, 3, 5, 10, 20]
    
    for metric in main_metrics:
        for k in k_values:
            key = f"{metric}@{k}"
            values = {
                method: data["aggregated"].get(key, 0.0)
                for method, data in metrics_by_method.items()
            }
            if values:
                best = max(values.items(), key=lambda x: x[1])[0]
                wins[best] += 1
    
    print(f"Victoires (sur {len(main_metrics) * len(k_values)} métriques):")
    for method, count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method.upper()}: {count}")
    
    print("-" * 80 + "\n")

if __name__ == "__main__":
    main()