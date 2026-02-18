#!/usr/bin/env python3
"""
Création de qrels v3 par consensus des retrievers
Phase 4 - Qrels de qualité

Usage:

python tools/create_qrels_v3.py \
  --runs-dir runs/20260124_200414_cdtravail_hybrid_rrf \
  --corpus "/home/fred/montage1/-- Projet RAG Avocats --/data_main/data/datalake_legifrance_v1/gold/chunks" \
  --output configs/qrels_cdtravail_v4_candidates.yaml \
  --top-k 10 \
  --min-retrievers 2 \
  --max-docs-per-query 6 \
  --exclude-abroge

    
Stratégie:
1. Analyser les résultats BM25, Dense, Hybrid pour les mêmes queries
2. Extraire documents qui apparaissent dans top-k de plusieurs retrievers
3. Filtrer articles abrogés
4. Proposer candidats pour validation manuelle
5. Export YAML
"""

import sys
import json
import yaml
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import pyarrow.parquet as pq
import pandas as pd

def load_results_jsonl(jsonl_file):
    """Charge résultats JSONL."""
    results = []
    if not Path(jsonl_file).exists():
        return results
    
    with open(jsonl_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))

    return results


def group_by_query(results):
    """Groupe résultats par query_id."""
    queries = defaultdict(list)
    for r in results:
        qid = r.get('query_id')
        queries[qid].append(r)
    return queries

def extract_top_k(results, k=10):
    """Extrait top-k doc_keys avec scores."""
    # Trier par rank ou score
    sorted_results = sorted(results, key=lambda x: x.get('rank', 999))
    top_k_results = sorted_results[:k]
    
    docs = {}
    for r in top_k_results:
        doc_key = r.get('doc_key')
        score = r.get('score', 0)
        rank = r.get('rank', 999)
        if doc_key:
            docs[doc_key] = {
                'score': score,
                'rank': rank,
                # V3: status/valid_from/valid_to avec fallback vers anciens noms
                'status': r.get('status') or r.get('etat', 'N/A'),
                'valid_from': r.get('valid_from') or r.get('date_debut', 'N/A'),
                'valid_to': r.get('valid_to') or r.get('date_fin', 'N/A'),
            }
    return docs

def find_consensus_docs(bm25_docs, dense_docs, hybrid_docs, min_retrievers=2):
    """
    Trouve documents qui apparaissent dans plusieurs retrievers.
    
    Args:
        bm25_docs: {doc_key: info}
        dense_docs: {doc_key: info}
        hybrid_docs: {doc_key: info}
        min_retrievers: Minimum de retrievers où le doc doit apparaître
        
    Returns:
        dict: {doc_key: {'retrievers': [...], 'scores': {...}, 'info': {...}}}
    """
    all_docs = set(bm25_docs.keys()) | set(dense_docs.keys()) | set(hybrid_docs.keys())
    
    consensus = {}
    for doc_key in all_docs:
        retrievers = []
        scores = {}
        
        if doc_key in bm25_docs:
            retrievers.append('bm25')
            scores['bm25'] = bm25_docs[doc_key]['score']
        
        if doc_key in dense_docs:
            retrievers.append('dense')
            scores['dense'] = dense_docs[doc_key]['score']
        
        if doc_key in hybrid_docs:
            retrievers.append('hybrid')
            scores['hybrid'] = hybrid_docs[doc_key]['score']
        
        # Garde uniquement si présent dans assez de retrievers
        if len(retrievers) >= min_retrievers:
            # Prendre info du premier retriever disponible
            info = bm25_docs.get(doc_key) or dense_docs.get(doc_key) or hybrid_docs.get(doc_key)
            
            consensus[doc_key] = {
                'retrievers': retrievers,
                'num_retrievers': len(retrievers),
                'scores': scores,
                'avg_rank': sum([
                    bm25_docs.get(doc_key, {}).get('rank', 999),
                    dense_docs.get(doc_key, {}).get('rank', 999),
                    hybrid_docs.get(doc_key, {}).get('rank', 999)
                ]) / 3,
                # V3: nouveaux noms
                'status': info.get('status', 'N/A'),
                'valid_from': info.get('valid_from', 'N/A'),
                'valid_to': info.get('valid_to', 'N/A'),
            }
    
    return consensus

def filter_abroge(consensus_docs, exclude_abroge=True):
    """Filtre les articles abrogés si demandé."""
    if not exclude_abroge:
        return consensus_docs

    filtered = {}
    for doc_key, info in consensus_docs.items():
        # V3: status au lieu de etat
        status = info.get('status', '').upper()
        if status != 'ABROGE':
            filtered[doc_key] = info

    return filtered

def load_corpus(corpus_path):
    """Charge corpus pour enrichir avec texte."""
    print(f"📦 Chargement corpus: {corpus_path}")
    table = pq.read_table(corpus_path)
    df = table.to_pandas()
    print(f"✅ {len(df):,} chunks chargés")
    return df.set_index('doc_key')

def find_latest_runs(runs_dir, retriever_type):
    """
    Trouve le run le plus récent pour un retriever.
    Ne considère que des dossiers (évite de matcher metrics_*.json).
    """
    runs_dir = Path(runs_dir)
    pattern = f"*{retriever_type}*"

    runs = [p for p in runs_dir.glob(pattern) if p.is_dir()]
    if not runs:
        return None

    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return runs[0]


def is_consolidated_run_dir(p: Path) -> bool:
    """
    Détecte un dossier de run qui contient directement les fichiers *_results.jsonl.
    Exemple: bm25_results.jsonl, dense_results.jsonl, hybrid_results.jsonl.
    """
    return (p / "bm25_results.jsonl").exists() or (p / "dense_results.jsonl").exists() or (p / "hybrid_results.jsonl").exists()

def main():
    parser = argparse.ArgumentParser(description="Créer qrels v3 par consensus")
    parser.add_argument("--runs-dir", default="runs", help="Dossier des runs")
    parser.add_argument("--corpus", required=True, help="Corpus Parquet")
    parser.add_argument("--output", default="qrels_v3_candidates.yaml", help="Fichier YAML de sortie")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k à considérer par retriever")
    parser.add_argument("--min-retrievers", type=int, default=2, 
                       help="Minimum de retrievers où doc doit apparaître (1=union, 2=consensus, 3=unanimité)")
    parser.add_argument("--exclude-abroge", action="store_true", 
                       help="Exclure articles abrogés")
    parser.add_argument("--max-docs-per-query", type=int, default=5,
                       help="Maximum de documents par query dans qrels candidats")
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)

    # Si --runs-dir pointe directement vers un run consolidé (3 JSONL dans le même dossier),
    # on utilise ce même dossier pour bm25/dense/hybrid.
    if is_consolidated_run_dir(runs_dir):
        print("✅ Dossier de run consolidé détecté (bm25/dense/hybrid dans le même dossier).")
        bm25_run = runs_dir
        dense_run = runs_dir
        hybrid_run = runs_dir
    else:
        # Mode normal: chercher les derniers runs par type dans un répertoire contenant plusieurs runs
        print("🔍 Recherche des runs les plus récents...")
        bm25_run = find_latest_runs(runs_dir, "bm25")
        dense_run = find_latest_runs(runs_dir, "dense")
        hybrid_run = find_latest_runs(runs_dir, "hybrid")

    
    print("=" * 80)
    print("🎯 CRÉATION QRELS V3 PAR CONSENSUS")
    print("=" * 80)
    print()
    
    # Charger corpus
    corpus = load_corpus(args.corpus)
       
    print(f"   BM25  : {bm25_run.name if bm25_run else 'Non trouvé'}")
    print(f"   Dense : {dense_run.name if dense_run else 'Non trouvé'}")
    print(f"   Hybrid: {hybrid_run.name if hybrid_run else 'Non trouvé'}")
    print()
    
    if not (bm25_run or dense_run or hybrid_run):
        print("❌ Aucun run trouvé")
        return
    
    # Charger résultats
    print("📋 Chargement des résultats...")
    bm25_results = load_results_jsonl(bm25_run / "bm25_results.jsonl") if bm25_run else []
    dense_results = load_results_jsonl(dense_run / "dense_results.jsonl") if dense_run else []
    hybrid_results = load_results_jsonl(hybrid_run / "hybrid_results.jsonl") if hybrid_run else []
    
    print(f"   BM25  : {len(bm25_results)} résultats")
    print(f"   Dense : {len(dense_results)} résultats")
    print(f"   Hybrid: {len(hybrid_results)} résultats")
    print()
    
    # Grouper par query
    bm25_by_query = group_by_query(bm25_results)
    dense_by_query = group_by_query(dense_results)
    hybrid_by_query = group_by_query(hybrid_results)
    
    # Trouver queries communes
    all_queries = set(bm25_by_query.keys()) | set(dense_by_query.keys()) | set(hybrid_by_query.keys())
    print(f"✅ {len(all_queries)} query(ies) unique(s) trouvée(s)")
    print()
    
    # Analyser consensus pour chaque query
    print("=" * 80)
    print("🔍 ANALYSE CONSENSUS PAR QUERY")
    print("=" * 80)
    print()
    
    qrels_candidates = {}
    stats = {
        'total_queries': len(all_queries),
        'total_candidates': 0,
        'by_num_retrievers': {1: 0, 2: 0, 3: 0},
        'abroge_filtered': 0,
    }
    
    for qid in sorted(all_queries):
        print(f"Query: {qid}")
        
        # Extraire top-k de chaque retriever
        bm25_docs = extract_top_k(bm25_by_query.get(qid, []), k=args.top_k)
        dense_docs = extract_top_k(dense_by_query.get(qid, []), k=args.top_k)
        hybrid_docs = extract_top_k(hybrid_by_query.get(qid, []), k=args.top_k)
        
        print(f"   Top-{args.top_k}: BM25={len(bm25_docs)}, Dense={len(dense_docs)}, Hybrid={len(hybrid_docs)}")
        
        # Trouver consensus
        consensus = find_consensus_docs(bm25_docs, dense_docs, hybrid_docs, 
                                       min_retrievers=args.min_retrievers)
        
        print(f"   Consensus (min {args.min_retrievers} retrievers): {len(consensus)} documents")
        
        # Filtrer abrogés
        before_filter = len(consensus)
        if args.exclude_abroge:
            consensus = filter_abroge(consensus)
            filtered_count = before_filter - len(consensus)
            if filtered_count > 0:
                print(f"   ⚠️ {filtered_count} article(s) abrogé(s) filtré(s)")
                stats['abroge_filtered'] += filtered_count
        
        # Trier par nombre de retrievers puis par avg_rank
        sorted_consensus = sorted(
            consensus.items(),
            key=lambda x: (-x[1]['num_retrievers'], x[1]['avg_rank'])
        )
        
        # Garder top-N
        top_consensus = sorted_consensus[:args.max_docs_per_query]
        
        # Créer qrels pour cette query
        if top_consensus:
            qrels_candidates[qid] = {}
            
            for doc_key, info in top_consensus:
                retrievers_str = "+".join(info['retrievers'])
                print(f"      ✓ {doc_key[:20]}... [{retrievers_str}] rank_avg={info['avg_rank']:.1f}")
                
                # Enrichir avec texte du corpus
                text_preview = "N/A"
                if doc_key in corpus.index:
                    text = corpus.loc[doc_key].get('text', '')
                    text_preview = text[:200] if text else "N/A"
                
                qrels_candidates[qid][doc_key] = {
                    'relevance': 1,  # Par défaut, à valider manuellement
                    'retrievers': info['retrievers'],
                    'num_retrievers': info['num_retrievers'],
                    'scores': info['scores'],
                    'avg_rank': float(info['avg_rank']),
                    # V3: nouveaux noms
                    'status': info.get('status', 'N/A'),
                    'valid_from': info.get('valid_from', 'N/A'),
                    'valid_to': info.get('valid_to', 'N/A'),
                    'text_preview': text_preview,
                    'validation_status': 'pending',  # pending, validated, rejected
                }
                
                stats['by_num_retrievers'][info['num_retrievers']] += 1
                stats['total_candidates'] += 1
        
        print()
    
    # Statistiques finales
    print("=" * 80)
    print("📊 STATISTIQUES")
    print("=" * 80)
    print(f"   Queries analysées       : {stats['total_queries']}")
    print(f"   Candidats total         : {stats['total_candidates']}")

    if stats["total_queries"] > 0:
        print(f"   Candidats par query     : {stats['total_candidates'] / stats['total_queries']:.1f}")
    else:
        print("   Candidats par query     : N/A (0 query)")

    print()
    print("   Par nombre de retrievers:")
    print(f"      3 retrievers (unanimité): {stats['by_num_retrievers'][3]}")
    print(f"      2 retrievers (consensus): {stats['by_num_retrievers'][2]}")
    print(f"      1 retriever  (unique)   : {stats['by_num_retrievers'][1]}")
    print()
    if args.exclude_abroge:
        print(f"   Articles abrogés filtrés: {stats['abroge_filtered']}")
        print()
    
    # Sauver YAML
    output_file = Path(args.output)
    print(f"💾 Sauvegarde dans {output_file}")
    
    with open(output_file, 'w') as f:
        yaml.dump(qrels_candidates, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print("✅ Fichier créé avec succès")
    print()
    print("=" * 80)
    print("🎯 PROCHAINES ÉTAPES")
    print("=" * 80)
    print(f"1. Valider manuellement les candidats avec:")
    print(f"   python tools/validate_qrels_manual.py --qrels {output_file}")
    print()
    print(f"2. Une fois validé, renommer en qrels_cdtravail_v3.yaml")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
