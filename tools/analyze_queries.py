#!/usr/bin/env python3
"""
Analyse et amélioration des queries de test
Phase 4 - Queries de qualité

Usage:
    python tools/analyze_queries.py --qrels configs/qrels_cdtravail_v2.yaml --corpus <path>
    
Analyse:
- Couverture des queries (combien de docs pertinents par query)
- Difficulté des queries (facile, moyen, difficile)
- Suggestions de nouvelles queries
- Variantes de queries existantes
"""

import sys
import json
import yaml
from pathlib import Path
import argparse
import pyarrow.parquet as pq
import pandas as pd
from collections import Counter

def load_qrels(qrels_file):
    """Charge qrels YAML."""
    with open(qrels_file) as f:
        return yaml.safe_load(f)

def load_corpus(corpus_path):
    """Charge corpus."""
    print(f"📦 Chargement corpus: {corpus_path}")
    table = pq.read_table(corpus_path)
    df = table.to_pandas()
    print(f"✅ {len(df):,} chunks chargés\n")
    return df.set_index('doc_key')

def analyze_query_coverage(qrels, corpus):
    """Analyse la couverture de chaque query."""
    coverage = {}
    
    for qid, docs in qrels.items():
        relevant_docs = list(docs.keys())
        
        # Compter docs dans corpus
        docs_in_corpus = sum(1 for doc in relevant_docs if doc in corpus.index)
        
        # Analyser statuts (V3: status, fallback: etat)
        statuts = []
        for doc in relevant_docs:
            if doc in corpus.index:
                row = corpus.loc[doc]
                statut = row.get('status') or row.get('etat') or 'N/A'
                statuts.append(statut)

        statut_counts = Counter(statuts)
        
        coverage[qid] = {
            'num_relevant': len(relevant_docs),
            'in_corpus': docs_in_corpus,
            'missing': len(relevant_docs) - docs_in_corpus,
            'statuts': dict(statut_counts),
            'has_abroge': 'ABROGE' in statut_counts,
        }
    
    return coverage

def classify_query_difficulty(num_relevant):
    """Classifie difficulté d'une query selon nombre de docs pertinents."""
    if num_relevant <= 2:
        return "🔴 Difficile (≤2 docs)"
    elif num_relevant <= 5:
        return "🟡 Moyen (3-5 docs)"
    else:
        return "🟢 Facile (>5 docs)"

def suggest_query_variants(query_text):
    """Suggère des variantes d'une query."""
    variants = []
    
    # Variante courte (mots-clés)
    if len(query_text.split()) > 5:
        keywords = extract_keywords(query_text)
        variants.append({
            'type': 'keywords',
            'text': ' '.join(keywords[:3]),
            'description': 'Version mots-clés uniquement'
        })
    
    # Variante formelle (article + code)
    if 'article' in query_text.lower() and 'code' in query_text.lower():
        # Déjà formelle
        variants.append({
            'type': 'informal',
            'text': query_text.replace('article', '').replace('Code', 'code'),
            'description': 'Version moins formelle'
        })
    
    # Variante avec synonymes
    synonyms = {
        'prévoit': 'dit',
        'stipule': 'indique',
        'dispose': 'précise',
    }
    variant_text = query_text
    for orig, syn in synonyms.items():
        if orig in variant_text:
            variant_text = variant_text.replace(orig, syn)
            break
    
    if variant_text != query_text:
        variants.append({
            'type': 'synonym',
            'text': variant_text,
            'description': 'Version avec synonyme'
        })
    
    return variants

def extract_keywords(text):
    """Extrait mots-clés d'un texte (simple)."""
    stopwords = {'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 
                 'que', 'qui', 'quoi', 'quel', 'quelle',
                 'est', 'et', 'ou', 'à', 'au', 'aux'}
    
    words = text.lower().split()
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    return keywords

def suggest_new_queries(corpus):
    """Suggère de nouvelles queries basées sur le corpus."""
    suggestions = []
    
    # Thèmes communs dans Code du Travail
    themes = {
        'contrat_travail': ['contrat', 'CDI', 'CDD', 'période essai'],
        'salaire': ['salaire', 'rémunération', 'minimum', 'heures supplémentaires'],
        'congés': ['congés payés', 'RTT', 'repos'],
        'licenciement': ['licenciement', 'rupture', 'indemnité'],
        'durée_travail': ['durée travail', '35 heures', 'temps partiel'],
        'sécurité': ['sécurité', 'accident travail', 'maladie professionnelle'],
    }
    
    for theme, keywords in themes.items():
        suggestions.append({
            'theme': theme,
            'keywords': keywords,
            'example_query': f"Que dit le Code du Travail sur {keywords[0]} ?",
        })
    
    return suggestions

def main():
    parser = argparse.ArgumentParser(description="Analyser queries")
    parser.add_argument("--qrels", required=True, help="Fichier qrels (YAML)")
    parser.add_argument("--corpus", required=True, help="Corpus Parquet")
    parser.add_argument("--suggest-new", action="store_true", help="Suggérer nouvelles queries")
    parser.add_argument("--suggest-variants", action="store_true", help="Suggérer variantes")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 ANALYSE DES QUERIES")
    print("=" * 80)
    print()
    
    # Charger données
    qrels = load_qrels(args.qrels)
    corpus = load_corpus(args.corpus)
    
    # Analyser couverture
    print("=" * 80)
    print("📊 COUVERTURE PAR QUERY")
    print("=" * 80)
    print()
    
    coverage = analyze_query_coverage(qrels, corpus)
    
    for qid, cov in coverage.items():
        difficulty = classify_query_difficulty(cov['num_relevant'])
        
        print(f"Query: {qid}")
        print(f"   Difficulté     : {difficulty}")
        print(f"   Docs pertinents: {cov['num_relevant']}")
        print(f"   Dans corpus    : {cov['in_corpus']}")
        if cov['missing'] > 0:
            print(f"   ⚠️ Manquants    : {cov['missing']}")
        
        # Statuts
        if cov['statuts']:
            print(f"   Statuts        : {', '.join(f'{k}={v}' for k, v in cov['statuts'].items())}")
            if cov['has_abroge']:
                print(f"      ⚠️ Contient articles abrogés")
        print()
    
    # Statistiques globales
    print("=" * 80)
    print("📊 STATISTIQUES GLOBALES")
    print("=" * 80)
    print()
    
    total_queries = len(qrels)
    total_relevant = sum(len(docs) for docs in qrels.values())
    avg_relevant = total_relevant / total_queries if total_queries > 0 else 0
    
    difficulty_counts = Counter([
        classify_query_difficulty(cov['num_relevant'])
        for cov in coverage.values()
    ])
    
    print(f"   Total queries       : {total_queries}")
    print(f"   Total docs pertinents: {total_relevant}")
    print(f"   Moyenne par query   : {avg_relevant:.1f}")
    print()
    print("   Distribution difficulté:")
    for diff, count in difficulty_counts.most_common():
        print(f"      {diff}: {count} queries")
    print()
    
    # Recommandations
    print("=" * 80)
    print("💡 RECOMMANDATIONS")
    print("=" * 80)
    print()
    
    # Queries avec peu de docs
    low_coverage = [qid for qid, cov in coverage.items() if cov['num_relevant'] <= 2]
    if low_coverage:
        print(f"⚠️ {len(low_coverage)} query(ies) avec ≤2 docs pertinents:")
        for qid in low_coverage:
            print(f"   - {qid}: {coverage[qid]['num_relevant']} doc(s)")
        print(f"   → Ajouter plus de docs pertinents ou créer variantes")
        print()
    
    # Queries avec articles abrogés
    with_abroge = [qid for qid, cov in coverage.items() if cov['has_abroge']]
    if with_abroge:
        print(f"⚠️ {len(with_abroge)} query(ies) avec articles abrogés:")
        for qid in with_abroge:
            print(f"   - {qid}")
        print(f"   → Remplacer par articles en vigueur")
        print()
    
    # Équilibre de difficulté
    easy = difficulty_counts.get("🟢 Facile (>5 docs)", 0)
    medium = difficulty_counts.get("🟡 Moyen (3-5 docs)", 0)
    hard = difficulty_counts.get("🔴 Difficile (≤2 docs)", 0)
    
    if hard > total_queries * 0.5:
        print("⚠️ Trop de queries difficiles (>50%)")
        print("   → Ajouter queries plus faciles pour équilibrer")
        print()
    elif easy > total_queries * 0.7:
        print("⚠️ Trop de queries faciles (>70%)")
        print("   → Ajouter queries plus difficiles pour challenger")
        print()
    else:
        print("✅ Bon équilibre de difficulté")
        print()
    
    # Suggérer nouvelles queries
    if args.suggest_new:
        print("=" * 80)
        print("💡 SUGGESTIONS DE NOUVELLES QUERIES")
        print("=" * 80)
        print()
        
        suggestions = suggest_new_queries(corpus)
        
        for sug in suggestions:
            print(f"Thème: {sug['theme']}")
            print(f"   Mots-clés: {', '.join(sug['keywords'])}")
            print(f"   Exemple  : {sug['example_query']}")
            print()
    
    # Suggérer variantes
    if args.suggest_variants:
        print("=" * 80)
        print("💡 SUGGESTIONS DE VARIANTES")
        print("=" * 80)
        print()
        
        # Pour l'instant on ne peut pas récupérer les query_text depuis qrels
        # On devrait les stocker dans un fichier séparé ou dans les qrels
        print("⚠️ Fonctionnalité nécessite query_text stocké dans qrels ou fichier séparé")
        print()
    
    print("=" * 80)
    print("🎯 PROCHAINES ÉTAPES")
    print("=" * 80)
    print()
    print("1. Enrichir queries avec peu de docs pertinents")
    print("2. Remplacer articles abrogés")
    print("3. Créer variantes de queries existantes")
    print("4. Ajouter nouvelles queries sur thèmes manquants")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
