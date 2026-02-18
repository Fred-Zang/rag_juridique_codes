#!/usr/bin/env python3
"""
Validation manuelle des qrels candidats
Phase 4 - Qrels de qualité

Usage:
    python tools/validate_qrels_manual.py --qrels qrels_v3_candidates.yaml --corpus <path>
    
Interface interactive pour valider/rejeter chaque candidat qrel.
"""

import sys
import json
import yaml
from pathlib import Path
import argparse
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime
import os
import subprocess
import urllib.parse
import shutil


class Colors:
    """Couleurs ANSI pour terminal."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_qrels(qrels_file):
    """Charge qrels YAML."""
    with open(qrels_file) as f:
        return yaml.safe_load(f)

def save_qrels(qrels, output_file):
    """Sauve qrels YAML."""
    with open(output_file, 'w') as f:
        yaml.dump(qrels, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

def load_corpus(corpus_path):
    """Charge corpus pour enrichir avec texte."""
    print(f"📦 Chargement corpus: {corpus_path}")
    table = pq.read_table(corpus_path)
    df = table.to_pandas()
    # Normaliser les colonnes du corpus pour exposer status/valid_from/valid_to
    df = normalize_corpus_df(df)
    print(f"✅ {len(df):,} chunks chargés\n")
    return df.set_index('doc_key')

def file_path_to_url(p: str) -> str:
    """
    Convertit un chemin local en URL file://… correctement échappée.
    Exemple: /home/user/a b.xml -> file:///home/user/a%20b.xml
    """
    if not p:
        return ""
    try:
        return Path(p).resolve().as_uri()
    except Exception:
        # Fallback si as_uri() échoue
        pp = str(Path(p).resolve())
        return "file://" + urllib.parse.quote(pp)


def terminal_hyperlink(label: str, url: str) -> str:
    """
    Crée un lien cliquable OSC-8 pour terminaux compatibles (VS Code terminal, iTerm, etc.).
    Si non supporté, le texte restera lisible (label + URL).
    """
    if not url:
        return label
    esc = "\033"
    return f"{esc}]8;;{url}{esc}\\{label}{esc}]8;;{esc}\\ ({url})"


def open_in_chromium(url: str):
    """
    Ouvre une URL file:// dans Chromium si possible, sinon fallback xdg-open.
    """
    if not url:
        return

    # Candidats Chromium selon distros
    candidates = ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable"]

    for exe in candidates:
        if shutil.which(exe):
            subprocess.Popen([exe, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return

    # Fallback: ouvre dans le navigateur par défaut
    if shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    print("⚠️ Impossible d’ouvrir automatiquement: Chromium/xdg-open introuvable.")


def normalize_doc_info(d: dict) -> dict:
    """
    Normalise les clés de métadonnées d'un doc qrels pour n'utiliser
    qu'un vocabulaire canonique dans le script.

    Canonique:
      - status (au lieu de etat)
      - valid_from / valid_to (au lieu de date_debut / date_fin)
    """
    if not isinstance(d, dict):
        return d

    out = dict(d)

    # Unifier statut
    if "status" not in out and "etat" in out:
        out["status"] = out.get("etat")

    # Unifier dates
    if "valid_from" not in out and "date_debut" in out:
        out["valid_from"] = out.get("date_debut")
    if "valid_to" not in out and "date_fin" in out:
        out["valid_to"] = out.get("date_fin")

    return out


def normalize_corpus_df(df):
    """
    Normalise les colonnes du corpus pour exposer:
      - status
      - valid_from
      - valid_to
    sans casser le reste.
    """
    if "status" not in df.columns:
        if "etat" in df.columns:
            df["status"] = df["etat"]
        elif "meta" in df.columns:
            # si meta est une colonne dict
            def _meta_status(v):
                return v.get("status") or v.get("etat") if isinstance(v, dict) else None
            df["status"] = df["meta"].apply(_meta_status)

    if "valid_from" not in df.columns and "date_debut" in df.columns:
        df["valid_from"] = df["date_debut"]
    if "valid_to" not in df.columns and "date_fin" in df.columns:
        df["valid_to"] = df["date_fin"]

    return df


def print_separator(char="=", length=80):
    """Affiche séparateur."""
    print(char * length)

def print_doc_info(doc_key, doc_info, corpus, query_text=None):
    """Affiche informations détaillées du document."""
    # URL XML utilisée par l'action [o] (toujours définie)
    xml_url = ""

    print_separator("─")
    print(f"{Colors.BOLD}Document: {doc_key}{Colors.ENDC}")
    print_separator("─")

    # Info query si disponible
    if query_text:
        print(f"\n{Colors.OKCYAN}Query:{Colors.ENDC} {query_text}\n")

    # Métadonnées
    print(f"{Colors.BOLD}Métadonnées:{Colors.ENDC}")
    print(f"   Statut        : {doc_info.get('status', 'N/A')}")
    print(f"   Date début    : {doc_info.get('valid_from', 'N/A')}")
    print(f"   Date fin      : {doc_info.get('valid_to', 'N/A')}")
    print(f"   Retrievers    : {', '.join(doc_info.get('retrievers', []))}")

    avg_rank = doc_info.get("avg_rank")
    try:
        avg_rank_str = f"{float(avg_rank):.1f}"
    except Exception:
        avg_rank_str = "N/A"
    print(f"   Rank moyen    : {avg_rank_str}")

    # Scores par retriever
    scores = doc_info.get('scores', {})
    if scores:
        print(f"\n{Colors.BOLD}Scores:{Colors.ENDC}")
        for retriever, score in scores.items():
            print(f"   {retriever:8s}: {score:.4f}")

    # Texte complet du corpus
    print(f"\n{Colors.BOLD}Texte:{Colors.ENDC}")
    rows = find_corpus_rows_by_doc_key(corpus, doc_key)

    if len(rows) == 0:
        print(f"{Colors.FAIL}Document non trouvé dans le corpus (doc_key mismatch article vs chunk).{Colors.ENDC}")
        return xml_url

    if len(rows) > 1:
        print(f"{Colors.WARNING}[{len(rows)} chunks trouvés pour cet article — affichage du 1er chunk non vide]{Colors.ENDC}")

    # Choisir le 1er chunk qui contient réellement du texte (en testant plusieurs champs)
    row0 = None
    for _, r in rows.iterrows():
        t = (r.get("text") or r.get("chunk_text") or r.get("content") or r.get("raw_text") or "")
        if isinstance(t, str) and t.strip():
            row0 = r
            break
    if row0 is None:
        row0 = rows.iloc[0]

    # Métadonnées utiles si présentes
    meta = row0.get("meta", {})
    source_path = None
    if isinstance(meta, dict):
        corpus_juridique = meta.get("corpus_juridique")
        article_num = meta.get("article_num")
        source_path = meta.get("source_path") or row0.get("source_path")

        if corpus_juridique:
            print(f"{Colors.OKGREEN}corpus_juridique: {corpus_juridique}{Colors.ENDC}")
        if article_num:
            print(f"{Colors.OKGREEN}article_num: {article_num}{Colors.ENDC}")
    else:
        source_path = row0.get("source_path")

    # Lien cliquable vers le XML (si disponible)
    if source_path:
        xml_url = file_path_to_url(source_path)
        print(terminal_hyperlink("Ouvrir le XML (clic)", xml_url))
        print(f"{Colors.OKGREEN}source_path: {source_path}{Colors.ENDC}")

    # Texte
    text = row0.get("text") or row0.get("chunk_text") or row0.get("content") or row0.get("raw_text") or ""
    if not isinstance(text, str) or not text.strip():
        print("Texte: N/A")
        return xml_url

    snippet = text.strip()
    if len(snippet) > 1200:
        print("\nTexte (extrait):\n")
        print(snippet[:1200] + "...")
        print(f"\n{Colors.WARNING}[Texte tronqué - {len(snippet)} caractères total]{Colors.ENDC}")
    else:
        print("\nTexte:\n")
        print(snippet)

    print()
    return xml_url


def validate_document(doc_key, doc_info, corpus, query_text=None, auto_mode=False):
    """
    Valide un document interactivement.
    
    Returns:
        str: 'validated', 'rejected', 'skip', 'quit', 'auto_yes', 'auto_no'
    """
    # Afficher info
    xml_url = print_doc_info(doc_key, doc_info, corpus, query_text)
    
    if auto_mode == 'yes':
        print(f"{Colors.OKGREEN}[Mode auto: VALIDÉ]{Colors.ENDC}\n")
        return 'validated'
    elif auto_mode == 'no':
        print(f"{Colors.FAIL}[Mode auto: REJETÉ]{Colors.ENDC}\n")
        return 'rejected'
    
    # Demander validation
    print(f"{Colors.BOLD}Action:{Colors.ENDC}")
    print("  [y] Valider (pertinent)")
    print("  [n] Rejeter (non pertinent)")
    print("  [s] Skip (décider plus tard)")
    print("  [v] Voir texte complet")
    print("  [Y] Valider tout le reste (mode auto)")
    print("  [N] Rejeter tout le reste (mode auto)")
    print("  [q] Quitter")
    print("  [o] Ouvrir le XML dans Chromium")

    
    while True:
        choice = input(f"\n{Colors.OKCYAN}Choix:{Colors.ENDC} ").strip()
        
        if choice == 'y':
            return 'validated'
        elif choice == 'n':
            return 'rejected'
        elif choice == 's':
            return 'skip'
        elif choice == 'v':
            rows = find_corpus_rows_by_doc_key(corpus, doc_key)
            if len(rows) == 0:
                print(f"{Colors.FAIL}Document non trouvé{Colors.ENDC}\n")
                continue

            # On prend le premier chunk non vide
            row0 = None
            for _, r in rows.iterrows():
                t = (r.get("text") or r.get("chunk_text") or r.get("content") or r.get("raw_text") or "")
                if isinstance(t, str) and t.strip():
                    row0 = r
                    break
            if row0 is None:
                row0 = rows.iloc[0]

            full_text = row0.get("text") or row0.get("chunk_text") or row0.get("content") or row0.get("raw_text") or "N/A"
            print(f"\n{Colors.BOLD}TEXTE COMPLET (chunk sélectionné):{Colors.ENDC}\n")
            print(full_text)
            print()
            continue

        elif choice.upper() == 'Y':
            return 'auto_yes'
        elif choice.upper() == 'N':
            return 'auto_no'
        elif choice == 'q':
            return 'quit'
        elif choice.lower() == "o":
            if xml_url:
                open_in_chromium(xml_url)
            else:
                print("⚠️ Pas de source_path/XML à ouvrir pour ce document.")
            # On ne quitte pas la validation : on laisse l'utilisateur décider ensuite
            continue
        else:
            print(f"{Colors.FAIL}Choix invalide. Utilisez y/n/s/v/Y/N/q{Colors.ENDC}")

def extract_article_id(doc_key: str) -> str:
    """
    Extrait un identifiant d'article LEGIARTI... depuis une clé composite éventuelle.
    Exemples:
      - "LEGIARTI0000..." -> "LEGIARTI0000..."
      - "LEGIARTI0000...|vf=...|chunk=2" -> "LEGIARTI0000..."
    """
    if not doc_key:
        return ""
    if doc_key.startswith("LEGIARTI"):
        return doc_key.split("|", 1)[0]
    return ""


def find_corpus_rows_by_doc_key(corpus_df, doc_key: str):
    """
    Retourne un DataFrame (0..N lignes) correspondant à doc_key.
    1) Match exact sur l'index.
    2) Sinon, si doc_key est un LEGIARTI..., match par préfixe (article_id).
       Utile quand le corpus est au niveau chunk (doc_key composite).
    """
    if doc_key in corpus_df.index:
        return corpus_df.loc[[doc_key]]

    article_id = extract_article_id(doc_key)
    if not article_id:
        return corpus_df.iloc[0:0]

    # index -> strings
    idx = corpus_df.index.astype(str)
    mask = idx.str.startswith(article_id)
    return corpus_df[mask]

# -----------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Valider qrels manuellement")
    parser.add_argument("--qrels", required=True, help="Fichier qrels candidats (YAML)")
    parser.add_argument("--corpus", required=True, help="Corpus Parquet")
    parser.add_argument("--output", help="Fichier de sortie (défaut: qrels_v3_validated.yaml)")
    parser.add_argument("--start-from", help="Commencer à partir d'une query spécifique")
    parser.add_argument("--queries", help="Fichier queries YAML (pour afficher le texte des queries)")
    args = parser.parse_args()
    
    if not args.output:
        # Générer nom de sortie automatiquement
        input_stem = Path(args.qrels).stem
        args.output = f"{input_stem}_validated.yaml"
    
    print_separator()
    print(f"{Colors.HEADER}{Colors.BOLD}🎯 VALIDATION MANUELLE QRELS{Colors.ENDC}")
    print_separator()
    print()
    
    # Charger données
    qrels = load_qrels(args.qrels)
    corpus = load_corpus(args.corpus)

    # Charge le YAML queries
    queries_map = {}
    if args.queries:
        with open(args.queries, "r", encoding="utf-8") as f:
            q = yaml.safe_load(f) or {}
        for item in q.get("queries", []):
            qid = item.get("id")
            txt = item.get("text")
            if qid and txt:
                queries_map[qid] = txt


    # Normalisation des docs qrels: etat->status, date_debut/date_fin->valid_from/valid_to
    for qid, docs in qrels.items():
        qrels[qid] = {k: normalize_doc_info(v) for k, v in docs.items()}

    print(f"📋 Qrels chargés: {len(qrels)} queries")
    total_docs = sum(len(docs) for docs in qrels.values())
    print(f"📄 Documents à valider: {total_docs}")
    print()
    
    # Statistiques
    stats = {
        'validated': 0,
        'rejected': 0,
        'skipped': 0,
        'total': total_docs,
    }
    
    # Mode auto
    auto_mode = None  # None, 'yes', 'no'
    
    # Commencer à partir d'une query spécifique
    start_processing = args.start_from is None
    
    # Valider chaque document
    try:
        for query_idx, (qid, docs) in enumerate(qrels.items(), 1):
            # Skip jusqu'à la query demandée
            if not start_processing:
                if qid == args.start_from:
                    start_processing = True
                else:
                    continue
            
            # Header query
            print_separator("=")
            print(f"{Colors.HEADER}{Colors.BOLD}Query {query_idx}/{len(qrels)}: {qid}{Colors.ENDC}")
            print_separator("=")
            print()
            
            # Extraire texte de query depuis premier doc
            query_text = queries_map.get(qid)
            
            for doc_idx, (doc_key, doc_info) in enumerate(docs.items(), 1):
                print(f"\n{Colors.BOLD}Document {doc_idx}/{len(docs)}{Colors.ENDC}\n")
                
                # Valider
                result = validate_document(doc_key, doc_info, corpus, query_text, auto_mode)
                
                if result == 'validated':
                    doc_info['validation_status'] = 'validated'
                    doc_info['relevance'] = 1
                    stats['validated'] += 1
                    print(f"{Colors.OKGREEN}✓ Validé{Colors.ENDC}\n")
                
                elif result == 'rejected':
                    doc_info['validation_status'] = 'rejected'
                    doc_info['relevance'] = 0
                    stats['rejected'] += 1
                    print(f"{Colors.FAIL}✗ Rejeté{Colors.ENDC}\n")
                
                elif result == 'skip':
                    doc_info['validation_status'] = 'pending'
                    stats['skipped'] += 1
                    print(f"{Colors.WARNING}⊘ Skipped{Colors.ENDC}\n")
                
                elif result == 'auto_yes':
                    auto_mode = 'yes'
                    doc_info['validation_status'] = 'validated'
                    doc_info['relevance'] = 1
                    stats['validated'] += 1
                    print(f"{Colors.OKGREEN}✓ Validé (mode auto activé){Colors.ENDC}\n")
                
                elif result == 'auto_no':
                    auto_mode = 'no'
                    doc_info['validation_status'] = 'rejected'
                    doc_info['relevance'] = 0
                    stats['rejected'] += 1
                    print(f"{Colors.FAIL}✗ Rejeté (mode auto activé){Colors.ENDC}\n")
                
                elif result == 'quit':
                    print(f"\n{Colors.WARNING}Validation interrompue par l'utilisateur{Colors.ENDC}")
                    raise KeyboardInterrupt
            
            # Stats intermédiaires
            print()
            print(f"{Colors.BOLD}Stats après query {qid}:{Colors.ENDC}")
            print(f"   Validés  : {stats['validated']}/{stats['total']}")
            print(f"   Rejetés  : {stats['rejected']}/{stats['total']}")
            print(f"   Skipped  : {stats['skipped']}/{stats['total']}")
            print()
    
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Validation interrompue.{Colors.ENDC}")
    
    # Sauvegarder résultats
    print()
    print_separator()
    print(f"{Colors.BOLD}💾 SAUVEGARDE{Colors.ENDC}")
    print_separator()
    print()
    
    output_path = Path(args.output)
    save_qrels(qrels, output_path)
    print(f"✅ Qrels sauvegardés dans: {output_path}")
    print()
    
    # Statistiques finales
    print_separator()
    print(f"{Colors.BOLD}📊 STATISTIQUES FINALES{Colors.ENDC}")
    print_separator()
    print(f"   Total documents : {stats['total']}")
    print(f"   Validés         : {stats['validated']} ({100*stats['validated']/stats['total']:.1f}%)")
    print(f"   Rejetés         : {stats['rejected']} ({100*stats['rejected']/stats['total']:.1f}%)")
    print(f"   Skipped         : {stats['skipped']} ({100*stats['skipped']/stats['total']:.1f}%)")
    print()
    
    # Créer version finale (uniquement validés)
    if stats['validated'] > 0:
        final_qrels = {}
        for qid, docs in qrels.items():
            validated_docs = {
                doc_key: {
                    'relevance': doc_info.get('relevance', 1),
                    'status': doc_info.get('status'),
                    'retrievers': doc_info.get('retrievers'),
                }
                for doc_key, doc_info in docs.items()
                if doc_info.get('validation_status') == 'validated'
            }
            if validated_docs:
                final_qrels[qid] = validated_docs
        
        # Sauver version finale
        final_output = output_path.parent / f"qrels_cdtravail_v3_final.yaml"
        save_qrels(final_qrels, final_output)
        print(f"✅ Qrels finaux (validés uniquement) dans: {final_output}")
        print()
    
    print_separator()
    print(f"{Colors.HEADER}🎉 VALIDATION TERMINÉE{Colors.ENDC}")
    print_separator()
    
    # Recommandations
    if stats['skipped'] > 0:
        print(f"\n{Colors.WARNING}⚠️ {stats['skipped']} documents ont été skipped.{Colors.ENDC}")
        print(f"Pour continuer la validation, utiliser:")
        print(f"   python tools/validate_qrels_manual.py --qrels {args.output} --corpus {args.corpus}")
        print()

if __name__ == "__main__":
    main()
