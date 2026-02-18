#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse récursive de fichiers XML (dossier + sous-dossiers) pour :
- vérifier la présence d'une balise <ID> et compter les valeurs uniques + effectifs
- extraire les valeurs de cid associées à la balise <CONTEXT> et compter uniques + effectifs
- analyser au maximum N fichiers (si N = None -> analyser tout le dossier)

Conçu pour de très gros volumes (2.5M fichiers) :
- parsing en streaming via xml.etree.ElementTree.iterparse
- arrêt anticipé dès que <ID> et cid de <CONTEXT> sont trouvés
- option d'échantillonnage : premiers N ou random (reservoir sampling)

# Exemple: analyser 50 000 XML (les premiers), et afficher Top-50 cid
python tools/analyze_xml_ids_cids.py --root-dir "/home/fred/montage1/-- Projet RAG Avocats --/data_main/data/legifrance_extract_tmp/2026-01-14/legi/global/code_et_TNC_en_vigueur" --max-files 50000 --sample-method first --top-k 50

# Exemple: échantillon aléatoire de 100 000 XML
python tools/analyze_xml_ids_cids.py --root-dir "//home/fred/montage1/-- Projet RAG Avocats --/data_main/data/legifrance_extract_tmp/2026-01-14/legi/global/code_et_TNC_en_vigueur" --max-files 100000 --sample-method random --seed 42

# Exemple: tout analyser (attention volume) + export CSV
python tools/analyze_xml_ids_cids.py --root-dir "/home/fred/montage1/-- Projet RAG Avocats --/data_main/data/legifrance_extract_tmp/2026-01-14/legi/global/code_et_TNC_en_vigueur" --out-dir "/home/fred/montage1/-- Projet RAG Avocats --/rag_bench/tools/test_logs"

"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

# Noms de balises attendues (sans namespace)
ID_TAG = "ID"

# On gère les deux graphies rencontrées dans les dumps
CONTEXT_TAGS = {"CONTEXT", "CONTEXTE"}

# Dans les XML Legifrance, le cid est très souvent porté par <TEXTE ... cid="...">
# On accepte aussi TEXT si jamais certains fichiers utilisent l'anglais.
TEXTE_TAGS = {"TEXTE", "TEXT"}

# Certains XML utilisent cid ou CID selon les producteurs/versions
CID_ATTR_CANDIDATES = ("cid", "CID")


def strip_ns(tag: str) -> str:
    """Supprime un namespace éventuel: '{ns}ID' -> 'ID'."""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


@dataclass
class XmlExtract:
    """Résultat minimal d'extraction pour un fichier XML."""
    has_id: bool
    id_value: Optional[str]
    has_context_cid: bool
    context_cid_value: Optional[str]


def extract_id_and_context_cid(xml_path: Path) -> XmlExtract:
    """
    Extrait :
    - la valeur de la première balise <ID> rencontrée
    - la valeur du cid associé au bloc <CONTEXT>/<CONTEXTE> :
        - cid sur <CONTEXT>/<CONTEXTE> lui-même
        - OU cid sur un <TEXTE>/<TEXT> enfant de ce bloc

    Parsing streaming + clear() pour limiter la mémoire sur gros volumes.
    """
    id_value: Optional[str] = None
    context_cid: Optional[str] = None

    # Indique si on se trouve actuellement à l'intérieur d'un bloc CONTEXT/CONTEXTE
    in_context_block = False

    try:
        for event, elem in ET.iterparse(str(xml_path), events=("start", "end")):
            tag = strip_ns(elem.tag)

            if event == "start":
                # Début d’un bloc CONTEXT/CONTEXTE
                if tag in CONTEXT_TAGS:
                    in_context_block = True

                    # Si le cid est directement sur <CONTEXT>/<CONTEXTE>, on le prend
                    if context_cid is None:
                        for a in CID_ATTR_CANDIDATES:
                            v = elem.attrib.get(a)
                            if v:
                                context_cid = v.strip()
                                break

                # Si on est dans CONTEXT/CONTEXTE et qu’on rencontre un <TEXTE>/<TEXT>,
                # on cherche un cid dessus (cas courant dans les dumps Legifrance).
                if in_context_block and context_cid is None and tag in TEXTE_TAGS:
                    for a in CID_ATTR_CANDIDATES:
                        v = elem.attrib.get(a)
                        if v:
                            context_cid = v.strip()
                            break

            else:  # event == "end"
                # Récupération de l'ID sur l'événement 'end' (texte stable à ce moment-là)
                if tag == ID_TAG and id_value is None:
                    if elem.text:
                        v = elem.text.strip()
                        if v:
                            id_value = v

                # Fin d’un bloc CONTEXT/CONTEXTE
                if tag in CONTEXT_TAGS:
                    in_context_block = False

                # Dès qu'on a trouvé les deux, on peut arrêter (gain perf)
                if id_value is not None and context_cid is not None:
                    elem.clear()
                    break

                # Nettoyage régulier pour réduire mémoire
                elem.clear()

        return XmlExtract(
            has_id=(id_value is not None),
            id_value=id_value,
            has_context_cid=(context_cid is not None),
            context_cid_value=context_cid,
        )

    except ET.ParseError:
        # XML illisible : on le compte comme manquant côté stats
        return XmlExtract(has_id=False, id_value=None, has_context_cid=False, context_cid_value=None)
    except OSError:
        # Fichier non lisible : on le compte comme manquant côté stats
        return XmlExtract(has_id=False, id_value=None, has_context_cid=False, context_cid_value=None)




def iter_xml_files(root_dir: Path) -> Iterator[Path]:
    """Itère récursivement sur tous les .xml."""
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".xml"):
                yield Path(dirpath) / fn


def reservoir_sample(paths: Iterable[Path], k: int, seed: int = 0) -> Iterator[Path]:
    """
    Reservoir sampling: échantillonne k éléments d'un flux sans connaître sa taille,
    en O(k) mémoire.
    """
    rng = random.Random(seed)
    reservoir = []
    for i, p in enumerate(paths, start=1):
        if i <= k:
            reservoir.append(p)
        else:
            j = rng.randint(1, i)
            if j <= k:
                reservoir[j - 1] = p
    # On rend dans l'ordre "réservoir" (non trié)
    yield from reservoir




def write_counter_csv(counter: Counter, out_path: Path, header: Tuple[str, str]) -> None:
    """Écrit un Counter en CSV (valeur, count)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for value, count in counter.most_common():
            w.writerow([value, count])


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse <ID> et cid de <CONTEXT> sur un corpus XML")
    parser.add_argument("--root-dir", type=str, required=True, help="Dossier racine contenant des XML (avec sous-dossiers)")

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Nombre max de XML à analyser (None = tout analyser)",
    )

    parser.add_argument(
        "--sample-method",
        type=str,
        default="first",
        choices=["first", "random"],
        help="Si --max-files est défini: 'first' prend les premiers N (rapide), 'random' fait un échantillon aléatoire",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed pour --sample-method random",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Nombre de valeurs affichées (Top-K) pour les compteurs",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Si fourni, écrit les compteurs en CSV dans ce dossier",
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists() or not root_dir.is_dir():
        print(f"❌ root-dir invalide: {root_dir}", file=sys.stderr)
        return 2

    # Construction de l'itérateur de fichiers
    all_paths_iter = iter_xml_files(root_dir)

    # Sélection N fichiers si demandé
    if args.max_files is None:
        selected_paths = all_paths_iter
        planned = None
    else:
        n = max(0, args.max_files)
        planned = n
        if n == 0:
            selected_paths = iter(())
        elif args.sample_method == "first":
            def take_first(it: Iterable[Path], k: int) -> Iterator[Path]:
                for i, p in enumerate(it):
                    if i >= k:
                        break
                    yield p
            selected_paths = take_first(all_paths_iter, n)
        else:
            selected_paths = reservoir_sample(all_paths_iter, n, seed=args.seed)

    # Compteurs
    id_counter: Counter[str] = Counter()
    context_cid_counter: Counter[str] = Counter()

    total = 0
    missing_id = 0
    missing_context_cid = 0

    # Exemples de fichiers problématiques (limités à 5 chacun)
    # On stocke les chemins en str pour un affichage simple.
    no_id_examples = []
    no_context_cid_examples = []
    no_both_examples = []
    max_examples = 5

    for p in selected_paths:
        total += 1
        ex = extract_id_and_context_cid(p)

        has_id = ex.has_id
        has_context = ex.has_context_cid

        if not has_id:
            missing_id += 1
            if len(no_id_examples) < max_examples:
                no_id_examples.append(str(p))
        else:
            id_counter[ex.id_value] += 1  # type: ignore[arg-type]

        if not has_context:
            missing_context_cid += 1
            if len(no_context_cid_examples) < max_examples:
                no_context_cid_examples.append(str(p))
        else:
            context_cid_counter[ex.context_cid_value] += 1  # type: ignore[arg-type]

        # Cas "aucun des deux"
        if (not has_id) and (not has_context):
            if len(no_both_examples) < max_examples:
                no_both_examples.append(str(p))           

        # ParseError est compté comme "manquant" ; si tu veux le distinguer :
        # on repère via None/False mais ici on ne sait pas si c'est parse ou absence.
        # Option simple: détecter par tentative parse dans extract_id_and_context_cid.
        # (On garde minimal.)

        if planned is not None and total >= planned and args.sample_method == "random":
            # reservoir_sample renvoie exactement planned, donc inutile, mais inoffensif
            pass

        if total % 10000 == 0:
            print(f"… {total:,} fichiers analysés")

    # Résumé
    print("\n========== RÉSUMÉ ==========")
    print(f"Dossier: {root_dir}")
    print(f"Fichiers analysés: {total:,}" + ("" if planned is None else f" (max-files={planned:,}, method={args.sample_method})"))
    print(f"ID manquant: {missing_id:,} ({(missing_id/total*100):.2f}%)" if total else "ID manquant: n/a")
    print(f"CONTEXT cid manquant: {missing_context_cid:,} ({(missing_context_cid/total*100):.2f}%)" if total else "CONTEXT cid manquant: n/a")
    print(f"IDs uniques: {len(id_counter):,}")
    print(f"CONTEXT cids uniques: {len(context_cid_counter):,}")

    # Exemples de chemins pour debug data
    print("\n========== EXEMPLES (chemins) ==========")

    print("\nXML sans <ID> (jusqu'à 5) :")
    if no_id_examples:
        for path_str in no_id_examples:
            print(f"- {path_str}")
    else:
        print("Aucun (sur l'échantillon analysé).")

    print("\nXML sans cid sur <CONTEXT> (jusqu'à 5) :")
    if no_context_cid_examples:
        for path_str in no_context_cid_examples:
            print(f"- {path_str}")
    else:
        print("Aucun (sur l'échantillon analysé).")

    print("\nXML sans <ID> ET sans cid sur <CONTEXT> (jusqu'à 5) :")
    if no_both_examples:
        for path_str in no_both_examples:
            print(f"- {path_str}")
    else:
        print("Aucun (sur l'échantillon analysé).")


    # Top-K cids (utile, car généralement peu de cid dominent, contrairement aux IDs souvent uniques)
    top_k = max(0, args.top_k)

    print("\n========== TOP CONTEXT cid ==========")
    for v, c in context_cid_counter.most_common(top_k):
        hint = ""
        if v.startswith("LEGITEXT"):
            hint = " (probablement: texte consolidé / code)"
        elif v.startswith("JORFTEXT"):
            hint = " (probablement: Journal Officiel - texte)"
        print(f"{v}\t{c:,}{hint}")

    # Pour les IDs, souvent tout est unique (count=1), donc afficher les duplicats est plus informatif
    print("\n========== IDs en doublon (count > 1) ==========")
    dup_ids = [(v, c) for v, c in id_counter.items() if c > 1]
    dup_ids.sort(key=lambda x: x[1], reverse=True)
    if not dup_ids:
        print("Aucun doublon détecté (sur l'échantillon).")
    else:
        for v, c in dup_ids[:top_k]:
            print(f"{v}\t{c:,}")

    # Export CSV optionnel
    if args.out_dir:
        out_dir = Path(args.out_dir)
        write_counter_csv(id_counter, out_dir / "ids_counts.csv", header=("id", "count"))
        write_counter_csv(context_cid_counter, out_dir / "context_cid_counts.csv", header=("context_cid", "count"))
        print(f"\n✅ CSV écrits dans: {out_dir}")

    print("\nNote interprétation:")
    print("- JORF = Journal Officiel de la République Française (souvent préfixe JORFTEXT...).")
    print("- LEGITEXT est fréquemment utilisé pour des textes consolidés/codes (ex: Code civil).")
    print("Ensuite, on peut bâtir une table de correspondance cid -> nom du code, via tes métadonnées ou référentiels.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
