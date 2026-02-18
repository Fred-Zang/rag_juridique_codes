#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils_xml.py

Helpers XML Legifrance (LEGI).

Objectif V3 (ENRICHISSEMENT COMPLET):
- Extraction enrichie : article_num, corpus_juridique, nature, code_titre
- Extraction liens juridiques (JSON)
- Extraction structure sections (JSON) pour LEGITEXT et LEGISCTA
- Temporalité déjà présente (valid_from, valid_to, status)
- Production texte RAG-friendly

Usage:
    from datalake.utils_xml import parse_xml_bytes, ParsedDoc

    with open("article.xml", "rb") as f:
        parsed = parse_xml_bytes(f.read(), source_path="path/to/article.xml")

    print(parsed.doc_key)        # LEGIARTI000006648153
    print(parsed.code_titre)     # Code du travail
    print(parsed.liens)          # [{"typelien": "CREATION", ...}, ...]
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# lxml est plus robuste et plus rapide sur du gros XML.
try:
    from lxml import etree  # type: ignore
    _HAS_LXML = True
except ImportError:
    import xml.etree.ElementTree as etree  # type: ignore
    _HAS_LXML = False


logger = logging.getLogger(__name__)


@dataclass
class ParsedDoc:
    """
    Structure de sortie du parsing XML (V3 enrichie).

    Champs principaux:
    - doc_key: Identifiant unique (LEGIARTI..., LEGITEXT..., LEGISCTA...)
    - corpus_juridique: Code parent (LEGITEXT...)
    - code_titre: Nom lisible du code ("Code du travail")
    - nature: Type de document ("Article", "CODE", "SECTION")
    - body_text: Texte du document (articles uniquement)

    Champs JSON (nouveaux):
    - liens: Relations juridiques (CREATION, ABROGATION, CITATION...)
    - struct_sections: Sections enfants directes (pour LEGITEXT/LEGISCTA)
    """
    doc_key: str
    title: Optional[str]
    body_text: str
    valid_from: Optional[str]
    valid_to: Optional[str]
    status: Optional[str]
    article_num: Optional[str]
    corpus_juridique: Optional[str]
    # ⭐ NOUVEAUX CHAMPS V3
    nature: Optional[str]            # "Article", "CODE", "SECTION"
    code_titre: Optional[str]        # "Code du travail", "Code de commerce"
    liens: Optional[str]             # JSON string: [{"typelien": ..., "sens": ..., "id": ..., "num": ...}, ...]
    struct_sections: Optional[str]   # JSON string: [{"id": ..., "niv": ..., "titre": ...}, ...]


def _strip_ns(tag: str) -> str:
    """Retire un namespace éventuel {ns}tag."""
    if tag is None:
        return ""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _iter_nodes(root) -> Iterable:
    """Itérateur sur tous les noeuds (namespace-safe)."""
    return root.iter()


def _collect_text_under(node) -> str:
    """Concatène tous les textes (text + tail) sous un noeud."""
    parts: List[str] = []
    for n in node.iter():
        if n.text:
            parts.append(n.text)
        if n.tail:
            parts.append(n.tail)
    return "".join(parts)


def _match_tag(node, tag_upper: str) -> bool:
    """Vérifie si le tag du noeud (sans namespace) correspond."""
    return _strip_ns(getattr(node, "tag", "")).upper() == tag_upper


def _find_nodes_by_tag(root, tag_upper: str) -> List:
    """Trouve tous les noeuds dont le tag (sans namespace) est tag_upper."""
    out = []
    for n in _iter_nodes(root):
        if _match_tag(n, tag_upper):
            out.append(n)
    return out


def _find_first_text_by_tag(root, tag_upper: str) -> Optional[str]:
    """Prend le premier noeud tag_upper et renvoie son texte concaténé."""
    nodes = _find_nodes_by_tag(root, tag_upper)
    if not nodes:
        return None
    txt = _collect_text_under(nodes[0])
    return txt.strip() if txt else None


def _get_attr(node, attr_name: str) -> Optional[str]:
    """Récupère un attribut de manière robuste (lxml ou ElementTree)."""
    if node is None:
        return None
    getter = getattr(node, "get", None)
    if getter:
        # Essaie plusieurs variantes de casse
        val = getter(attr_name) or getter(attr_name.lower()) or getter(attr_name.upper())
        return val.strip() if val else None
    return None


# Expressions régulières pour normalisation HTML-like
_HTML_LIKE_BREAKS = re.compile(r"</?(br|BR)\s*/?>")
_HTML_LIKE_BLOCKS = re.compile(r"</?(p|P|div|DIV|tr|TR|li|LI|h1|H1|h2|H2|h3|H3)\b[^>]*>")
_HTML_LIKE_CELL = re.compile(r"</?(td|TD|th|TH)\b[^>]*>")
_HTML_LIKE_TAGS = re.compile(r"<[^>]+>")


def normalize_text(s: str) -> str:
    """
    Normalise un texte:
    - remplace quelques tags HTML-like par des séparateurs
    - supprime le reste des tags
    - normalise les espaces
    """
    if not s:
        return ""

    s = _HTML_LIKE_BREAKS.sub("\n", s)
    s = _HTML_LIKE_CELL.sub("\t", s)
    s = _HTML_LIKE_BLOCKS.sub("\n", s)
    s = _HTML_LIKE_TAGS.sub("", s)

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_corpus_juridique_from_xml(root) -> Optional[str]:
    """
    Extrait LEGITEXT... (corpus_juridique) directement depuis le XML.

    Cherche dans l'ordre:
    1. Attribut `cid` sur un noeud TEXTE (ex: <TEXTE cid="LEGITEXT000006072050" ...>)
    2. Balise <CID> dans META_TEXTE_CHRONICLE

    Args:
        root: Racine XML (Element)

    Returns:
        corpus_juridique (LEGITEXT...) ou None
    """
    if root is None:
        return None

    # 1. Cherche attribut cid sur noeud TEXTE
    for node in _iter_nodes(root):
        if _match_tag(node, "TEXTE"):
            cid = _get_attr(node, "cid")
            if cid and re.fullmatch(r"LEGITEXT\d+", cid):
                logger.debug("corpus_juridique extrait du XML (TEXTE[@cid]): %s", cid)
                return cid

    # 2. Cherche balise <CID> (pour LEGITEXT)
    cid_text = _find_first_text_by_tag(root, "CID")
    if cid_text and re.fullmatch(r"LEGITEXT\d+", cid_text):
        logger.debug("corpus_juridique extrait du XML (<CID>): %s", cid_text)
        return cid_text

    logger.debug("Aucun corpus_juridique trouvé dans le XML")
    return None


def extract_corpus_juridique_from_path(source_path: str) -> Optional[str]:
    """
    Extrait LEGITEXT... (corpus_juridique) depuis le chemin du fichier XML.

    Fallback utilisé quand l'extraction depuis le XML (cid) échoue.

    Exemple:
        .../LEGI/TEXT/00/00/06/07/20/LEGITEXT000006072050/article/...
        → LEGITEXT000006072050

    Args:
        source_path: Chemin complet du fichier XML source

    Returns:
        corpus_juridique (LEGITEXT...) ou None
    """
    if not source_path:
        return None

    match = re.search(r'LEGITEXT\d+', source_path)
    if match:
        corpus_juridique = match.group(0)
        logger.debug("corpus_juridique extrait du chemin: %s", corpus_juridique)
        return corpus_juridique

    logger.debug("Aucun corpus_juridique trouvé dans le chemin: %s", source_path)
    return None


def _extract_doc_key(root) -> Optional[str]:
    """
    Essaye d'extraire un identifiant stable.

    Returns:
        doc_key si trouvé, None sinon
    """
    for node in _iter_nodes(root):
        if _match_tag(node, "ID"):
            txt = _collect_text_under(node).strip()
            if txt.startswith("LEGI") or txt.startswith("JORF"):
                logger.debug("doc_key trouvé: %s", txt)
                return txt

    logger.warning("Aucun doc_key trouvé (ID manquant)")
    return None


def _extract_title(root) -> Optional[str]:
    """
    Cherche un titre potentiel. La structure varie; on tente quelques tags usuels.

    Pour les SECTION_TA, priorise TITRE_TA (ex: "Partie législative")
    avant TITRE_TXT (qui contient le nom du code parent).
    """
    # Detecte si c'est une section (SECTION_TA)
    root_tag = _strip_ns(getattr(root, "tag", "")).upper()
    is_section = root_tag == "SECTION_TA"

    if is_section:
        # Pour les sections, prioriser TITRE_TA
        tags_order = ["TITRE_TA", "TITRE", "LIBELLE", "INTITULE"]
    else:
        # Pour les autres, ordre standard
        tags_order = ["TITRE", "TITRE_TXT", "TITRE_TA", "LIBELLE", "INTITULE"]

    for tag in tags_order:
        t = _find_first_text_by_tag(root, tag)
        if t and len(t) >= 3:
            logger.debug("Titre trouvé: %s... (tag=%s)", t[:50], tag)
            return t

    logger.debug("Aucun titre trouvé")
    return None


def _extract_temporal_fields(root) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extrait des champs temporels si présents.

    Returns:
        (valid_from, valid_to, status)
    """
    valid_from = _find_first_text_by_tag(root, "DATE_DEBUT")
    valid_to = _find_first_text_by_tag(root, "DATE_FIN")
    status = _find_first_text_by_tag(root, "ETAT")

    # Fallback: cherche dans VERSION si pas trouvé
    if not status:
        for node in _iter_nodes(root):
            if _match_tag(node, "VERSION"):
                status = _get_attr(node, "etat")
                if status:
                    break

    logger.debug("Temporalité: valid_from=%s, valid_to=%s, status=%s", valid_from, valid_to, status)

    return valid_from, valid_to, status


def _extract_article_num(root) -> Optional[str]:
    """
    Extrait le numéro d'article (ex: L1234-5, R3324-22).

    Cherche dans les tags: NUM, NUMERO, NUM_ARTICLE

    Returns:
        Numéro d'article normalisé ou None
    """
    for tag in ["NUM", "NUMERO", "NUM_ARTICLE"]:
        num = _find_first_text_by_tag(root, tag)
        if num and len(num) >= 2:
            num_clean = num.strip()
            logger.debug("Numéro article trouvé: %s (tag=%s)", num_clean, tag)
            return num_clean

    logger.debug("Aucun numéro article trouvé")
    return None


def _extract_nature(root) -> Optional[str]:
    """
    ⭐ NOUVEAU V3

    Extrait la nature du document.

    Cherche dans:
    1. <NATURE>Article</NATURE> ou <NATURE>CODE</NATURE>
    2. Déduit depuis la balise racine si absent

    Returns:
        "Article", "CODE", "SECTION", ou None
    """
    # 1. Cherche balise <NATURE>
    nature = _find_first_text_by_tag(root, "NATURE")
    if nature:
        logger.debug("Nature trouvée: %s", nature)
        return nature

    # 2. Déduit depuis la racine
    root_tag = _strip_ns(getattr(root, "tag", "")).upper()
    if root_tag == "ARTICLE":
        return "Article"
    elif root_tag == "TEXTELR":
        return "CODE"
    elif root_tag == "SECTION_TA":
        return "SECTION"

    logger.debug("Nature non déterminée")
    return None


def _extract_code_titre(root) -> Optional[str]:
    """
    ⭐ NOUVEAU V3

    Extrait le nom lisible du code juridique.

    Cherche <TITRE_TXT> dans CONTEXTE (ex: "Code du travail", "Code de commerce")

    Returns:
        Nom du code ou None
    """
    # Cherche dans CONTEXTE > TEXTE > TITRE_TXT
    for contexte_node in _find_nodes_by_tag(root, "CONTEXTE"):
        for texte_node in _find_nodes_by_tag(contexte_node, "TEXTE"):
            for titre_node in _find_nodes_by_tag(texte_node, "TITRE_TXT"):
                txt = _collect_text_under(titre_node).strip()
                if txt and len(txt) >= 3:
                    logger.debug("code_titre trouvé: %s", txt)
                    return txt

    # Fallback: cherche TITRE_TXT directement
    titre = _find_first_text_by_tag(root, "TITRE_TXT")
    if titre and len(titre) >= 3:
        logger.debug("code_titre trouvé (fallback): %s", titre)
        return titre

    logger.debug("Aucun code_titre trouvé")
    return None


def _extract_liens(root) -> Optional[str]:
    """
    ⭐ NOUVEAU V3

    Extrait les liens juridiques en JSON.

    Attributs extraits: typelien, sens, id, num

    Exemple de sortie:
        [
            {"typelien": "CREATION", "sens": "source", "id": "JORFTEXT000000414656", "num": "2002-1095"},
            {"typelien": "CITATION", "sens": "source", "id": "LEGIARTI000006648138", "num": "L322-4-6"}
        ]

    Returns:
        JSON string ou None si aucun lien
    """
    liens_list: List[Dict[str, Any]] = []

    # Cherche dans <LIENS>/<LIEN ...>
    for liens_node in _find_nodes_by_tag(root, "LIENS"):
        for lien_node in liens_node:
            if not _match_tag(lien_node, "LIEN"):
                continue

            lien_dict: Dict[str, Any] = {}

            # Extraction des attributs clés
            typelien = _get_attr(lien_node, "typelien")
            if typelien:
                lien_dict["typelien"] = typelien

            sens = _get_attr(lien_node, "sens")
            if sens:
                lien_dict["sens"] = sens

            lien_id = _get_attr(lien_node, "id")
            if lien_id:
                lien_dict["id"] = lien_id

            num = _get_attr(lien_node, "num")
            if num:
                lien_dict["num"] = num

            # Ajoute seulement si au moins typelien ou id présent
            if lien_dict.get("typelien") or lien_dict.get("id"):
                liens_list.append(lien_dict)

    if liens_list:
        logger.debug("Liens extraits: %d", len(liens_list))
        return json.dumps(liens_list, ensure_ascii=False)

    logger.debug("Aucun lien trouvé")
    return None


def _extract_struct_sections(root) -> Optional[str]:
    """
    ⭐ NOUVEAU V3

    Extrait les sections enfants directes (niveau +1) en JSON.

    Pour LEGITEXT: cherche dans <STRUCT>/<LIEN_SECTION_TA>
    Pour LEGISCTA: cherche dans <STRUCTURE_TA>/<LIEN_SECTION_TA>

    Attributs extraits: id, niv, titre (texte du noeud)

    Exemple de sortie:
        [
            {"id": "LEGISCTA000006113738", "niv": "2", "titre": "LIVRE Ier : Du commerce en général."},
            {"id": "LEGISCTA000006113739", "niv": "2", "titre": "LIVRE II : Des sociétés commerciales..."}
        ]

    Returns:
        JSON string ou None si aucune section
    """
    sections_list: List[Dict[str, Any]] = []

    # Cherche dans <STRUCT> ou <STRUCTURE_TA>
    struct_nodes = _find_nodes_by_tag(root, "STRUCT") + _find_nodes_by_tag(root, "STRUCTURE_TA")

    for struct_node in struct_nodes:
        for section_node in struct_node:
            if not _match_tag(section_node, "LIEN_SECTION_TA"):
                continue

            section_dict: Dict[str, Any] = {}

            # ID de la section
            section_id = _get_attr(section_node, "id")
            if section_id:
                section_dict["id"] = section_id

            # Niveau hiérarchique
            niv = _get_attr(section_node, "niv")
            if niv:
                section_dict["niv"] = niv

            # Titre (texte du noeud)
            titre = _collect_text_under(section_node).strip()
            if titre:
                section_dict["titre"] = titre

            # Ajoute seulement si id présent
            if section_dict.get("id"):
                sections_list.append(section_dict)

    if sections_list:
        logger.debug("Sections extraites: %d", len(sections_list))
        return json.dumps(sections_list, ensure_ascii=False)

    logger.debug("Aucune section trouvée")
    return None


def _extract_body_text(root) -> str:
    """
    Extraction robuste du texte:

    1) Priorité: BLOC_TEXTUEL/CONTENU
    2) Ajout: NOTA/CONTENU (fusionné dans le body_text)
    3) Fallback: concat de tous les CONTENU
    4) Dernier fallback: tout le texte du document

    Returns:
        Texte normalisé du document
    """
    contents = []

    # 1) CONTENU sous BLOC_TEXTUEL
    bloc_nodes = _find_nodes_by_tag(root, "BLOC_TEXTUEL")
    for b in bloc_nodes:
        for n in b.iter():
            if _match_tag(n, "CONTENU"):
                txt = _collect_text_under(n)
                if txt and txt.strip():
                    contents.append(txt)

    logger.debug("Texte extrait de BLOC_TEXTUEL: %d segments", len(contents))

    # 2) NOTA/CONTENU (fusionné - comportement conservé)
    nota_nodes = _find_nodes_by_tag(root, "NOTA")
    for nota in nota_nodes:
        for n in nota.iter():
            if _match_tag(n, "CONTENU"):
                txt = _collect_text_under(n)
                if txt and txt.strip():
                    contents.append(txt)

    if nota_nodes:
        logger.debug("Texte extrait de NOTA: %d segments", len([c for c in contents if c]))

    # 3) Fallback: tous les CONTENU
    if not contents:
        logger.debug("Fallback: extraction de tous les CONTENU")
        for n in _iter_nodes(root):
            if _match_tag(n, "CONTENU"):
                txt = _collect_text_under(n)
                if txt and txt.strip():
                    contents.append(txt)

    # 4) Dernier fallback
    if not contents:
        logger.warning("Aucun CONTENU trouvé, fallback sur texte complet")
        contents.append(_collect_text_under(root))

    merged = "\n\n".join([normalize_text(c) for c in contents if c])

    text_len = len(merged)
    logger.debug("Texte final: %d caractères", text_len)

    return merged.strip()


def parse_xml_bytes(xml_bytes: bytes, source_path: str = "") -> ParsedDoc:
    """
    Parse un XML (bytes) et retourne un objet ParsedDoc enrichi (V3).

    Supporte les 3 patterns XML Legifrance:
    - LEGIARTI (articles avec contenu)
    - LEGITEXT (structure code)
    - LEGISCTA (sections)

    Args:
        xml_bytes: Contenu XML en bytes
        source_path: Chemin du fichier XML (pour extraction corpus_juridique fallback)

    Returns:
        ParsedDoc avec tous les champs enrichis (V3)

    Raises:
        ValueError: Si XML vide
        etree.ParseError: Si XML malformé
    """
    if not xml_bytes:
        raise ValueError("Empty XML bytes")

    logger.debug("Parsing XML (%d bytes)...", len(xml_bytes))

    try:
        if _HAS_LXML:
            root = etree.fromstring(xml_bytes)  # nosec - source locale contrôlée
        else:
            root = etree.fromstring(xml_bytes)  # type: ignore
    except etree.ParseError as e:
        logger.error("Erreur parsing XML: %s", e)
        raise

    # Extraction standard
    doc_key = _extract_doc_key(root) or "UNKNOWN"
    title = _extract_title(root)
    valid_from, valid_to, status = _extract_temporal_fields(root)
    body_text = _extract_body_text(root)
    article_num = _extract_article_num(root)

    # corpus_juridique: priorité au XML (cid), fallback sur le chemin
    corpus_juridique = extract_corpus_juridique_from_xml(root)
    if not corpus_juridique:
        corpus_juridique = extract_corpus_juridique_from_path(source_path)

    # ⭐ NOUVEAU V3: Extraction enrichie
    nature = _extract_nature(root)
    code_titre = _extract_code_titre(root)
    liens = _extract_liens(root)
    struct_sections = _extract_struct_sections(root)

    logger.debug(
        "Parsing terminé: doc_key=%s, nature=%s, code_titre=%s, liens=%s, struct=%s",
        doc_key, nature, code_titre[:30] if code_titre else None,
        "oui" if liens else "non", "oui" if struct_sections else "non"
    )

    return ParsedDoc(
        doc_key=doc_key,
        title=title,
        body_text=body_text,
        valid_from=valid_from,
        valid_to=valid_to,
        status=status,
        article_num=article_num,
        corpus_juridique=corpus_juridique,
        # ⭐ NOUVEAU V3
        nature=nature,
        code_titre=code_titre,
        liens=liens,
        struct_sections=struct_sections,
    )
