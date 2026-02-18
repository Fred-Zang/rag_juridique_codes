#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_chunk_xml_html.py - VERSION 2 avec auto-détection
Phase 1 - Visualisation XML et métadonnées chunks (VERSION HTML)

Objectif:
- Inspecter un chunk depuis le Parquet gold (par doc_key, index, ou corpus_juridique)
- Afficher toutes les métadonnées disponibles en HTML
- Lire et afficher l'aperçu XML depuis source_path
- Traçabilité complète des champs (disponibles, utilisés, affichés)
- Liens cliquables vers Legifrance et fichiers XML
- Auto-détection du corpus via paths.gold_corpus_dir

Usage SIMPLIFIÉ (auto-détection) :
    # Par doc_key (corpus auto-détecté)
    python tools/inspect_chunk_xml_html.py --doc-key LEGIARTI000006648153 --show-xml

    # Par corpus_juridique (chercher chunks Code du Travail)
    python tools/inspect_chunk_xml_html.py --corpus-juridique LEGITEXT000006072050 --limit 10

    # Par index (debug)
    python tools/inspect_chunk_xml_html.py --index 0 --show-xml

Usage classique (avec --corpus) :
    # Spécifier corpus manuellement
    python tools/inspect_chunk_xml_html.py --corpus /chemin/vers/gold/chunks --doc-key XXX

Logs: tools/test_logs/inspect_chunk_xml_html_<timestamp>.md
Résultats: tools/inspections/inspect_chunk_xml_<timestamp>.html ← HTML !
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

# Import du module paths pour centraliser les chemins
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag_bench.paths import get_project_paths, resolve_path
from rag_bench.io_parquet import load_chunks_from_parquet


# ============================================================================
# CONFIGURATION LOGGING
# ============================================================================

class MarkdownLogFormatter(logging.Formatter):
    """Formatter orienté Markdown pour les logs."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        msg = record.getMessage()
        level_icon = {
            'DEBUG': '🔍',
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }.get(record.levelname, '•')
        return f"- [{ts}] {level_icon} **{record.levelname}** — {msg}"


def setup_logging(log_dir: Path, level: str = "INFO") -> tuple[logging.Logger, Path]:
    """Configure un logger avec console + fichier Markdown."""
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_log_path = log_dir / f"inspect_chunk_xml_html_{ts}.md"

    # En-tête Markdown
    header = [
        f"# 🔍 Logs Inspection Chunk XML (HTML)",
        "",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Script**: `tools/inspect_chunk_xml_html.py`",
        f"- **Logs**: `{md_log_path}`",
        "",
        "---",
        "",
    ]
    md_log_path.write_text("\n".join(header), encoding="utf-8")

    logger = logging.getLogger("inspect_chunk_xml_html")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Nettoyage handlers existants
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    # Fichier markdown handler
    fh = logging.FileHandler(md_log_path, mode="a", encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(MarkdownLogFormatter())
    logger.addHandler(fh)

    return logger, md_log_path


# ============================================================================
# CHARGEMENT & RECHERCHE CHUNKS
# ============================================================================

def load_chunks_safe(corpus_path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Charge le corpus Parquet avec gestion d'erreurs."""
    logger.info(f"📦 Chargement corpus: {corpus_path}")
    
    try:
        chunks = load_chunks_from_parquet(
            parquet_path=corpus_path,
            flatten=True  # Format plat pour accès direct
        )
        logger.info(f"✅ Corpus chargé: {len(chunks):,} chunks (format plat)")
        return chunks
    except Exception as e:
        logger.error(f"❌ Erreur chargement corpus: {e}")
        raise


def find_chunk_by_doc_key(chunks: List[Dict[str, Any]], doc_key: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Trouve un chunk par doc_key."""
    logger.info(f"🔎 Recherche doc_key: {doc_key}")
    
    for chunk in chunks:
        if chunk.get("doc_key") == doc_key:
            logger.info(f"✅ Chunk trouvé: {doc_key}")
            return chunk
    
    logger.warning(f"⚠️ Chunk non trouvé: {doc_key}")
    return None


def find_chunks_by_corpus_juridique(chunks: List[Dict[str, Any]], corpus_juridique: str, limit: int, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Trouve des chunks par corpus_juridique (avec limite)."""
    logger.info(f"🔎 Recherche corpus_juridique: {corpus_juridique} (limite: {limit})")

    results = []
    for chunk in chunks:
        if chunk.get("corpus_juridique") == corpus_juridique:
            results.append(chunk)
            if len(results) >= limit:
                break

    logger.info(f"✅ {len(results)} chunks trouvés pour corpus_juridique={corpus_juridique}")
    return results


def find_chunk_by_index(chunks: List[Dict[str, Any]], index: int, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Récupère un chunk par index (pour debug)."""
    logger.info(f"🔎 Récupération chunk index: {index}")
    
    if 0 <= index < len(chunks):
        logger.info(f"✅ Chunk récupéré: index {index}")
        return chunks[index]
    else:
        logger.error(f"❌ Index invalide: {index} (corpus: {len(chunks)} chunks)")
        return None


# ============================================================================
# ANALYSE CHAMPS
# ============================================================================

def analyze_chunk_fields(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Analyse complète des champs du chunk."""
    all_fields = list(chunk.keys())
    
    filled_fields = [k for k, v in chunk.items() if v is not None and str(v).strip() != "" and str(v) != "nan"]
    empty_fields = [k for k in all_fields if k not in filled_fields]
    
    # Champs utilisés pour l'affichage
    used_for_display = [
        "doc_key", "corpus_juridique", "article_num", "doc_type",
        "valid_from", "valid_to", "status", "chunk_index",
        "source_path", "text", "chunk_text",
        "start_char", "end_char", "chunk_id", "unit_id", "version_key"
    ]
    
    not_displayed = [k for k in all_fields if k not in used_for_display]
    
    return {
        "all_fields": all_fields,
        "filled_fields": filled_fields,
        "empty_fields": empty_fields,
        "used_for_display": used_for_display,
        "not_displayed": not_displayed
    }


# ============================================================================
# LECTURE XML
# ============================================================================

def read_xml_preview(source_path: str, n_lines: int, logger: logging.Logger) -> str:
    """Lit les N premières lignes du fichier XML."""
    xml_path = Path(source_path)
    
    if not xml_path.exists():
        logger.warning(f"⚠️ Fichier XML non trouvé: {source_path}")
        return f"⚠️ Fichier XML non trouvé sur disque: {source_path}"
    
    try:
        logger.info(f"📄 Lecture XML: {source_path}")
        lines = []
        with xml_path.open("r", encoding="utf-8", errors="replace") as f:
            for i in range(n_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
        
        logger.info(f"✅ {len(lines)} lignes XML lues")
        return "\n".join(lines)
    
    except Exception as e:
        logger.error(f"❌ Erreur lecture XML: {e}")
        return f"❌ Erreur lecture XML: {e}"


# ============================================================================
# GÉNÉRATION HTML
# ============================================================================

def make_legifrance_url(doc_key: Optional[str]) -> Optional[str]:
    """Construit URL Legifrance pour un doc_key."""
    if not doc_key or not doc_key.startswith("LEGIARTI"):
        return None
    return f"https://www.legifrance.gouv.fr/codes/article_lc/{doc_key}"


def make_file_uri(path: str) -> str:
    """Construit URI file:// pour un chemin."""
    if not path.startswith("/"):
        path = str(Path(path).absolute())
    return "file://" + quote(path)


def format_chunk_inspection_html(
    chunks_data: List[tuple[Dict[str, Any], Dict[str, Any], Optional[str]]],
    show_xml: bool
) -> str:
    """
    Formate l'inspection complète de chunks en HTML.
    
    Args:
        chunks_data: Liste de (chunk, fields_analysis, xml_preview)
        show_xml: Afficher ou non le XML
    
    Returns:
        String HTML complet
    """
    html_parts = []
    
    # Header avec CSS (style analyze_results.py)
    html_parts.append(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Inspection Chunks</title>
        <style>
            :root {{
                --bg: #0f1115;
                --panel: #161a22;
                --panel-2: #111520;
                --panel-3: #0c0f16;
                --text: #e7eaf0;
                --muted: #aab2c0;
                --border: #2a3140;
                --shadow: rgba(0, 0, 0, 0.45);
                --accent: #8b93ff;
                --accent-2: #b07cff;
                --ok: #38d070;
                --info: #3aa7ff;
                --warn: #ffcc66;
                --highlight: rgba(255, 204, 102, 0.20);
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: var(--bg);
                color: var(--text);
            }}

            .header {{
                background: linear-gradient(135deg, #2b2f6a 0%, #3a2457 100%);
                color: white;
                padding: 30px;
                border-radius: 12px;
                margin-bottom: 30px;
                border: 1px solid rgba(255,255,255,0.08);
                box-shadow: 0 10px 30px var(--shadow);
            }}

            .chunk-section {{
                background: var(--panel);
                padding: 25px;
                margin-bottom: 30px;
                border-radius: 12px;
                border: 1px solid var(--border);
                box-shadow: 0 8px 26px var(--shadow);
            }}

            .chunk-title {{
                font-size: 24px;
                color: var(--accent);
                margin-bottom: 20px;
                border-bottom: 3px solid rgba(139,147,255,0.75);
                padding-bottom: 10px;
            }}

            .metadata {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}

            .meta-item {{
                background: var(--panel-3);
                padding: 15px;
                border-radius: 10px;
                border: 1px solid var(--border);
            }}

            .meta-label {{
                font-size: 11px;
                color: var(--muted);
                text-transform: uppercase;
                font-weight: 700;
                letter-spacing: 0.05em;
                margin-bottom: 8px;
            }}

            .meta-value {{
                font-size: 14px;
                color: var(--text);
                font-family: 'Courier New', monospace;
                word-break: break-all;
            }}

            .meta-value a {{
                color: var(--accent);
                text-decoration: none;
                border-bottom: 1px dotted var(--accent);
            }}

            .meta-value a:hover {{
                color: var(--ok);
                border-bottom-color: var(--ok);
            }}

            .text-content {{
                padding: 15px;
                background: rgba(255,255,255,0.03);
                border: 1px solid var(--border);
                border-radius: 12px;
                line-height: 1.8;
                color: var(--text);
                white-space: pre-wrap;
                font-size: 14px;
                margin-bottom: 20px;
            }}

            .xml-content {{
                padding: 15px;
                background: var(--panel-3);
                border: 1px solid var(--border);
                border-radius: 12px;
                line-height: 1.6;
                color: #88c0d0;
                white-space: pre;
                font-size: 12px;
                font-family: 'Courier New', monospace;
                overflow-x: auto;
                margin-bottom: 20px;
            }}

            .section-title {{
                font-size: 18px;
                color: var(--accent);
                margin: 25px 0 15px 0;
                font-weight: 700;
            }}

            .badge {{
                display: inline-block;
                padding: 5px 12px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: 700;
                margin-right: 10px;
                border: 1px solid rgba(255,255,255,0.10);
            }}

            .badge-ok {{
                background: rgba(56,208,112,0.20);
                color: var(--ok);
                border-color: rgba(56,208,112,0.35);
            }}

            .badge-warn {{
                background: rgba(255,204,102,0.20);
                color: var(--warn);
                border-color: rgba(255,204,102,0.35);
            }}

            .badge-empty {{
                background: rgba(170,178,192,0.15);
                color: var(--muted);
                border-color: rgba(170,178,192,0.25);
            }}

            .fields-list {{
                background: var(--panel-2);
                padding: 15px;
                border-radius: 10px;
                border: 1px solid var(--border);
                margin-bottom: 20px;
            }}

            .field-item {{
                padding: 8px 0;
                border-bottom: 1px solid rgba(42,49,64,0.5);
                font-size: 13px;
            }}

            .field-item:last-child {{
                border-bottom: none;
            }}

            .field-name {{
                font-family: 'Courier New', monospace;
                color: var(--accent);
                font-weight: 600;
            }}

            .field-value {{
                color: var(--muted);
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }}

            .stats {{
                background: rgba(139,147,255,0.10);
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 20px;
                border: 1px solid rgba(139,147,255,0.22);
            }}

            .doc-key-large {{
                font-family: 'Courier New', monospace;
                font-size: 20px;
                color: var(--accent);
                font-weight: 700;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Inspection Chunks - Format HTML</h1>
            <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Chunks inspectés: {len(chunks_data)}</p>
        </div>
    """)
    
    # Pour chaque chunk
    for i, (chunk, fields_analysis, xml_preview) in enumerate(chunks_data, 1):
        doc_key = chunk.get("doc_key", "UNKNOWN")
        corpus_juridique = chunk.get("corpus_juridique")
        article_num = chunk.get("article_num")
        
        # Liens
        legifrance_url = make_legifrance_url(doc_key)
        source_path = chunk.get("source_path")
        xml_file_uri = make_file_uri(source_path) if source_path else None
        
        html_parts.append(f"""
        <div class="chunk-section">
            <div class="chunk-title">
                Chunk {i}/{len(chunks_data)}: 
                <span class="doc-key-large">{html.escape(doc_key)}</span>
            </div>
        """)
        
        # Métadonnées principales en grille
        html_parts.append("""
            <div class="metadata">
        """)
        
        # doc_key avec lien Legifrance
        if legifrance_url:
            html_parts.append(f"""
                <div class="meta-item">
                    <div class="meta-label">Doc Key</div>
                    <div class="meta-value">
                        <a href="{legifrance_url}" target="_blank">{html.escape(doc_key)} 🔗</a>
                    </div>
                </div>
            """)
        else:
            html_parts.append(f"""
                <div class="meta-item">
                    <div class="meta-label">Doc Key</div>
                    <div class="meta-value">{html.escape(doc_key)}</div>
                </div>
            """)
        
        # corpus_juridique
        html_parts.append(f"""
            <div class="meta-item">
                <div class="meta-label">Corpus Juridique</div>
                <div class="meta-value">{html.escape(str(corpus_juridique or 'N/A'))}</div>
            </div>
        """)
        
        # article_num
        html_parts.append(f"""
            <div class="meta-item">
                <div class="meta-label">Article Num</div>
                <div class="meta-value">{html.escape(str(article_num or 'N/A'))}</div>
            </div>
        """)
        
        # doc_type
        doc_type = chunk.get("doc_type")
        html_parts.append(f"""
            <div class="meta-item">
                <div class="meta-label">Doc Type</div>
                <div class="meta-value">{html.escape(str(doc_type or 'N/A'))}</div>
            </div>
        """)
        
        # valid_from
        valid_from = chunk.get("valid_from")
        html_parts.append(f"""
            <div class="meta-item">
                <div class="meta-label">Valid From</div>
                <div class="meta-value">{html.escape(str(valid_from or 'N/A'))}</div>
            </div>
        """)
        
        # valid_to
        valid_to = chunk.get("valid_to")
        html_parts.append(f"""
            <div class="meta-item">
                <div class="meta-label">Valid To</div>
                <div class="meta-value">{html.escape(str(valid_to or 'N/A'))}</div>
            </div>
        """)
        
        # status
        status = chunk.get("status")
        html_parts.append(f"""
            <div class="meta-item">
                <div class="meta-label">Status</div>
                <div class="meta-value">
                    <span class="badge badge-ok">{html.escape(str(status or 'N/A'))}</span>
                </div>
            </div>
        """)
        
        # chunk_index
        chunk_index = chunk.get("chunk_index")
        html_parts.append(f"""
            <div class="meta-item">
                <div class="meta-label">Chunk Index</div>
                <div class="meta-value">{html.escape(str(chunk_index if chunk_index is not None else 'N/A'))}</div>
            </div>
        """)
        
        # source_path avec lien file://
        if xml_file_uri:
            html_parts.append(f"""
                <div class="meta-item" style="grid-column: 1 / -1;">
                    <div class="meta-label">Source Path (XML)</div>
                    <div class="meta-value">
                        <a href="{xml_file_uri}" target="_blank">{html.escape(source_path)} 📂</a>
                    </div>
                </div>
            """)
        else:
            html_parts.append(f"""
                <div class="meta-item" style="grid-column: 1 / -1;">
                    <div class="meta-label">Source Path (XML)</div>
                    <div class="meta-value">{html.escape(str(source_path or 'N/A'))}</div>
                </div>
            """)
        
        html_parts.append("""
            </div>
        """)
        
        # Texte chunk
        text = chunk.get('text') or chunk.get('chunk_text', '')
        text_preview = text[:500] + ("..." if len(text) > 500 else "")
        html_parts.append(f"""
            <div class="section-title">📝 Texte Chunk</div>
            <div class="text-content">{html.escape(text_preview)}</div>
            <div class="stats">
                Longueur totale: <strong>{len(text):,}</strong> caractères
            </div>
        """)
        
        # Traçabilité champs
        html_parts.append(f"""
            <div class="section-title">🔬 Traçabilité Champs</div>
            <div class="stats">
                Total champs disponibles: <strong>{len(fields_analysis['all_fields'])}</strong> |
                Champs remplis: <strong>{len(fields_analysis['filled_fields'])}/{len(fields_analysis['all_fields'])}</strong> |
                Champs vides: <strong>{len(fields_analysis['empty_fields'])}</strong>
            </div>
        """)
        
        # Champs remplis
        if fields_analysis['filled_fields']:
            html_parts.append("""
                <div class="section-title">✅ Champs Remplis (avec valeurs)</div>
                <div class="fields-list">
            """)
            for field in fields_analysis['filled_fields']:
                value = chunk.get(field)
                value_str = str(value)[:100] + ("..." if len(str(value)) > 100 else "")
                html_parts.append(f"""
                    <div class="field-item">
                        <span class="field-name">{html.escape(field)}</span> = 
                        <span class="field-value">{html.escape(value_str)}</span>
                    </div>
                """)
            html_parts.append("""
                </div>
            """)
        
        # Champs vides
        if fields_analysis['empty_fields']:
            html_parts.append(f"""
                <div class="section-title">❌ Champs Vides ({len(fields_analysis['empty_fields'])})</div>
                <div class="fields-list">
            """)
            for field in fields_analysis['empty_fields']:
                html_parts.append(f"""
                    <div class="field-item">
                        <span class="field-name">{html.escape(field)}</span> = 
                        <span class="badge badge-empty">vide</span>
                    </div>
                """)
            html_parts.append("""
                </div>
            """)
        
        # Aperçu XML
        if show_xml and xml_preview:
            html_parts.append(f"""
                <div class="section-title">📄 Aperçu XML (50 premières lignes)</div>
                <div class="xml-content">{html.escape(xml_preview)}</div>
            """)
        
        html_parts.append("""
        </div>
        """)
    
    # Footer
    html_parts.append("""
    </body>
    </html>
    """)
    
    return "\n".join(html_parts)


# ============================================================================
# MAIN
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    paths = get_project_paths()
    
    parser = argparse.ArgumentParser(
        description="Inspection chunk XML avec sortie HTML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Par doc_key
  python tools/inspect_chunk_xml_html.py --corpus gold/chunks --doc-key LEGIARTI000006648153 --show-xml
  
  # Par index
  python tools/inspect_chunk_xml_html.py --corpus gold/chunks --index 0 --show-xml
  
  # Par corpus_juridique
  python tools/inspect_chunk_xml_html.py --corpus gold/chunks --corpus-juridique LEGITEXT000006072050 --limit 5
        """
    )

    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Chemin vers gold/chunks (Parquet). Par défaut : auto-détecté"
    )

    search_group = parser.add_mutually_exclusive_group(required=True)
    search_group.add_argument(
        "--doc-key",
        type=str,
        help="Rechercher chunk par doc_key"
    )
    search_group.add_argument(
        "--index",
        type=int,
        help="Récupérer chunk par index"
    )
    search_group.add_argument(
        "--corpus-juridique",
        type=str,
        help="Rechercher chunks par corpus_juridique"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Nombre max de chunks (pour --corpus-juridique)"
    )
    parser.add_argument(
        "--show-xml",
        action="store_true",
        help="Afficher aperçu XML"
    )
    parser.add_argument(
        "--xml-lines",
        type=int,
        default=50,
        help="Nombre de lignes XML"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de logging"
    )
    
    return parser.parse_args()


def main():
    """Point d'entrée principal."""
    args = parse_args()
    paths = get_project_paths()
    
    # Setup logging
    logger, log_path = setup_logging(paths.tools_test_logs_dir, args.log_level)
    
    logger.info("=" * 80)
    logger.info("🔍 DÉMARRAGE INSPECTION CHUNK XML (HTML)")
    logger.info("=" * 80)
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Afficher XML: {'Oui' if args.show_xml else 'Non'}")
    logger.info("")
    
    # Résolution chemin corpus (avec auto-détection)
    if args.corpus:
        # Chemin fourni par l'utilisateur
        corpus_path = resolve_path(args.corpus, repo_root=paths.repo_root)
        if not corpus_path:
            logger.error("❌ Chemin corpus invalide")
            sys.exit(1)
    else:
        # ✅ Auto-détection via paths.gold_corpus_dir
        corpus_path = str(paths.gold_corpus_dir)
        logger.info(f"✅ Corpus auto-détecté: {corpus_path}")
    
    # Vérifier que le corpus existe
    if not Path(corpus_path).exists():
        logger.error(f"❌ Corpus non trouvé: {corpus_path}")
        logger.error("💡 Solution: spécifiez --corpus /chemin/vers/gold/chunks")
        logger.error("💡 Ou définissez: export RAG_BENCH_GOLD_CORPUS=/chemin/vers/gold/chunks")
        sys.exit(1)
    
    # Chargement corpus
    try:
        chunks = load_chunks_safe(corpus_path, logger)
    except Exception as e:
        logger.error(f"❌ Impossible de charger le corpus: {e}")
        sys.exit(1)
    
    # Recherche chunk(s)
    chunks_to_inspect = []
    
    if args.doc_key:
        chunk = find_chunk_by_doc_key(chunks, args.doc_key, logger)
        if chunk:
            chunks_to_inspect = [chunk]
        else:
            logger.error(f"❌ Chunk non trouvé: {args.doc_key}")
            sys.exit(1)
    
    elif args.index is not None:
        chunk = find_chunk_by_index(chunks, args.index, logger)
        if chunk:
            chunks_to_inspect = [chunk]
        else:
            logger.error(f"❌ Index invalide: {args.index}")
            sys.exit(1)
    
    elif args.corpus_juridique:
        chunks_to_inspect = find_chunks_by_corpus_juridique(chunks, args.corpus_juridique, args.limit, logger)
        if not chunks_to_inspect:
            logger.error(f"❌ Aucun chunk trouvé pour corpus_juridique: {args.corpus_juridique}")
            sys.exit(1)
    
    # Préparer données pour HTML
    logger.info(f"📊 Préparation inspection de {len(chunks_to_inspect)} chunk(s)...")
    chunks_data = []
    
    for chunk in chunks_to_inspect:
        # Analyse champs
        fields_analysis = analyze_chunk_fields(chunk)
        
        # Lecture XML (si demandé)
        xml_preview = None
        if args.show_xml:
            source_path = chunk.get('source_path')
            if source_path:
                xml_preview = read_xml_preview(source_path, args.xml_lines, logger)
        
        chunks_data.append((chunk, fields_analysis, xml_preview))
    
    # Génération HTML
    logger.info("🎨 Génération HTML...")
    html_content = format_chunk_inspection_html(chunks_data, show_xml=args.show_xml)
    
    # Sauvegarde HTML
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_path = paths.tools_inspections_dir / f"inspect_chunk_xml_html{ts}.html"
    html_path.write_text(html_content, encoding='utf-8')
    logger.info(f"✅ HTML généré: {html_path}")
    
    # Résumé final
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ INSPECTION TERMINÉE AVEC SUCCÈS")
    logger.info("=" * 80)
    logger.info(f"Chunks inspectés: {len(chunks_to_inspect)}")
    logger.info(f"Logs: {log_path}")
    logger.info(f"HTML: {html_path}")
    logger.info("")
    
    # Lien pour ouvrir dans navigateur
    print(f"\n🌐 Ouvrir dans navigateur:")
    print(f"   file://{html_path.absolute()}")


if __name__ == "__main__":
    main()
