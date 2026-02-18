#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_corpus_codes.py - VERSION 2 avec auto-détection
Phase 2 - Analyse complète des codes juridiques dans le corpus

Objectif:
- Compter chunks par corpus_juridique (code juridique)
- Analyser répartition par status, temporalité, article_num
- **CRITIQUE** : Vérifier si LEGITEXT000006072050 = tout le Code du Travail
- Auto-détection du corpus via paths.gold_corpus_dir
- Affichage des noms de codes ("Code du travail", etc.)

Usage SIMPLIFIÉ (auto-détection) :
    # Top-20 codes (corpus auto-détecté)
    python tools/analyze_corpus_codes.py --top-n 20
    
    # Détail Code du Travail
    python tools/analyze_corpus_codes.py --detail-code LEGITEXT000006072050
    
    # Export JSON
    python tools/analyze_corpus_codes.py --top-n 50 --output codes_analysis.json

Usage classique (avec --corpus) :
    # Spécifier corpus manuellement
    python tools/analyze_corpus_codes.py --corpus /chemin/vers/gold/chunks --top-n 20

Logs: tools/test_logs/analyze_corpus_codes_<timestamp>.md
Résultats: tools/inspections/analyze_corpus_codes_<timestamp>.html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Racine du projet = dossier qui contient "src" et "tools"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from rag_bench.paths import get_project_paths, resolve_path
from rag_bench.io_parquet import load_chunks_from_parquet

# ⭐ V3 : code_titre est maintenant dans les chunks, plus besoin de code_names_generated


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
    md_log_path = log_dir / f"analyze_corpus_codes_{ts}.md"

    # En-tête Markdown
    header = [
        f"# 📊 Logs Analyse Corpus Codes",
        "",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Script**: `tools/analyze_corpus_codes.py`",
        f"- **Logs**: `{md_log_path}`",
        "",
        "---",
        "",
    ]
    md_log_path.write_text("\n".join(header), encoding="utf-8")

    logger = logging.getLogger("analyze_corpus_codes")
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
# ANALYSE CORPUS
# ============================================================================

def analyze_codes(chunks: List[Dict[str, Any]], logger: logging.Logger) -> Dict[str, Any]:
    """
    Analyse complète des codes juridiques dans le corpus.
    
    Returns:
        {
            "total_chunks": int,
            "codes": {
                "LEGITEXT...": {
                    "count": int,
                    "percentage": float,
                    "status_distribution": {...},
                    "article_num_samples": [...],
                    "article_prefixes": {...},  # L, R, D, A
                    "temporal_ranges": {...}
                }
            }
        }
    """
    logger.info("🔍 Démarrage analyse codes...")
    
    total_chunks = len(chunks)
    logger.info(f"Total chunks à analyser: {total_chunks:,}")
    
    # Comptage par corpus_juridique
    code_counts = Counter()
    code_details = defaultdict(lambda: {
        "chunks": [],
        "status": Counter(),
        "article_nums": [],
        "doc_types": Counter(),
        "years": [],
        "code_titre": None  # ⭐ V3 : nom du code directement depuis les chunks
    })

    chunks_without_corpus_juridique = 0

    for chunk in chunks:
        corpus_juridique = chunk.get("corpus_juridique")

        if not corpus_juridique or corpus_juridique == "None" or corpus_juridique.strip() == "":
            chunks_without_corpus_juridique += 1
            continue

        code_counts[corpus_juridique] += 1

        # Détails pour ce code
        details = code_details[corpus_juridique]
        details["chunks"].append(chunk)

        # ⭐ V3 : récupérer code_titre (prendre le premier trouvé)
        if details["code_titre"] is None:
            code_titre = chunk.get("code_titre")
            if code_titre and code_titre != "None" and code_titre.strip():
                details["code_titre"] = code_titre

        # Status
        status = chunk.get("status", "")
        if status:
            details["status"][status] += 1
        
        # Article num
        article_num = chunk.get("article_num")
        if article_num and article_num != "None":
            details["article_nums"].append(article_num)
        
        # Doc type
        doc_type = chunk.get("doc_type")
        if doc_type:
            details["doc_types"][doc_type] += 1
        
        # Années (pour temporalité)
        valid_from = chunk.get("valid_from")
        if valid_from and valid_from != "None":
            try:
                year = int(valid_from[:4])
                details["years"].append(year)
            except:
                pass
    
    logger.info(f"✅ Analyse terminée")
    logger.info(f"Codes uniques trouvés: {len(code_counts)}")
    logger.info(f"Chunks sans corpus_juridique: {chunks_without_corpus_juridique:,} ({chunks_without_corpus_juridique/total_chunks*100:.1f}%)")

    # Construire résultats
    results = {
        "total_chunks": total_chunks,
        "unique_codes": len(code_counts),
        "chunks_without_corpus_juridique": chunks_without_corpus_juridique,
        "codes": {}
    }

    for corpus_juridique, count in code_counts.items():
        details = code_details[corpus_juridique]
        
        # Répartition status
        status_dist = dict(details["status"])
        
        # Échantillon article_nums (10 premiers)
        article_nums_sample = details["article_nums"][:10]
        
        # Compter préfixes articles (L, R, D, A)
        article_prefixes = Counter()
        for num in details["article_nums"]:
            if num and len(num) > 0:
                prefix = num[0].upper()
                if prefix in ['L', 'R', 'D', 'A']:
                    article_prefixes[prefix] += 1
        
        # Répartition temporelle
        years = details["years"]
        temporal_ranges = {}
        if years:
            temporal_ranges = {
                "min_year": min(years),
                "max_year": max(years),
                "avg_year": sum(years) / len(years)
            }
        
        # Répartition doc_types
        doc_types_dist = dict(details["doc_types"])
        
        results["codes"][corpus_juridique] = {
            "count": count,
            "percentage": (count / total_chunks) * 100,
            "status_distribution": status_dist,
            "article_num_samples": article_nums_sample,
            "article_prefixes": dict(article_prefixes),
            "temporal_ranges": temporal_ranges,
            "doc_types": doc_types_dist,
            "code_titre": details["code_titre"]  # ⭐ V3 : nom du code
        }
    
    return results


def format_analysis_html(
    results: Dict[str, Any],
    detail_code: Optional[str] = None,
    top_n: Optional[int] = 20
) -> str:
    """
    Formate les résultats d'analyse en HTML (style analyze_results.py).
    
    Args:
        results: Résultats analyse
        detail_code: Code à détailler (ex: LEGITEXT000006072050)
        top_n: Nombre de codes à afficher (None = tout afficher)
    
    Returns:
        HTML complet
    """
    html_parts = []
    
    # Header
    html_parts.append(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Analyse Corpus - Codes Juridiques</title>
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

            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }}

            .stat-card {{
                background: var(--panel);
                padding: 20px;
                border-radius: 12px;
                border: 1px solid var(--border);
                text-align: center;
            }}

            .stat-value {{
                font-size: 32px;
                font-weight: 800;
                color: var(--accent);
                margin-bottom: 5px;
            }}

            .stat-label {{
                font-size: 14px;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}

            .section {{
                background: var(--panel);
                padding: 25px;
                margin-bottom: 30px;
                border-radius: 12px;
                border: 1px solid var(--border);
                box-shadow: 0 8px 26px var(--shadow);
            }}

            .section-title {{
                font-size: 24px;
                color: var(--accent);
                margin-bottom: 20px;
                border-bottom: 3px solid rgba(139,147,255,0.75);
                padding-bottom: 10px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}

            th {{
                background: var(--panel-3);
                color: var(--accent);
                padding: 12px;
                text-align: left;
                border-bottom: 2px solid var(--border);
                font-weight: 700;
                text-transform: uppercase;
                font-size: 12px;
                letter-spacing: 0.05em;
            }}

            td {{
                padding: 12px;
                border-bottom: 1px solid var(--border);
                color: var(--text);
            }}

            tr:hover {{
                background: rgba(139,147,255,0.05);
            }}

            .code-name {{
                font-family: 'Courier New', monospace;
                color: var(--accent);
                font-weight: 600;
            }}

            .highlight-row {{
                background: rgba(56,208,112,0.08);
                border-left: 4px solid var(--ok);
            }}

            .badge {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 11px;
                font-weight: 700;
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

            .detail-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}

            .detail-card {{
                background: var(--panel-2);
                padding: 15px;
                border-radius: 10px;
                border: 1px solid var(--border);
            }}

            .detail-label {{
                font-size: 11px;
                color: var(--muted);
                text-transform: uppercase;
                font-weight: 700;
                margin-bottom: 8px;
            }}

            .detail-value {{
                font-size: 18px;
                color: var(--text);
                font-family: 'Courier New', monospace;
            }}

            .warning-box {{
                background: rgba(255,204,102,0.15);
                color: var(--warn);
                padding: 15px;
                border-radius: 10px;
                border: 1px solid rgba(255,204,102,0.3);
                margin: 20px 0;
            }}

            .ok-box {{
                background: rgba(56,208,112,0.15);
                color: var(--ok);
                padding: 15px;
                border-radius: 10px;
                border: 1px solid rgba(56,208,112,0.3);
                margin: 20px 0;
            }}
            
            .inspect-btn {{
                background: var(--accent);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.2s ease;
                font-family: 'Segoe UI', sans-serif;
            }}
            
            .inspect-btn:hover {{
                background: var(--accent-2);
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(139,147,255,0.3);
            }}
            
            .inspect-btn:active {{
                transform: translateY(0);
            }}
            
            .copy-notification {{
                position: fixed;
                top: 20px;
                right: 20px;
                background: var(--ok);
                color: white;
                padding: 15px 20px;
                border-radius: 8px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.3);
                font-weight: 700;
                z-index: 9999;
                animation: slideIn 0.3s ease;
            }}
            
            @keyframes slideIn {{
                from {{
                    transform: translateX(400px);
                    opacity: 0;
                }}
                to {{
                    transform: translateX(0);
                    opacity: 1;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Analyse Corpus - Codes Juridiques</h1>
            <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """)
    
    # Stats globales
    html_parts.append(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{results['total_chunks']:,}</div>
                <div class="stat-label">Total Chunks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{results['unique_codes']}</div>
                <div class="stat-label">Codes Uniques</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{results['chunks_without_corpus_juridique']:,}</div>
                <div class="stat-label">Sans corpus_juridique</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{results['chunks_without_corpus_juridique']/results['total_chunks']*100:.1f}%</div>
                <div class="stat-label">% Sans corpus_juridique</div>
            </div>
        </div>
    """)
    
    # Top codes
    html_parts.append(f"""
        <div class="section">
            <h2 class="section-title">🏆 Codes Juridiques</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rang</th>
                        <th>Corpus Juridique</th>
                        <th>Chunks</th>
                        <th>% Total</th>
                        <th>Status</th>
                        <th>Préfixes Articles</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
    """)
    
    # Trier codes par count
    sorted_codes = sorted(
        results['codes'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    
    # Déterminer combien de codes afficher (None => tout afficher)
    display_n = len(sorted_codes) if top_n is None else max(0, top_n)

    for rank, (corpus_juridique, code_data) in enumerate(sorted_codes[:display_n], 1):
        count = code_data['count']
        percentage = code_data['percentage']
        status_dist = code_data.get('status_distribution', {})
        prefixes = code_data.get('article_prefixes', {})

        # Status badges
        status_badges = []
        for status, count_status in sorted(status_dist.items(), key=lambda x: x[1], reverse=True)[:3]:
            status_badges.append(f'<span class="badge badge-ok">{status}: {count_status}</span>')
        status_html = ' '.join(status_badges) if status_badges else "N/A"

        # Préfixes articles
        prefixes_str = ", ".join([f"{k}: {v}" for k, v in prefixes.items()]) if prefixes else "N/A"

        # Highlight si Code du Travail
        row_class = 'highlight-row' if corpus_juridique == 'LEGITEXT000006072050' else ''

        # ⭐ V3 : Nom du code directement depuis les données (ex: "Code du travail")
        code_display_name = code_data.get('code_titre') or corpus_juridique

        # Commande inspect (échapper les quotes)
        inspect_cmd = f"python tools/inspect_chunk_xml_html.py --corpus-juridique {corpus_juridique} --limit 5 --show-xml"
        inspect_cmd_escaped = inspect_cmd.replace('"', '&quot;')

        html_parts.append(f"""
                    <tr class="{row_class}">
                        <td><strong>#{rank}</strong></td>
                        <td>
                            <span class="code-name">{corpus_juridique}</span>
                            <br><small style="color: var(--muted); font-size: 11px;">{code_display_name}</small>
                        </td>
                        <td><strong>{count:,}</strong></td>
                        <td>{percentage:.2f}%</td>
                        <td>{status_html}</td>
                        <td>{prefixes_str}</td>
                        <td>
                            <button class="inspect-btn" onclick="copyToClipboard('{inspect_cmd_escaped}')" title="Copier la commande">
                                🔍 Inspecter
                            </button>
                        </td>
                    </tr>
        """)
    
    html_parts.append("""
                </tbody>
            </table>
        </div>
    """)
    
    # Détail code spécifique (si demandé)
    if detail_code and detail_code in results['codes']:
        code_data = results['codes'][detail_code]
        
        html_parts.append(f"""
            <div class="section">
                <h2 class="section-title">🔍 Détail Code: {detail_code}</h2>
        """)
        
        # Warning ou OK selon préfixes
        prefixes = code_data.get('article_prefixes', {})
        has_L = 'L' in prefixes
        has_R = 'R' in prefixes
        has_D = 'D' in prefixes
        has_A = 'A' in prefixes
        
        if has_L and not (has_R or has_D or has_A):
            html_parts.append(f"""
                <div class="warning-box">
                    <strong>⚠️ ATTENTION</strong> : Ce code contient <strong>seulement</strong> des articles L (législatifs).
                    <br>Les parties réglementaires (R, D, A) sont <strong>absentes</strong> !
                    <br>➡️ Le corpus pourrait être <strong>incomplet</strong>.
                </div>
            """)
        elif has_L and (has_R or has_D or has_A):
            html_parts.append(f"""
                <div class="ok-box">
                    <strong>✅ OK</strong> : Ce code contient plusieurs types d'articles (L, R, D, A).
                    <br>Le corpus semble <strong>complet</strong>.
                </div>
            """)
        
        # Grille détails
        html_parts.append(f"""
                <div class="detail-grid">
                    <div class="detail-card">
                        <div class="detail-label">Total Chunks</div>
                        <div class="detail-value">{code_data['count']:,}</div>
                    </div>
                    <div class="detail-card">
                        <div class="detail-label">% Corpus</div>
                        <div class="detail-value">{code_data['percentage']:.2f}%</div>
                    </div>
        """)
        
        # Préfixes articles détaillés
        for prefix in ['L', 'R', 'D', 'A']:
            count_prefix = prefixes.get(prefix, 0)
            badge_class = 'badge-ok' if count_prefix > 0 else 'badge-warn'
            html_parts.append(f"""
                    <div class="detail-card">
                        <div class="detail-label">Articles {prefix}</div>
                        <div class="detail-value">
                            {count_prefix:,} <span class="badge {badge_class}">{prefix}</span>
                        </div>
                    </div>
            """)
        
        # Status distribution
        status_dist = code_data.get('status_distribution', {})
        for status, count_status in sorted(status_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
            html_parts.append(f"""
                    <div class="detail-card">
                        <div class="detail-label">Status: {status}</div>
                        <div class="detail-value">{count_status:,}</div>
                    </div>
            """)
        
        # Temporalité
        temporal = code_data.get('temporal_ranges', {})
        if temporal:
            html_parts.append(f"""
                    <div class="detail-card">
                        <div class="detail-label">Période</div>
                        <div class="detail-value">{temporal.get('min_year', 'N/A')} - {temporal.get('max_year', 'N/A')}</div>
                    </div>
            """)
        
        html_parts.append("""
                </div>
        """)
        
        # Échantillon article_nums
        article_nums = code_data.get('article_num_samples', [])
        if article_nums:
            html_parts.append(f"""
                <h3 style="color: var(--accent); margin-top: 30px;">📝 Échantillon Articles (10 premiers)</h3>
                <div style="background: var(--panel-2); padding: 15px; border-radius: 10px; border: 1px solid var(--border); font-family: 'Courier New', monospace;">
                    {', '.join(article_nums)}
                </div>
            """)
        
        html_parts.append("""
            </div>
        """)
    
    # Footer
    html_parts.append("""
    <script>
        function copyToClipboard(text) {
            // Copier dans le presse-papiers
            navigator.clipboard.writeText(text).then(function() {
                // Afficher notification
                const notification = document.createElement('div');
                notification.className = 'copy-notification';
                notification.textContent = '✅ Commande copiée !';
                document.body.appendChild(notification);
                
                // Supprimer après 2 secondes
                setTimeout(() => {
                    notification.remove();
                }, 2000);
            }).catch(function(err) {
                // Fallback : afficher la commande dans une alerte
                alert('Commande à copier :\\n\\n' + text);
            });
        }
    </script>
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
        description="Analyse complète des codes juridiques dans le corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Top-20 codes
  python tools/analyze_corpus_codes.py --corpus gold/chunks --top-n 20
  
  # Détail Code du Travail
  python tools/analyze_corpus_codes.py --corpus gold/chunks --detail-code LEGITEXT000006072050
  
  # Export JSON
  python tools/analyze_corpus_codes.py --corpus gold/chunks --top-n 50 --output codes_analysis.json
        """
    )
    
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,  # ✅ CHANGÉ : auto-détection via paths.gold_corpus_dir
        help="Chemin vers gold/chunks (Parquet). Par défaut : auto-détecté"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,  # 20 pour top-20 ou bien None pour tout avoir (tous les codes juridiques existants)
        help="Nombre de codes à afficher (défaut: tout)"
    )
    parser.add_argument(
        "--detail-code",
        type=str,
        help="Code à détailler (ex: LEGITEXT000006072050)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Fichier JSON de sortie (optionnel)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de logging (défaut: INFO)"
    )
    
    return parser.parse_args()


def main():
    """Point d'entrée principal."""
    args = parse_args()
    paths = get_project_paths()
    
    # Setup logging
    logger, log_path = setup_logging(paths.tools_test_logs_dir, args.log_level)
    
    logger.info("=" * 80)
    logger.info("📊 DÉMARRAGE ANALYSE CORPUS CODES")
    logger.info("=" * 80)
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Top-N: {args.top_n}")
    if args.detail_code:
        logger.info(f"Détail code: {args.detail_code}")
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
    
    # Chargement corpus (format plat)
    try:
        logger.info(f"📦 Chargement corpus: {corpus_path}")
        chunks = load_chunks_from_parquet(
            parquet_path=corpus_path,
            flatten=True  # Format plat
        )
        logger.info(f"✅ Corpus chargé: {len(chunks):,} chunks")
    except Exception as e:
        logger.error(f"❌ Impossible de charger le corpus: {e}")
        sys.exit(1)
    
    # Analyse
    try:
        results = analyze_codes(chunks, logger)
    except Exception as e:
        logger.error(f"❌ Erreur analyse: {e}")
        sys.exit(1)
    
    # Export JSON (si demandé)
    if args.output:
        output_path = Path(args.output)
        logger.info(f"💾 Export JSON: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ JSON exporté: {output_path}")
    
    # Génération HTML
    logger.info("🎨 Génération HTML...")
    html_content = format_analysis_html(results, detail_code=args.detail_code, top_n=args.top_n)
    
    # Sauvegarde HTML
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_path = paths.tools_inspections_dir / f"analyze_corpus_codes_{ts}.html"
    html_path.write_text(html_content, encoding='utf-8')
    logger.info(f"✅ HTML généré: {html_path}")
    
    # Résumé final
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ ANALYSE TERMINÉE AVEC SUCCÈS")
    logger.info("=" * 80)
    logger.info(f"Total chunks: {results['total_chunks']:,}")
    logger.info(f"Codes uniques: {results['unique_codes']}")
    logger.info(f"Logs: {log_path}")
    logger.info(f"HTML: {html_path}")
    if args.output:
        logger.info(f"JSON: {args.output}")
    logger.info("")
    
    # Afficher lien HTML pour ouverture
    print(f"\n🌐 Ouvrir dans navigateur:")
    print(f"   file://{html_path.absolute()}")


if __name__ == "__main__":
    main()
