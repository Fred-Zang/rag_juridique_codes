# -*- coding: utf-8 -*-
"""
Gestion centralisée des chemins du projet (portable Linux / WSL / Windows).

Objectif :
- éviter les chemins codés en dur,
- éviter de dépendre du dossier courant (CWD),
- supporter des overrides via variables d’environnement,
- homogénéiser la résolution de chemins entre run.py et les scripts de benchmark.

Principe :
- On définit une "racine projet" (project root) :
  - si RAG_BENCH_HOME est défini, on l’utilise,
  - sinon on remonte depuis ce fichier jusqu’à trouver un pyproject.toml,
  - sinon on fallback sur une hypothèse raisonnable de src-layout.

- On résout ensuite les chemins du YAML :
  - absolus : inchangés,
  - relatifs : ancrés sur la racine projet (repo_root),
  - chemins Windows "D:\..." sous Linux :
      - sous WSL : on tente de mapper vers /mnt/d/...
      - sinon : on laisse tel quel (ça échouera explicitement plutôt que créer un dossier "D:").
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _is_wsl() -> bool:
    """
    Détecte WSL de manière robuste (utile pour mapper D:\... vers /mnt/d/...).
    """
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _find_repo_root(start: Path) -> Optional[Path]:
    """
    Remonte dans l’arborescence à partir de `start` pour trouver un pyproject.toml.
    Retourne None si non trouvé.
    """
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def get_repo_root() -> Path:
    """
    Détermine la racine du projet.

    Priorité :
    1) RAG_BENCH_HOME (si défini)
    2) détection par présence de pyproject.toml en remontant depuis ce module
    3) fallback src-layout: .../<repo>/src/rag_bench/paths.py -> parents[2] = <repo>
    """
    env_home = os.getenv("RAG_BENCH_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()

    here = Path(__file__).resolve()
    detected = _find_repo_root(here)
    if detected is not None:
        return detected.resolve()

    return here.parents[2].resolve()


def _map_windows_drive_path_to_wsl(p: str) -> Optional[str]:
    """
    Mappe un chemin Windows de type D:\foo\bar ou D:/foo/bar vers /mnt/d/foo/bar (WSL).
    Retourne None si le mapping n’est pas possible.
    """
    if not _WINDOWS_DRIVE_RE.match(p):
        return None

    drive = p[0].lower()
    rest = p[2:].lstrip("\\/").replace("\\", "/")
    candidate = f"/mnt/{drive}/{rest}"
    if Path(candidate).exists():
        return str(Path(candidate).resolve())

    return None


def resolve_path(path_str: Optional[str], *, repo_root: Path, config_path: Optional[Path] = None) -> Optional[str]:
    """
    Résout un chemin provenant du YAML.

    Règles :
    - None / "" -> None
    - ~ et variables d’environnement -> expand
    - chemin absolu -> normalisé
    - chemin relatif -> ancré sur repo_root (portable)
    - chemin Windows "D:\..." sous Linux :
        - si WSL et /mnt/d/... existe -> on mappe
        - sinon -> on renvoie tel quel (échec explicite plus tard)
    """
    if not path_str or str(path_str).strip() == "":
        return None

    raw = os.path.expandvars(str(path_str))
    raw = str(Path(raw).expanduser())

    # Chemin Windows du type D:\... sous Linux/WSL
    if os.name != "nt" and _WINDOWS_DRIVE_RE.match(raw):
        if _is_wsl():
            mapped = _map_windows_drive_path_to_wsl(raw)
            if mapped is not None:
                return mapped
        return raw

    p = Path(raw)

    if p.is_absolute():
        return str(p.resolve())

    # Chemin relatif : ancré sur la racine projet
    return str((repo_root / p).resolve())


@dataclass(frozen=True)
class ProjectPaths:
    """
    Regroupe les répertoires standards du projet.

    Ces chemins ne remplacent pas la config YAML : ils servent à fournir des defaults
    et à centraliser la logique de base.
    """
    repo_root: Path
    configs_dir: Path
    data_dir: Path
    runs_dir: Path
    result_tests_dir: Path
    cache_dense_dir: Path
    tools_dir: Path
    tools_test_logs_dir: Path
    tools_inspections_dir: Path
    gold_corpus_dir: Path  # Nouveau : chemin vers gold/chunks

    def ensure_dirs(self) -> None:
        """
        Crée les répertoires attendus si nécessaire.
        """
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.result_tests_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dense_dir.mkdir(parents=True, exist_ok=True)
        self.tools_test_logs_dir.mkdir(parents=True, exist_ok=True)
        self.tools_inspections_dir.mkdir(parents=True, exist_ok=True)


def get_project_paths() -> ProjectPaths:
    """
    Construit les chemins standard du projet, avec overrides ENV possibles.

    Overrides :
    - RAG_BENCH_HOME : racine projet
    - RAG_BENCH_RUNS_DIR : override du dossier runs
    - RAG_BENCH_CACHE_DENSE_DIR : override du cache dense (embeddings)
    - RAG_BENCH_GOLD_CORPUS : override du corpus gold/chunks
    """
    repo_root = get_repo_root()

    runs_dir = Path(os.getenv("RAG_BENCH_RUNS_DIR", str(repo_root / "runs"))).expanduser().resolve()
    cache_dense_dir = Path(
        os.getenv("RAG_BENCH_CACHE_DENSE_DIR", str(repo_root / "result_tests" / "cache_dense"))
    ).expanduser().resolve()
    
    # Détection automatique du corpus gold (avec fallback)
    gold_corpus_env = os.getenv("RAG_BENCH_GOLD_CORPUS")
    if gold_corpus_env:
        gold_corpus_dir = Path(gold_corpus_env).expanduser().resolve()
    else:
        # Chercher dans plusieurs endroits possibles
        candidates = [
            repo_root.parent / "data_main" / "data" / "datalake_legifrance_v1" / "gold" / "chunks",  # Setup Fred
            repo_root / "data" / "gold" / "chunks",  # Fallback standard
        ]
        gold_corpus_dir = None
        for candidate in candidates:
            if candidate.exists():
                gold_corpus_dir = candidate.resolve()
                break
        
        # Si aucun trouvé, utiliser le fallback (même s'il n'existe pas)
        if gold_corpus_dir is None:
            gold_corpus_dir = (repo_root / "data" / "gold" / "chunks").resolve()

    paths = ProjectPaths(
        repo_root=repo_root,
        configs_dir=(repo_root / "configs").resolve(),
        data_dir=(repo_root / "data").resolve(),
        runs_dir=runs_dir,
        result_tests_dir=(repo_root / "result_tests").resolve(),
        cache_dense_dir=cache_dense_dir,
        tools_dir=(repo_root / "tools").resolve(),
        tools_test_logs_dir=(repo_root / "tools" / "test_logs").resolve(),
        tools_inspections_dir=(repo_root / "tools" / "inspections").resolve(),
        gold_corpus_dir=gold_corpus_dir,
    )
    paths.ensure_dirs()
    return paths