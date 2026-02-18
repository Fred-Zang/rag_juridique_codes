"""
Chargement de configuration runtime depuis un YAML.

Objectifs:
- YAML = configuration par défaut (reproductible, versionnée)
- ENV = overrides (prod/CI/container)
- On garde la compatibilité avec l'existant: paths.py lit les ENV.

Nouveau dans cette version:
- On retourne le dict YAML pour que l'API puisse résoudre `corpus_juridique -> parquet_path`
  au moment des requêtes (sélection de corpus sans redémarrer).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Charge un YAML en dict Python."""
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML est requis pour charger runtime_online.yaml. "
            "Installe-le via: pip install pyyaml"
        ) from e

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Le YAML runtime doit être un mapping (dict) à la racine.")
    return data


def _setenv_if_missing(key: str, value: str) -> None:
    """
    Définir une ENV seulement si elle n'existe pas déjà.

    Règle:
    - YAML = valeur par défaut
    - ENV existante = override prioritaire
    """
    if os.getenv(key) is None:
        os.environ[key] = value


def load_runtime_online_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Charge le YAML runtime et retourne le dict.

    Cette fonction ne fait PAS de side-effects, elle sert juste à obtenir `cfg`.
    Les side-effects (ENV) sont dans `apply_runtime_online_yaml`.
    """
    path = Path(yaml_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"runtime_online.yaml introuvable: {path}")
    return _load_yaml(path)


def apply_runtime_online_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Applique un runtime_online.yaml en:
    1) chargeant la config YAML
    2) injectant des variables d'environnement utiles (compat avec paths.py et runtime existant)
    3) retournant le dict YAML (pour usage runtime: registry corpora)

    Variables gérées:
    - RAG_BENCH_CACHE_DENSE_DIR  (dossier cache embeddings unique)
    - RAG_DEFAULT_RETRIEVER      (défaut retriever)
    - RAG_DEFAULT_K             (défaut k)
    - RAG_CORPUS_PATH           (parquet par défaut, basé sur default_corpus_juridique)
    """
    cfg = load_runtime_online_yaml(yaml_path)

    # 1) Cache dense unique: YAML -> ENV pour que paths.py lise la même source de vérité
    cache_dense_dir = (cfg.get("paths", {}) or {}).get("cache_dense_dir")
    if isinstance(cache_dense_dir, str) and cache_dense_dir.strip():
        _setenv_if_missing(
            "RAG_BENCH_CACHE_DENSE_DIR",
            str(Path(cache_dense_dir).expanduser().resolve()),
        )

    # 2) Defaults retrieval: YAML -> ENV pour cohérence runtime
    retrieval = cfg.get("retrieval", {}) or {}

    default_retriever = retrieval.get("default_retriever_type")
    if isinstance(default_retriever, str) and default_retriever.strip():
        _setenv_if_missing("RAG_DEFAULT_RETRIEVER", default_retriever)

    default_k = retrieval.get("default_k")
    if isinstance(default_k, int) and default_k > 0:
        _setenv_if_missing("RAG_DEFAULT_K", str(default_k))

    # 3) Sélection du parquet par défaut via `runtime.default_corpus_juridique`
    #    (clé = LEGITEXT...) et registry `corpora:`
    runtime = cfg.get("runtime", {}) or {}
    default_cj = runtime.get("default_corpus_juridique")

    corpora = cfg.get("corpora", {}) or {}
    if isinstance(default_cj, str) and default_cj in corpora:
        entry = corpora.get(default_cj) or {}
        parquet_path = entry.get("parquet_path")
        if isinstance(parquet_path, str) and parquet_path.strip():
            _setenv_if_missing("RAG_CORPUS_PATH", str(Path(parquet_path).expanduser().resolve()))

    return cfg
