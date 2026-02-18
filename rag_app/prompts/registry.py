"""
Registre centralisé des prompts pour le RAG juridique.

Permet de gérer les versions de prompts et leur sélection.

Usage:
    from rag_app.prompts import get_prompt, list_prompts

    # Obtenir le prompt par défaut
    prompt = get_prompt("rag_qa")

    # Obtenir une version spécifique
    prompt = get_prompt("rag_qa", version="v3")

    # Lister les prompts disponibles
    print(list_prompts())
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from langchain_core.prompts import BasePromptTemplate, PromptTemplate

from rag_app.prompts.templates import PROMPT_TEMPLATES, CHAT_PROMPT_TEMPLATES
from rag_bench.logging_utils import setup_logging

logger = setup_logging()


class PromptRegistry:
    """
    Registre centralisé des prompts avec versionnage.

    Permet de :
    - Récupérer des prompts par nom et version
    - Enregistrer de nouveaux prompts dynamiquement
    - Lister les prompts disponibles

    Example:
        >>> registry = PromptRegistry()
        >>> prompt = registry.get("rag_qa", version="v2")
        >>> print(prompt.input_variables)
        ['context', 'question']
    """

    def __init__(self):
        """Initialise le registre avec les prompts par défaut."""
        self._prompts: Dict[str, Dict] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Charge les prompts par défaut depuis templates.py."""
        for name, versions in PROMPT_TEMPLATES.items():
            self._prompts[name] = versions.copy()

        for name, versions in CHAT_PROMPT_TEMPLATES.items():
            self._prompts[name] = versions.copy()

    def get(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> BasePromptTemplate:
        """
        Récupère un prompt par son nom et version.

        Args:
            name: Nom du prompt (ex: "rag_qa")
            version: Version souhaitée (ex: "v2") ou None pour default

        Returns:
            PromptTemplate ou ChatPromptTemplate

        Raises:
            KeyError: Si le prompt ou la version n'existe pas
        """
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' non trouvé. Disponibles: {list(self._prompts.keys())}")

        versions = self._prompts[name]

        if version is None:
            version = versions.get("default", "v1")

        if version not in versions:
            available = [k for k in versions.keys() if k != "default"]
            raise KeyError(f"Version '{version}' non trouvée pour '{name}'. Disponibles: {available}")

        return versions[version]

    def register(
        self,
        name: str,
        version: str,
        template: Union[str, BasePromptTemplate],
        set_default: bool = False,
    ) -> None:
        """
        Enregistre un nouveau prompt ou une nouvelle version.

        Args:
            name: Nom du prompt
            version: Version (ex: "v4", "custom")
            template: Template string ou PromptTemplate
            set_default: Définir comme version par défaut
        """
        if name not in self._prompts:
            self._prompts[name] = {"default": version}

        if isinstance(template, str):
            template = PromptTemplate.from_template(template)

        self._prompts[name][version] = template

        if set_default:
            self._prompts[name]["default"] = version

        logger.info("Prompt enregistré: %s/%s", name, version)

    def list_prompts(self) -> Dict[str, List[str]]:
        """
        Liste tous les prompts et leurs versions.

        Returns:
            Dict {nom: [versions]}
        """
        result = {}
        for name, versions in self._prompts.items():
            result[name] = [k for k in versions.keys() if k != "default"]
        return result

    def get_default_version(self, name: str) -> str:
        """Retourne la version par défaut d'un prompt."""
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' non trouvé")
        return self._prompts[name].get("default", "v1")


# Instance globale du registre
_registry = PromptRegistry()


def get_prompt(name: str, version: Optional[str] = None) -> BasePromptTemplate:
    """
    Raccourci pour récupérer un prompt du registre global.

    Args:
        name: Nom du prompt
        version: Version (None = default)

    Returns:
        PromptTemplate
    """
    return _registry.get(name, version)


def list_prompts() -> Dict[str, List[str]]:
    """Raccourci pour lister les prompts disponibles."""
    return _registry.list_prompts()


def register_prompt(
    name: str,
    version: str,
    template: Union[str, BasePromptTemplate],
    set_default: bool = False,
) -> None:
    """Raccourci pour enregistrer un prompt."""
    _registry.register(name, version, template, set_default)
