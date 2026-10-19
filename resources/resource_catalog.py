"""
Manually curated learning resource catalog for key technical skills.

NOTE: This catalog is strictly manually curated and static.
It is NOT AI-personalized, dynamically retrieved, or experimentally validated.
"""
import copy
from typing import List, Dict
from normalization.skill_normalizer import normalize_skill

CURATED_RESOURCES: Dict[str, List[Dict[str, str]]] = {
    "Python": [
        {
            "title": "Official Python Tutorial",
            "url": "https://docs.python.org/3/tutorial/",
            "description": "Core Python tutorial covering syntax, data structures, and standard library.",
            "source_type": "Official Documentation",
        }
    ],
    "SQL": [
        {
            "title": "PostgreSQL Tutorial Documentation",
            "url": "https://www.postgresql.org/docs/current/tutorial.html",
            "description": "Relational database concepts, SQL queries, and table operations.",
            "source_type": "Official Documentation",
        }
    ],
    "Power BI": [
        {
            "title": "Microsoft Power BI Documentation",
            "url": "https://learn.microsoft.com/en-us/power-bi/",
            "description": "Guides for data visualization, DAX expressions, and dashboard design.",
            "source_type": "Official Documentation",
        }
    ],
    "Machine Learning": [
        {
            "title": "Google Machine Learning Crash Course",
            "url": "https://developers.google.com/machine-learning/crash-course",
            "description": "Foundational course covering ML concepts, model evaluation, and regression.",
            "source_type": "Learning Course",
        }
    ],
    "scikit-learn": [
        {
            "title": "scikit-learn User Guide",
            "url": "https://scikit-learn.org/stable/user_guide.html",
            "description": "Comprehensive reference for machine learning algorithms and pipelines in Python.",
            "source_type": "Official Documentation",
        }
    ],
    "Docker": [
        {
            "title": "Docker Documentation - Get Started",
            "url": "https://docs.docker.com/get-started/",
            "description": "Guide covering containers, Dockerfiles, images, and basic networking.",
            "source_type": "Official Documentation",
        }
    ],
}


def get_resources(skill: str) -> List[Dict[str, str]]:
    """
    Retrieve manually curated learning resources for a given skill.

    Normalizes the skill name to match known aliases (e.g., 'powerbi' -> 'Power BI').
    Returns an empty list for unknown skills. Returns copies to prevent caller mutation.
    """
    if not isinstance(skill, str):
        raise TypeError(f"Skill must be a string, got {type(skill).__name__}")

    canonical = normalize_skill(skill)
    if not canonical:
        return []

    for key, resources in CURATED_RESOURCES.items():
        if key.lower() == canonical.lower():
            return copy.deepcopy(resources)

    return []
