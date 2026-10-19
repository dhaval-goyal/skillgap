"""Rule-based skill normalizer for canonicalizing extracted skill names."""

CANONICAL_SKILLS = {
    "powerbi": "Power BI",
    "power-bi": "Power BI",
    "power bi": "Power BI",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "python": "Python",
    "sql": "SQL",
}


def normalize_skill(skill: str) -> str:
    """Normalize a raw skill string to its canonical name or clean form."""
    if not isinstance(skill, str):
        raise TypeError(f"Skill must be a string, got {type(skill).__name__}")

    cleaned = " ".join(skill.strip().split())
    if not cleaned:
        return ""

    lookup_key = cleaned.lower()
    return CANONICAL_SKILLS.get(lookup_key, cleaned)
