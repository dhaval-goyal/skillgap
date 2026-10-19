"""Skill gap computation module for comparing required vs resume skills."""
from typing import Iterable, List, Set
from normalization.skill_normalizer import normalize_skill


def _validate_skills(skills: Iterable[str], name: str) -> None:
    """Validate that input is an iterable collection of strings (not a string)."""
    if isinstance(skills, (str, bytes)) or not isinstance(skills, Iterable):
        raise TypeError(f"'{name}' must be an iterable collection of strings, got {type(skills).__name__}")
    for item in skills:
        if not isinstance(item, str):
            raise TypeError(f"All items in '{name}' must be strings, found {type(item).__name__}")


def compute_skill_gap(required_skills: Iterable[str], resume_skills: Iterable[str]) -> List[str]:
    """
    Compute missing skills by comparing job required skills against resume skills.

    Normalizes both sets to canonical names, eliminates aliases and duplicates,
    and returns a deterministically sorted list of missing skills.

    Note: Extracted mentions indicate candidate mention in resume, not
    independently verified competence.

    Example:
        >>> compute_skill_gap(["Python", "SQL", "Power BI", "Docker"], ["python", "SQL", "powerbi"])
        ['Docker']
    """
    _validate_skills(required_skills, "required_skills")
    _validate_skills(resume_skills, "resume_skills")

    norm_required: Set[str] = {normalize_skill(s) for s in required_skills}
    norm_resume: Set[str] = {normalize_skill(s) for s in resume_skills}

    norm_required.discard("")
    norm_resume.discard("")

    resume_lower = {s.lower() for s in norm_resume}
    missing = [s for s in norm_required if s.lower() not in resume_lower]
    return sorted(missing)
