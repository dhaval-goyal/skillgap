"""Skill gap analysis and ranking package."""
from gap.analyzer import compute_skill_gap
from gap.ranker import rank_skills_by_frequency, rank_missing_skills

__all__ = [
    "compute_skill_gap",
    "rank_skills_by_frequency",
    "rank_missing_skills",
]
