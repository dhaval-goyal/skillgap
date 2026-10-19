"""Module for ranking skills by mention frequency in job descriptions."""
from collections import Counter
from typing import Iterable, List, Dict, Any, Set
from normalization.skill_normalizer import normalize_skill
from gap.analyzer import _validate_skills


def rank_skills_by_frequency(jd_mentions: Iterable[str]) -> List[Dict[str, Any]]:
    """Rank JD skill mentions by frequency with alphabetical tie-breaking."""
    _validate_skills(jd_mentions, "jd_mentions")
    counts: Counter = Counter()
    display_names: Dict[str, str] = {}
    for mention in jd_mentions:
        canonical = normalize_skill(mention)
        if canonical:
            key = canonical.lower()
            counts[key] += 1
            if key not in display_names or (canonical[0].isupper() and not display_names[key][0].isupper()):
                display_names[key] = canonical

    sorted_items = sorted(counts.items(), key=lambda item: (-item[1], display_names[item[0]]))
    return [{"skill": display_names[key], "count": count} for key, count in sorted_items]


def rank_missing_skills(missing_skills: Iterable[str], jd_mentions: Iterable[str]) -> List[Dict[str, Any]]:
    """Prioritize missing skills based on their mention count in the JD."""
    _validate_skills(missing_skills, "missing_skills")
    _validate_skills(jd_mentions, "jd_mentions")

    norm_missing: Set[str] = {normalize_skill(s) for s in missing_skills if normalize_skill(s)}
    all_ranked = rank_skills_by_frequency(jd_mentions)
    ranked_map = {item["skill"].lower(): (item["skill"], item["count"]) for item in all_ranked}

    missing_dict = {}
    for skill in norm_missing:
        key = skill.lower()
        if key in ranked_map:
            jd_skill, count = ranked_map[key]
            missing_dict[jd_skill] = count
        else:
            missing_dict[skill] = 0

    sorted_missing = sorted(missing_dict.items(), key=lambda item: (-item[1], item[0]))
    return [{"skill": skill, "count": count} for skill, count in sorted_missing]
