"""
SkillGap Pipeline: End-to-end integration module connecting skill extraction,
normalization, gap analysis, frequency ranking, curated resources, and optional assessment.

NOTE: This is an academic research prototype. It does not output hire/reject scores,
does not store raw resume text, and does not establish job readiness.
"""
from typing import List, Dict, Any, Optional, Callable
from normalization.skill_normalizer import normalize_skill
from gap.analyzer import compute_skill_gap
from gap.ranker import rank_missing_skills
from resources.resource_catalog import get_resources
from assessment.quiz_engine import calculate_score, compare_confidence
from inference import get_extractor


def extract_skills(text: str, model_path: Optional[str] = None) -> List[str]:
    """
    Extract skill mentions from text using fine-tuned DistilBERT token classifier.

    Loads the model once via get_extractor() and decodes predicted BIO token spans.
    Raises FileNotFoundError if local weights are not present.
    """
    extractor = get_extractor(model_path)
    return extractor.extract(text)


def analyze_job_match(
    job_description: str,
    resume_text: str,
    extract_fn: Optional[Callable[[str], List[str]]] = None,
) -> Dict[str, Any]:
    """
    Analyze the match between a job description and a resume.

    Steps:
    1. Extracts skill mentions from both texts.
    2. Normalizes skills to canonical names (handling aliases).
    3. Computes missing skills using set difference.
    4. Ranks missing skills by mention frequency in the job description.
    5. Retrieves curated educational resources for missing skills.
    6. Identifies missing skills that lack catalog resources.
    """
    if not isinstance(job_description, str) or not isinstance(resume_text, str):
        raise TypeError("job_description and resume_text must be strings.")

    clean_jd = job_description.strip()
    clean_resume = resume_text.strip()
    if not clean_jd or not clean_resume:
        raise ValueError("job_description and resume_text cannot be empty.")

    extractor = extract_fn if extract_fn is not None else extract_skills
    jd_mentions = extractor(clean_jd)
    resume_mentions = extractor(clean_resume)

    if not isinstance(jd_mentions, list) or not isinstance(resume_mentions, list):
        raise TypeError("Extraction function must return a list of skill strings.")

    norm_jd = {normalize_skill(s) for s in jd_mentions if normalize_skill(s)}
    norm_res = {normalize_skill(s) for s in resume_mentions if normalize_skill(s)}

    missing = compute_skill_gap(jd_mentions, resume_mentions)
    ranked_missing = rank_missing_skills(missing, jd_mentions)

    resources_map: Dict[str, List[Dict[str, str]]] = {}
    unmapped_resources: List[str] = []
    for skill in missing:
        res = get_resources(skill)
        resources_map[skill] = res
        if not res:
            unmapped_resources.append(skill)

    return {
        "job_skills_extracted": jd_mentions,
        "resume_skills_extracted": resume_mentions,
        "job_skills_normalized": sorted(norm_jd),
        "resume_skills_normalized": sorted(norm_res),
        "missing_skills": missing,
        "ranked_missing_skills": ranked_missing,
        "resources": resources_map,
        "unmapped_resources": sorted(unmapped_resources),
    }


def assess_skill(skill: str, answers: Dict[str, str], self_rating: int) -> Dict[str, Any]:
    """
    Evaluate quiz answers and confidence alignment for a supported skill.

    Optional step; does not modify gap calculations or assign hiring scores.
    """
    score_result = calculate_score(skill, answers)
    confidence_result = compare_confidence(self_rating, score_result["score_percent"])
    return {
        "skill": score_result["skill"],
        "score": score_result,
        "confidence_comparison": confidence_result,
    }
