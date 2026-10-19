"""
Quiz assessment engine for scoring and self-rating confidence comparison.

NOTE: This is an educational prototype for self-reflection.
It does not establish or measure professional competence or job readiness.
"""
from typing import List, Dict, Any
from normalization.skill_normalizer import normalize_skill
from assessment.quiz_data import QUESTIONS

SUPPORTED_QUIZ_SKILLS = {"Python", "SQL", "Power BI"}


def _get_canonical_skill(skill: str) -> str:
    """Validate skill input and resolve to supported canonical name."""
    if not isinstance(skill, str):
        raise TypeError(f"Skill must be a string, got {type(skill).__name__}")
    canonical = normalize_skill(skill)
    if canonical not in SUPPORTED_QUIZ_SKILLS:
        raise ValueError(f"Unsupported quiz skill: '{skill}'. Supported: {sorted(SUPPORTED_QUIZ_SKILLS)}")
    return canonical


def get_questions(skill: str) -> List[Dict[str, Any]]:
    """
    Return student-facing questions for a supported skill.

    Excludes correct answers and explanations to prevent leaking the answer key.
    """
    canonical = _get_canonical_skill(skill)
    return [
        {
            "id": q["id"],
            "skill": q["skill"],
            "topic": q["topic"],
            "question": q["question"],
            "options": list(q["options"]),
        }
        for q in QUESTIONS
        if q["skill"] == canonical
    ]


def calculate_score(skill: str, answers: Dict[str, str]) -> Dict[str, Any]:
    """
    Score submitted answers for a skill. Unanswered questions receive 0 points.

    Validates that question IDs belong to the skill and answers match valid options.
    """
    canonical = _get_canonical_skill(skill)
    if not isinstance(answers, dict):
        raise TypeError(f"answers must be a dictionary, got {type(answers).__name__}")

    skill_questions = {q["id"]: q for q in QUESTIONS if q["skill"] == canonical}
    for qid, ans in answers.items():
        if qid not in skill_questions:
            raise ValueError(f"Unknown question ID '{qid}' for skill '{canonical}'")
        if not isinstance(ans, str) or ans not in skill_questions[qid]["options"]:
            raise ValueError(f"Invalid answer '{ans}' for question '{qid}'")

    correct_count = sum(1 for qid, ans in answers.items() if ans == skill_questions[qid]["correct_answer"])
    total_q = len(skill_questions)
    attempted_q = len(answers)
    score_pct = round((correct_count / total_q) * 100.0, 2) if total_q > 0 else 0.0
    attempted_pct = round((correct_count / attempted_q) * 100.0, 2) if attempted_q > 0 else 0.0

    return {
        "skill": canonical,
        "correct_count": correct_count,
        "attempted_count": attempted_q,
        "total_questions": total_q,
        "score_percent": score_pct,
        "attempted_percent": attempted_pct,
    }


def calculate_overall_score(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall aggregate score across completed skill quiz results."""
    if not isinstance(results, list):
        raise TypeError(f"results must be a list of result dictionaries, got {type(results).__name__}")
    total_correct = sum(r["correct_count"] for r in results)
    total_attempted = sum(r["attempted_count"] for r in results)
    total_questions = sum(r["total_questions"] for r in results)
    overall_pct = round((total_correct / total_questions) * 100.0, 2) if total_questions > 0 else 0.0

    return {
        "total_correct": total_correct,
        "total_attempted": total_attempted,
        "total_questions": total_questions,
        "overall_score_percent": overall_pct,
    }


def compare_confidence(rating: int, quiz_percent: float) -> Dict[str, Any]:
    """
    Compare a 1-5 self-rating with quiz performance percentage.

    Formula: self_rating_percent = ((rating - 1) / 4) * 100
             confidence_gap = self_rating_percent - quiz_percent
    """
    if not isinstance(rating, (int, float)):
        raise TypeError(f"rating must be an integer between 1 and 5, got {type(rating).__name__}")
    if not (1 <= rating <= 5):
        raise ValueError(f"Self-rating must be between 1 and 5, got {rating}")
    if not isinstance(quiz_percent, (int, float)):
        raise TypeError(f"quiz_percent must be a number, got {type(quiz_percent).__name__}")

    self_rating_percent = round(((rating - 1) / 4.0) * 100.0, 2)
    confidence_gap = round(self_rating_percent - quiz_percent, 2)

    if confidence_gap > 0:
        interpretation = "Self-rating is higher than quiz performance."
    elif confidence_gap < 0:
        interpretation = "Quiz performance is higher than self-rating."
    else:
        interpretation = "Self-rating and quiz performance match."

    return {
        "self_rating": rating,
        "self_rating_percent": self_rating_percent,
        "quiz_percent": round(float(quiz_percent), 2),
        "confidence_gap": confidence_gap,
        "interpretation": interpretation,
        "disclaimer": "This short 5-question quiz is an experimental prototype and does not measure professional competence or job readiness.",
    }
