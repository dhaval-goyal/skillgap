"""
Evaluation utilities and diagnostic suites for SkillGap skill extraction.

Provides:
1. Strict entity-level precision, recall, and F1 calculation (span-based).
2. Dual-section diagnostic test suite:
   - Section 1: Gold-annotated validation examples from jjzha/skillspan.
   - Section 2: Manually authored sanity-check test cases.
3. Diagnostic runner to evaluate any SkillExtractor against the suite.
"""
from typing import List, Tuple, Set, Dict, Any, Optional


def extract_entities_from_bio(tags: List[str]) -> Set[Tuple[str, int, int]]:
    """
    Extract entity spans from a BIO tag sequence.

    Parameters:
        tags: List of BIO strings, e.g. ['O', 'B-SKILL', 'I-SKILL', 'O'].

    Returns:
        Set of tuples (entity_type, start_token_idx, end_token_idx_exclusive).
    """
    entities: Set[Tuple[str, int, int]] = set()
    current_entity: Optional[str] = None
    start_idx: Optional[int] = None

    for idx, tag in enumerate(tags):
        # Ignore special token masks if present
        if tag in ("-100", "[PAD]", "[CLS]", "[SEP]"):
            tag = "O"

        if tag.startswith("B-"):
            if current_entity is not None and start_idx is not None:
                entities.add((current_entity, start_idx, idx))
            current_entity = tag.split("-", 1)[1]
            start_idx = idx
        elif tag.startswith("I-"):
            entity_type = tag.split("-", 1)[1]
            if current_entity == entity_type and start_idx is not None:
                # Continuation of current entity
                pass
            else:
                # Malformed transition: close previous entity if any and start new
                if current_entity is not None and start_idx is not None:
                    entities.add((current_entity, start_idx, idx))
                current_entity = entity_type
                start_idx = idx
        else:  # 'O' or unrecognized
            if current_entity is not None and start_idx is not None:
                entities.add((current_entity, start_idx, idx))
                current_entity = None
                start_idx = None

    if current_entity is not None and start_idx is not None:
        entities.add((current_entity, start_idx, len(tags)))

    return entities


def compute_entity_metrics(
    predictions: List[List[str]],
    references: List[List[str]]
) -> Dict[str, float]:
    """
    Compute entity-level Precision, Recall, and F1 across multiple sequences.

    Matches exact span boundaries and entity types. Ignores -100 subtoken masks.

    Parameters:
        predictions: List of predicted tag lists (one per sequence).
        references: List of gold tag lists (one per sequence).

    Returns:
        Dict with keys: 'precision', 'recall', 'f1', 'true_positives', 'false_positives', 'false_negatives'.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred_tags, gold_tags in zip(predictions, references):
        # Filter out ignored subtoken slots where gold tag is -100
        clean_pred: List[str] = []
        clean_gold: List[str] = []
        for p, g in zip(pred_tags, gold_tags):
            if str(g) == "-100":
                continue
            clean_pred.append(str(p))
            clean_gold.append(str(g))

        pred_spans = extract_entities_from_bio(clean_pred)
        gold_spans = extract_entities_from_bio(clean_gold)

        tp = len(pred_spans.intersection(gold_spans))
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
    }


# ============================================================================
# DIAGNOSTIC EVALUATION SUITE
# ============================================================================
# Divided explicitly into:
# Section 1: Gold-annotated validation records from jjzha/skillspan (dev split)
# Section 2: Manually constructed sanity check sentences for SkillGap requirements
# ============================================================================

DIAGNOSTIC_SUITE: Dict[str, List[Dict[str, Any]]] = {
    "gold_validation_examples": [
        {
            "id": "gold_dev_containers",
            "domain": "Containers & Cloud Infrastructure",
            "source": "SkillSpan dev split (dev.json)",
            "text": "Understanding of containers with heavy experience in at least one or more of: Docker Kubernetes Rancher",
            "tokens": [
                "Understanding", "of", "containers", "with", "heavy", "experience",
                "in", "at", "least", "one", "or", "more", "of:", "Docker", "Kubernetes", "Rancher"
            ],
            "tags_knowledge": [
                "O", "O", "B", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B", "B", "B"
            ],
            "expected_skills": ["containers", "Docker", "Kubernetes", "Rancher"],
        },
        {
            "id": "gold_dev_cloud_providers",
            "domain": "Cloud Platforms (AWS, Azure, GCP)",
            "source": "SkillSpan dev split (dev.json)",
            "text": "Working experience with cloud infrastructure/services ( AWS or Azure preferred also Google Cloud Platform )",
            "tokens": [
                "Working", "experience", "with", "cloud", "infrastructure/services",
                "(", "AWS", "or", "Azure", "preferred", "also", "Google", "Cloud", "Platform", ")"
            ],
            "tags_knowledge": [
                "O", "O", "O", "B", "I", "O", "B", "O", "B", "O", "O", "B", "I", "I", "I"
            ],
            "expected_skills": ["cloud infrastructure/services", "AWS", "Azure", "Google Cloud Platform )"],
        },
        {
            "id": "gold_dev_languages_frameworks",
            "domain": "Backend Languages & Frameworks",
            "source": "SkillSpan dev split (dev.json)",
            "text": "Deep hands-on Java ( plus Spring framework ) and/or C#/.NET .",
            "tokens": [
                "Deep", "hands-on", "Java", "(", "plus", "Spring", "framework", ")", "and/or", "C#/.NET", "."
            ],
            "tags_knowledge": [
                "O", "O", "B", "O", "O", "B", "I", "I", "O", "B", "O"
            ],
            "expected_skills": ["Java", "Spring framework )", "C#/.NET"],
        },
        {
            "id": "gold_dev_databases_web",
            "domain": "Databases & Web Technologies",
            "source": "SkillSpan dev split (dev.json)",
            "text": "Experience with some or all of the following: MySQL Aurora Nginx Hibernate Apache Java Spring Jenkins Git",
            "tokens": [
                "Experience", "with", "some", "or", "all", "of", "the", "following:",
                "MySQL", "Aurora", "Nginx", "Hibernate", "Apache", "Java", "Spring", "Jenkins", "Git"
            ],
            "tags_knowledge": [
                "O", "O", "O", "O", "O", "O", "O", "O", "B", "B", "B", "B", "B", "B", "B", "B", "B"
            ],
            "expected_skills": ["MySQL", "Aurora", "Nginx", "Hibernate", "Apache", "Java", "Spring", "Jenkins", "Git"],
        },
        {
            "id": "gold_dev_multiword_architecture",
            "domain": "Enterprise Architecture & Microservices",
            "source": "SkillSpan dev split (dev.json)",
            "text": "Understands the challenges presented by cloud adoption and migration in both enterprise and green field contexts; how to build cloud native applications scratch and how to tackle monolithic system estates through the introduction of APIs microservices and gateways",
            "tokens": [
                "Understands", "the", "challenges", "presented", "by", "cloud", "adoption", "and", "migration",
                "in", "both", "enterprise", "and", "green", "field", "contexts;", "how", "to", "build",
                "cloud", "native", "applications", "scratch", "and", "how", "to", "tackle", "monolithic",
                "system", "estates", "through", "the", "introduction", "of", "APIs", "microservices", "and", "gateways"
            ],
            "tags_knowledge": [
                "O", "O", "O", "O", "O", "B", "I", "I", "I",
                "O", "O", "B", "I", "I", "I", "I", "O", "O", "O",
                "O", "O", "O", "O", "O", "O", "O", "O", "O",
                "O", "O", "O", "O", "O", "O", "B", "B", "O", "B"
            ],
            "expected_skills": [
                "cloud adoption and migration", "enterprise and green field contexts;", "APIs", "microservices", "gateways"
            ],
        },
    ],
    "sanity_check_examples": [
        {
            "id": "sanity_prompt_primary",
            "domain": "Stage 11.6 Target Prompt",
            "source": "Manually written SkillGap test case",
            "text": "We need a data analyst with strong SQL and Python skills, experience with Power BI dashboards, and familiarity with Docker.",
            "target_keywords": ["SQL", "Python", "Power BI", "Docker"],
        },
        {
            "id": "sanity_data_engineering",
            "domain": "Data Engineering & Warehousing",
            "source": "Manually written SkillGap test case",
            "text": "Seeking a data engineer proficient in Apache Spark, PostgreSQL, and Snowflake data pipelines.",
            "target_keywords": ["Apache Spark", "PostgreSQL", "Snowflake"],
        },
        {
            "id": "sanity_cloud_devops",
            "domain": "Cloud DevOps Stack",
            "source": "Manually written SkillGap test case",
            "text": "Must have hands-on experience with Kubernetes, Terraform, and AWS cloud architecture.",
            "target_keywords": ["Kubernetes", "Terraform", "AWS"],
        },
        {
            "id": "sanity_ml_stack",
            "domain": "Machine Learning Stack",
            "source": "Manually written SkillGap test case",
            "text": "Requirements include scikit-learn, PyTorch, and machine learning model deployment.",
            "target_keywords": ["scikit-learn", "PyTorch", "machine learning"],
        },
    ],
}


def evaluate_diagnostic_suite(extractor: Any) -> Dict[str, Any]:
    """
    Run the full diagnostic suite through an extractor and return evaluation summary.

    Parameters:
        extractor: An object implementing .extract(text: str) -> List[str].

    Returns:
        Structured dictionary summarizing results on gold and sanity examples.
    """
    results: Dict[str, Any] = {
        "gold_results": [],
        "sanity_results": [],
        "gold_summary": {"total_expected": 0, "total_matched": 0},
        "sanity_summary": {"total_targets": 0, "total_found": 0},
    }

    # Evaluate gold validation examples
    for ex in DIAGNOSTIC_SUITE["gold_validation_examples"]:
        extracted = extractor.extract(ex["text"])
        extracted_lower = {s.lower() for s in extracted}
        expected = ex["expected_skills"]
        matched = [e for e in expected if any(e.lower() in s or s in e.lower() for s in extracted_lower)]

        results["gold_results"].append({
            "id": ex["id"],
            "domain": ex["domain"],
            "text": ex["text"],
            "expected": expected,
            "extracted": extracted,
            "matched": matched,
            "recall": len(matched) / len(expected) if expected else 1.0,
        })
        results["gold_summary"]["total_expected"] += len(expected)
        results["gold_summary"]["total_matched"] += len(matched)

    # Evaluate sanity check examples
    for ex in DIAGNOSTIC_SUITE["sanity_check_examples"]:
        extracted = extractor.extract(ex["text"])
        extracted_lower = {s.lower() for s in extracted}
        targets = ex["target_keywords"]
        found = [t for t in targets if any(t.lower() in s for s in extracted_lower)]

        results["sanity_results"].append({
            "id": ex["id"],
            "domain": ex["domain"],
            "text": ex["text"],
            "targets": targets,
            "extracted": extracted,
            "found": found,
            "hit_rate": len(found) / len(targets) if targets else 1.0,
        })
        results["sanity_summary"]["total_targets"] += len(targets)
        results["sanity_summary"]["total_found"] += len(found)

    gold_exp = results["gold_summary"]["total_expected"]
    gold_mat = results["gold_summary"]["total_matched"]
    results["gold_summary"]["overall_recall"] = round(gold_mat / gold_exp, 4) if gold_exp > 0 else 0.0

    san_tar = results["sanity_summary"]["total_targets"]
    san_fnd = results["sanity_summary"]["total_found"]
    results["sanity_summary"]["overall_hit_rate"] = round(san_fnd / san_tar, 4) if san_tar > 0 else 0.0

    return results
