# SkillGap: Skill Extraction & Gap Analysis

SkillGap is an educational and self-assessment NLP system designed to extract skills from text using Named Entity Recognition (BIO tagging) and analyze gaps between demonstrated competencies and target career requirements.

> **Disclaimer & Ethics:** This is an educational guidance tool, not a hiring or screening platform. It does not output hire/reject scores and does not store raw user resume text.

## Task & Dataset

- **Task:** Skill extraction using BIO sequence tagging (token classification).
- **Dataset:** Hugging Face [`jjzha/skillspan`](https://huggingface.co/datasets/jjzha/skillspan), using the provided train, validation, and test splits.
- **Approaches:**
  1. **Baseline:** spaCy `PhraseMatcher`, constructed using skill phrases from the training split.
  2. **Transformer:** Fine-tuned DistilBERT token-classification model.

## Repository Structure

- `preprocessing/`: Dataset loading, tokenization, and BIO label alignment.
- `baseline/`: Training-split skill dictionary and spaCy PhraseMatcher.
- `model/`: DistilBERT training, model checkpoints, and evaluation.
- `inference/`: DistilBERT token classification inference (`SkillExtractor`) and BIO span decoding (`decode_bio_spans`).
- `normalization/`: Rule-based skill canonicalization (`normalize_skill`, mapping aliases like `powerbi` to `Power BI`, whitespace/case standardization).
- `gap/`: Skill gap calculation (`compute_skill_gap`), frequency ranking (`rank_skills_by_frequency`), and missing skill prioritization (`rank_missing_skills`).
- `resources/`: Manually curated learning resource catalog (`get_resources`) for recommended skill learning materials.
- `assessment/`: Skill quiz prototype (`get_questions`, `calculate_score`, `compare_confidence`) and confidence-gap comparison.
- `skillgap_pipeline.py`: End-to-end integration pipeline orchestrating extraction, gap analysis, ranking, and learning resources.
- `app/`: Interactive interface for self-guided gap analysis.
- `experiments_log.csv`: Hyperparameter experiments and validation metrics.

## Model Evaluation

The following results were obtained on the held-out test split. Precision, recall, and F1 are reported for skill-span extraction.

| Model | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|
| spaCy PhraseMatcher | 0.0892 | 0.1586 | 0.1142 |
| DistilBERT (learning rate = 5e-5) | 0.4353 | 0.4345 | 0.4349 |

DistilBERT substantially outperformed the rule-based baseline, increasing test F1 from 0.1142 to 0.4349. This is an absolute improvement of approximately 32.07 percentage points.

### Experimental Setup

- **Transformer:** `distilbert-base-uncased`
- **Selected learning rate:** `5e-5`
- **Training epochs:** 3
- **Maximum sequence length:** 256 tokens
- **Evaluation approach:** Validation metrics used for model selection; the test split used for final evaluation.

## Error Analysis

Validation-set error analysis is used to investigate:

- False-positive skill predictions.
- Missed skill spans.
- Partial or incorrect entity boundaries.
- Potential challenges involving ambiguous phrases and longer expressions.

The findings will inform future validation-based experiments. Test-set results are reserved for final reporting rather than iterative model tuning.

## Learning Resource Catalog (Stage 10A)

The `resources/` package provides a deterministic lookup function, `get_resources(skill: str)`, mapping canonical skills to curated documentation and tutorials.

> **Note on Catalog Nature:** The learning resource catalog is **manually curated and static**. It is not AI-personalized, dynamically retrieved from external searches, or experimentally validated.

Initial curated skills: Python, SQL, Power BI, Machine Learning, scikit-learn, and Docker.

## Quiz Assessment Prototype (Stage 10B)

The `assessment/` package provides a lightweight, deterministic quiz engine with 15 multiple-choice questions (5 Python, 5 SQL, 5 Power BI). It evaluates skill understanding and contrasts performance against a learner's 1–5 self-rating.

> **Academic Prototype Disclaimer:** This short 5-question per skill quiz is an experimental self-reflection tool, **not** an accredited exam or validated predictor of professional employability.

### Formulas
- **Self-Rating Percentage:**
  $$\text{self\_rating\_percent} = \frac{\text{rating} - 1}{4} \times 100$$
- **Confidence Gap:**
  $$\text{confidence\_gap} = \text{self\_rating\_percent} - \text{quiz\_percent}$$
  - *Positive:* Self-rating is higher than quiz performance.
  - *Negative:* Quiz performance is higher than self-rating.
  - *Zero:* Both match.

### Usage Example
```python
from assessment import get_questions, calculate_score, compare_confidence

# 1. Fetch student-facing questions (answer key hidden)
questions = get_questions("Python")

# 2. Score submitted answers
result = calculate_score("Python", {"py_01": "x = (1, 2, 3)"})

# 3. Compare with self-rating (e.g., 4 out of 5)
comparison = compare_confidence(rating=4, quiz_percent=result["score_percent"])
print(comparison["interpretation"])
```

## Pipeline Integration (Stage 11)

The [`skillgap_pipeline.py`](file:///c:/Users/goyal/skillgap/skillgap_pipeline.py) module orchestrates the complete workflow:
1. Extraction of raw skill mentions from job descriptions and résumés.
2. Canonical normalization to resolve naming variations and aliases.
3. Gap analysis identifying missing skills via set difference.
4. Frequency-based ranking of missing skills using mention counts.
5. Attachment of curated educational resources.
6. Optional skill quiz assessment and self-rating confidence comparison.

> **Academic Prototype Notice:** This pipeline is an educational research tool. It does not output hiring recommendations, calculate hire/reject scores, store raw résumés, or claim validated prediction of workplace employability.

### Model Inference Prerequisites
Model training was conducted in Google Colab (checkpoint: `/content/drive/MyDrive/SkillGap/distilbert_experiments/distilbert_lr_5e-05/best_model`).
To run live DistilBERT inference locally:
1. Download or copy the `best_model` folder from Google Drive into `checkpoints/best_model/` (or provide `model_path` to `SkillExtractor`).
2. If weights are not placed locally, calling `extract_skills()` raises a descriptive `FileNotFoundError`, while `analyze_job_match()` can still be run by providing a custom extraction callable `extract_fn`.

### Usage Example
```python
from skillgap_pipeline import analyze_job_match, assess_skill

# Example with a mock extraction function
report = analyze_job_match(
    job_description="Seeking a Python developer experienced with Docker and SQL.",
    resume_text="Experienced in Python and SQL development.",
    extract_fn=lambda text: ["Python", "Docker", "SQL"] if "Seeking" in text else ["Python", "SQL"]
)

print("Missing skills:", report["missing_skills"])
# Output: ['Docker']

print("Ranked missing:", report["ranked_missing_skills"])
# Output: [{'skill': 'Docker', 'count': 1}]

# Optional: assess understanding on a missing skill
quiz_eval = assess_skill("Python", {"py_01": "x = (1, 2, 3)"}, self_rating=4)
```

### Running Tests
To run the complete test suite across all modules:
```bash
python -m unittest discover -s tests -v
```

## Limitations

The current model does not identify every skill correctly. Performance may be affected by ambiguous expressions, rare skills, entity-boundary errors, and annotation conventions. The current results establish a baseline for further research and do not demonstrate production readiness.

## Future Work

- Complete manual review of DistilBERT validation errors.
- Improve skill extraction using validation-driven experiments.
- Expand skill normalization to large-scale external taxonomies (such as ESCO or O*NET).
- Compare extracted job requirements with assessed learner competencies.
- Build a skill-gap analysis and personalized practice recommendation workflow.
- Develop and evaluate the interactive self-assessment interface.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers and Datasets
- spaCy
- pandas
- Google Colab

## Project Status

**In development.** Dataset exploration, baseline construction, DistilBERT fine-tuning, final test evaluation, Stage 8 skill normalization, Stage 9 gap analysis and priority ranking, Stage 10A learning resource catalog, Stage 10B quiz assessment prototype, and Stage 11 pipeline integration have been completed. Validation error analysis and user interface dashboard workflows remain ongoing.
