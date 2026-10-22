# SkillGap AI: Skill Extraction & Gap Analysis

SkillGap AI is an educational self-assessment and NLP project designed to help learners explore skill gaps between their existing competencies and target career requirements.

The project combines research into **skill extraction using BIO-tagged Named Entity Recognition (NER)** with an interactive Streamlit dashboard for skill assessment, job-description analysis, gap visualization, and learning guidance.

**[Live Demo — SkillGap AI](https://skillgap-ai-preview-xy5ybvuyh6xcpfpryeebdz.streamlit.app/)**

> **Disclaimer:** SkillGap is an educational and career-planning tool, not a hiring or screening platform. It does not generate hire/reject recommendations or claim to predict professional employability.

## Key Features

- **Skill assessment:** Evaluate configured skills using self-ratings, test scores, and project scores.
- **Skill-gap analysis:** Compare assessed skill levels against a target score.
- **Visual dashboard:** Display assessment results, proficiency levels, and skill gaps through charts.
- **Job-description analysis:** Detect supported skill mentions using a fixed vocabulary and estimate their requirement status using rule-based heuristics.
- **Actionable guidance:** Display recommendations based on identified skill gaps.
- **CSV export:** Download assessment results for further analysis.
- **NLP research pipeline:** Experiment with BIO-tagged skill extraction using spaCy PhraseMatcher and DistilBERT.
- **Learning resources and quiz prototype:** Provide curated learning links and a lightweight self-assessment quiz in the research modules.

## Application Screenshots

Add screenshots of the deployed application here:

- Dashboard overview
- Skill assessment and gap visualization
- Job-description analysis and requirement labels

Store the images in a folder such as `screenshots/` and update the paths to match the files committed to your repository.

## Technologies

- Python
- Streamlit
- pandas
- spaCy
- PyTorch
- Hugging Face Transformers and Datasets
- Google Colab

The deployed dashboard and the experimental NLP pipeline are separate parts of the project. The live dashboard uses rule-based job-description analysis; it should not be interpreted as running the fine-tuned DistilBERT model.

## Research Task and Dataset

- **Task:** Skill extraction using BIO sequence tagging and token classification.
- **Dataset:** Hugging Face [`jjzha/skillspan`](https://huggingface.co/datasets/jjzha/skillspan).
- **Data splits:** Training, validation, and test splits provided by the dataset.
- **Labeling approach:** `tags_knowledge`, using `B-SKILL`, `I-SKILL`, and `O` labels.

### Approaches

1. **Baseline:** spaCy `PhraseMatcher`, constructed using skill phrases from the training split.
2. **Transformer:** Fine-tuned DistilBERT token-classification model.

## Model Evaluation

The following are the reported results on the held-out test split. Precision, recall, and F1 measure skill-span extraction performance.

| Model | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|
| spaCy PhraseMatcher | 0.0892 | 0.1586 | 0.1142 |
| DistilBERT, learning rate = 5e-5 | 0.4353 | 0.4345 | 0.4349 |

DistilBERT improved test F1 by approximately **0.3207**, or **32.07 percentage points**, compared with the PhraseMatcher baseline.

### Experimental Setup

- **Model:** `distilbert-base-uncased`
- **Selected learning rate:** `5e-5`
- **Training epochs:** 3
- **Maximum sequence length:** 256 tokens
- **Model selection:** Validation metrics
- **Final evaluation:** Held-out test split

These results reflect the reported experimental setup and do not establish production readiness or guarantee reliable extraction across arbitrary job descriptions.

## Research Pipeline and Modules

The research codebase includes the following components. Verify that each listed path exists in the repository before publishing this structure.

| Component | Purpose |
|---|---|
| `preprocessing/` | Dataset loading, tokenization, and BIO label alignment |
| `baseline/` | Skill phrase dictionary and spaCy PhraseMatcher baseline |
| `model/` | DistilBERT training and evaluation code |
| `inference/` | Token-classification inference and BIO span decoding |
| `normalization/` | Canonical skill names and alias mapping |
| `gap/` | Missing-skill identification and priority ranking |
| `resources/` | Manually curated learning resources |
| `assessment/` | Quiz scoring and self-rating confidence comparison |
| `skillgap_pipeline.py` | Integration of extraction, gap analysis, ranking, and resources |
| `experiments_log.csv` | Hyperparameter experiments and validation metrics |

Model checkpoints and other large experiment artifacts are not required for the public Streamlit dashboard.

## Requirement Status Classification

The deployed dashboard includes a rule-based job-description analyzer that detects supported skill mentions and estimates their requirement status.

Supported labels:

- **Required** — the description explicitly indicates a requirement.
- **Preferred** — the skill is desirable but not necessarily mandatory.
- **Optional** — the description indicates that the skill is optional.
- **Not required / excluded** — the description explicitly says the skill is not required.
- **Mentioned — status unclear** — a skill is detected without a clear requirement signal.
- **Mixed signals — review** — conflicting requirement signals are detected for the same skill.

These labels are heuristic estimates, not definitive interpretations of job requirements. Complex wording, negation, and context may cause incorrect classifications. Users should review the original job description before relying on the results.

### Example Test Cases

| Job description | Expected result |
|---|---|
| Python is required. SQL is preferred. | Python: Required; SQL: Preferred |
| Python is optional. SQL is required. | Python: Optional; SQL: Required |
| Python is not required for this role, but SQL is preferred. | Python: Not required / excluded; SQL: Preferred |
| Python is required. Python is optional. | Python: Mixed signals — review |

These examples demonstrate selected functional tests, not a comprehensive accuracy evaluation.

## Learning Resource Catalog

The research `resources/` module provides a deterministic lookup function, `get_resources(skill: str)`, that maps canonical skills to manually curated documentation and tutorials.

Initial curated skills include Python, SQL, Power BI, Machine Learning, scikit-learn, and Docker.

**Limitation:** The catalog is static and manually curated. It is not dynamically retrieved from external searches or experimentally validated as an effective learning intervention.

## Quiz Assessment Prototype

The research `assessment/` module provides a deterministic quiz engine with 15 multiple-choice questions: five each for Python, SQL, and Power BI.

It compares quiz performance with a learner's self-rating on a scale of 1–5.

### Formulas

**Self-rating percentage**

\[
\text{Self-rating percentage}
=\frac{\text{rating}-1}{4}\times100
\]

**Confidence gap**

\[
\text{Confidence gap}
=\text{Self-rating percentage}-\text{Quiz percentage}
\]

- Positive value: self-rating exceeds quiz performance.
- Negative value: quiz performance exceeds self-rating.
- Zero: both measures match.

> **Prototype limitation:** This short quiz is an experimental self-reflection tool, not an accredited examination or a validated predictor of professional employability.

## Pipeline Integration

The research `skillgap_pipeline.py` module integrates the following workflow:

1. Extract skill mentions from job descriptions and résumé text.
2. Normalize skill names and resolve supported aliases.
3. Identify missing skills through comparison of skill sets.
4. Rank missing skills using mention counts.
5. Attach curated learning resources.
6. Optionally evaluate quiz performance and compare it with self-ratings.

Model-based inference requires the trained model checkpoint to be available at the configured path. The deployed Streamlit dashboard is a separate application and should not be assumed to load this checkpoint.

## Testing and Limitations

The deployed dashboard has been manually tested for:

- Skill-gap calculations and score validation.
- Charts and recommendations.
- CSV downloads.
- Supported skill extraction.
- Missing assessment handling.
- Selected requirement-status edge cases, including negation and conflicting signals.

The research pipeline and model experiments have separate evaluation results. Functional tests of the dashboard should not be interpreted as a comprehensive evaluation of NLP accuracy.

### Current Limitations

- Skill extraction relies on a fixed vocabulary and may miss skills outside configured patterns.
- Requirement-status classification uses heuristic rules rather than a separately trained classifier.
- Complex sentences, contextual language, and negation may produce incorrect labels.
- The model's reported test F1 of 0.4349 indicates substantial room for improvement in skill-span extraction.
- The static learning resource catalog is not dynamically personalized or experimentally validated.
- The quiz prototype is not a validated measure of professional competence.

## Privacy and Ethical Scope

The public deployment package is intended to contain the Streamlit application and its required dependencies without saved model checkpoints or credentials used during experimentation. Verify the actual repository contents and deployment configuration before relying on this statement.

SkillGap is designed for educational self-assessment and career planning. It is not intended for automated hiring decisions, candidate rejection, or validated predictions of workplace performance.

## Running Locally

For the deployed Streamlit dashboard, clone the repository and install its declared dependencies.

```bash
git clone https://github.com/dhaval-goyal/skillgap-ai-preview.git
cd skillgap-ai-preview
pip install -r requirements.txt
streamlit run app.py
```

The repository's `requirements.txt` must contain the dependencies required by the current dashboard. The experimental NLP pipeline may require additional dependencies and a separately configured model checkpoint.

## Future Work

- Improve skill extraction through validation-driven experiments.
- Analyze false positives, missed skill spans, and boundary errors.
- Expand skill normalization using established skill taxonomies.
- Improve requirement classification and test additional contextual edge cases.
- Connect assessed competencies with missing job skills.
- Expand learning resources and evaluate the usefulness of recommendations.
- Conduct broader functional and usability testing.

## Project Status

**Research prototype with a deployed Streamlit dashboard.**

The project includes reported PhraseMatcher and DistilBERT experiments, skill normalization and gap-analysis modules, learning-resource and quiz prototypes, pipeline integration, and a manually tested dashboard. Further model evaluation, usability testing, and improvements to extraction and requirement classification remain areas for future work.
