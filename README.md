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
- `inference/`: Skill extraction and span decoding.
- `normalization/`: Mapping extracted spans to standardized skill taxonomies.
- `gap/`: Skill-gap computation against target requirements.
- `assessment/`: Practice alignment and competency checks.
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

## Limitations

The current model does not identify every skill correctly. Performance may be affected by ambiguous expressions, rare skills, entity-boundary errors, and annotation conventions. The current results establish a baseline for further research and do not demonstrate production readiness.

## Future Work

- Complete manual review of DistilBERT validation errors.
- Improve skill extraction using validation-driven experiments.
- Develop skill normalization against a standardized taxonomy.
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

**In development.** Dataset exploration, baseline construction, DistilBERT fine-tuning, and final test evaluation have been completed. Validation error analysis and the broader skill-gap assessment workflow remain ongoing.
