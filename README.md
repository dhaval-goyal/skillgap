# SkillGap: Skill Extraction & Gap Analysis

SkillGap is an educational and self-assessment NLP system designed to extract skills from text using Named Entity Recognition (BIO tagging) and analyze gaps between demonstrated competencies and target career requirements.

> **Disclaimer & Ethics**: This is an educational guidance tool, **not** a hiring or screening platform. It does not output hire/reject scores and does not store raw user resume text.

## Task & Dataset
- **Task**: Skill extraction with BIO sequence tagging (Token Classification).
- **Dataset**: Hugging Face [`jjzha/skillspan`](https://huggingface.co/datasets/jjzha/skillspan) using the official train, dev, and test splits.
- **Approaches**:
  1. **Baseline**: spaCy `PhraseMatcher` constructed strictly from the training split dictionary.
  2. **Transformer**: Fine-tuned DistilBERT token classification model.

## Repository Structure
- `preprocessing/`: Dataset download, tokenization, and BIO label alignment.
- `baseline/`: Training-split dictionary extraction and spaCy PhraseMatcher.
- `model/`: DistilBERT model definition, training loop, and evaluation.
- `inference/`: Entity extraction and span decoding from raw text.
- `normalization/`: Rule-based skill canonicalization (mapping surface aliases like `powerbi` to `Power BI`, handling whitespace/casing).
- `gap/`: Skill gap computation against target requirements.
- `assessment/`: Practice alignment and competency checks.
- `app/`: Interactive interface for self-guided gap analysis.
- `experiments_log.csv`: Tracking hyperparameter runs and validation metrics.

## Status & Benchmarks
- Baseline Validation F1: *Not yet measured*
- DistilBERT Validation F1: *Not yet measured*
- Test Split Evaluation: *Reserved for final benchmark only*
