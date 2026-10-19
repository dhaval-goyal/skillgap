"""
DistilBERT inference module for extracting skill entity spans.

Encapsulates tokenization, model forward pass, and BIO span decoding.
"""
from typing import List, Tuple, Dict, Optional, Any

ID2LABEL: Dict[int, str] = {0: "O", 1: "B-SKILL", 2: "I-SKILL"}
LABEL2ID: Dict[str, int] = {"O": 0, "B-SKILL": 1, "I-SKILL": 2}
DEFAULT_CHECKPOINT_DIR: str = "checkpoints/best_model"


def decode_bio_spans(text: str, tags: List[str], offsets: List[Tuple[int, int]]) -> List[str]:
    """
    Decode BIO predicted tags and token character offsets into raw text spans.

    Skips special/padding tokens (offset start == end) and extracts text slices.
    """
    spans: List[str] = []
    current_start: Optional[int] = None
    current_end: Optional[int] = None

    for tag, (start, end) in zip(tags, offsets):
        if start == end:
            continue
        if tag == "B-SKILL":
            if current_start is not None:
                spans.append(text[current_start:current_end].strip())
            current_start, current_end = start, end
        elif tag == "I-SKILL":
            if current_start is not None:
                current_end = end
            else:
                current_start, current_end = start, end
        else:
            if current_start is not None:
                spans.append(text[current_start:current_end].strip())
                current_start, current_end = None, None

    if current_start is not None:
        spans.append(text[current_start:current_end].strip())

    return [s for s in spans if s]


class SkillExtractor:
    """Manages DistilBERT model loading, token classification, and extraction."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_CHECKPOINT_DIR
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    def load(self) -> None:
        """Load tokenizer and model from checkpoint path."""
        import os
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"DistilBERT model weights not found at '{self.model_path}'. "
                "Training was performed in Google Colab (checkpoint: "
                "'/content/drive/MyDrive/SkillGap/distilbert_experiments/distilbert_lr_5e-05/best_model'). "
                "Download and place the model folder in 'checkpoints/best_model' for local inference."
            )
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
        except ImportError as err:
            raise ImportError("transformers is required for model inference. Install via requirements.txt.") from err

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.eval()

    def extract(self, text: str) -> List[str]:
        """Extract skill mentions from text using model and tokenizer."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if not text.strip():
            return []
        if self.model is None or self.tokenizer is None:
            self.load()

        import torch
        inputs = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        raw_offsets = inputs.pop("offset_mapping")[0]
        offsets = raw_offsets.tolist() if hasattr(raw_offsets, "tolist") else list(raw_offsets)

        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_preds = torch.argmax(outputs.logits, dim=-1)[0]
            predictions = raw_preds.tolist() if hasattr(raw_preds, "tolist") else list(raw_preds)

        id2label = getattr(self.model.config, "id2label", ID2LABEL)
        tags = [id2label.get(pid, ID2LABEL.get(pid, "O")) for pid in predictions]

        return decode_bio_spans(text, tags, offsets)


_EXTRACTOR_CACHE: Dict[str, SkillExtractor] = {}


def get_extractor(model_path: Optional[str] = None) -> SkillExtractor:
    """Retrieve or initialize a cached SkillExtractor instance."""
    path = model_path or DEFAULT_CHECKPOINT_DIR
    if path not in _EXTRACTOR_CACHE:
        _EXTRACTOR_CACHE[path] = SkillExtractor(model_path=path)
    return _EXTRACTOR_CACHE[path]
