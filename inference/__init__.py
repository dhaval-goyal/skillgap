"""Inference package for skill entity extraction."""
from inference.extractor import (
    SkillExtractor,
    decode_bio_spans,
    get_extractor,
    ID2LABEL,
    LABEL2ID,
    DEFAULT_CHECKPOINT_DIR,
)

__all__ = [
    "SkillExtractor",
    "decode_bio_spans",
    "get_extractor",
    "ID2LABEL",
    "LABEL2ID",
    "DEFAULT_CHECKPOINT_DIR",
]
