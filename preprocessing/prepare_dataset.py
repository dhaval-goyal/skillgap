"""
Data preprocessing utilities for token-level skill extraction.

NOTE ON ENTITY SEMANTICS:
We use 'tags_knowledge' as our current experimental target. In the SkillSpan
dataset, both 'tags_knowledge' and 'tags_competence' are provided. Before
final model training, target entity semantics (Knowledge vs Competence vs Union)
must be systematically verified against the project scope.
"""
from typing import List, Union, Dict, Optional, Any
from preprocessing.bio_utils import to_skill_bio, is_valid_bio

def preprocess_example(
    tokens: List[str], 
    tags: List[Union[str, int]], 
    entity_name: str = "SKILL"
) -> Dict[str, List[str]]:
    """Validate token/tag alignment and BIO sequence; return structured example."""
    if len(tokens) != len(tags):
        raise ValueError(f"Length mismatch: {len(tokens)} tokens vs {len(tags)} tags")
    
    labels = to_skill_bio(tags, entity_name=entity_name)
    valid, err_msg = is_valid_bio(labels)
    if not valid:
        raise ValueError(f"Invalid BIO sequence: {err_msg}")
    
    return {"tokens": tokens, "labels": labels}


def preprocess_dataset_record(
    record: Dict[str, Any],
    target_field: str = "tags_knowledge",
    entity_name: str = "SKILL"
) -> Dict[str, List[str]]:
    """
    Extract and validate a record from a dataset (e.g. HuggingFace jjzha/skillspan).

    Parameters:
        record: Dictionary containing 'tokens' and the target tag field.
        target_field: Annotation field to extract ('tags_knowledge' or 'tags_skill').
        entity_name: Canonical entity name suffix for BIO tags (e.g. 'SKILL').

    Returns:
        Dictionary with validated 'tokens' and 'labels'.
    """
    if "tokens" not in record:
        raise KeyError("Record missing 'tokens' field")
    if target_field not in record:
        raise KeyError(f"Record missing '{target_field}' field")

    tokens = list(record["tokens"])
    tags = list(record[target_field])
    return preprocess_example(tokens, tags, entity_name=entity_name)


def align_labels_with_tokens(
    labels: List[str],
    word_ids: List[Optional[int]],
    label2id: Dict[str, int],
    label_all_subtokens: bool = False
) -> List[int]:
    """
    Align word-level BIO labels with WordPiece subtokens for transformer training.

    Parameters:
        labels: Word-level BIO tags (e.g. ['B-SKILL', 'O']).
        word_ids: List of word indices for each subtoken (None for [CLS], [SEP], [PAD]).
        label2id: Dictionary mapping tag strings to integer IDs.
        label_all_subtokens: If True, continuation subtokens inherit the word's label.
                             If False (standard), continuation subtokens receive -100.

    Returns:
        List of aligned integer label IDs with special tokens masked as -100.
    """
    aligned_labels: List[int] = []
    previous_word_id: Optional[int] = None

    for word_id in word_ids:
        if word_id is None:
            # Special tokens ([CLS], [SEP], padding) are ignored by PyTorch loss
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            # First subtoken of the word gets the word's label
            tag = labels[word_id]
            aligned_labels.append(label2id[tag])
        else:
            # Subsequent subtoken of the same word
            if label_all_subtokens:
                tag = labels[word_id]
                aligned_labels.append(label2id[tag])
            else:
                aligned_labels.append(-100)
        previous_word_id = word_id

    return aligned_labels

