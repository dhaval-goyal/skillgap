"""
Data preprocessing utilities for token-level skill extraction.

NOTE ON ENTITY SEMANTICS:
We use 'tags_knowledge' as our current experimental target. In the SkillSpan
dataset, both 'tags_knowledge' and 'tags_competence' are provided. Before
final model training, target entity semantics (Knowledge vs Competence vs Union)
must be systematically verified against the project scope.
"""
from typing import List, Union, Dict
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
