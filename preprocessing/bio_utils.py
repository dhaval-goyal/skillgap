from typing import List, Tuple, Union, Optional

INT_TO_BIO = {0: "O", 1: "B", 2: "I"}

def to_skill_bio(tags: List[Union[str, int]], entity_name: str = "SKILL") -> List[str]:
    """Convert raw B/I/O tags or integer IDs into standard B-SKILL/I-SKILL/O."""
    converted = []
    for tag in tags:
        t = INT_TO_BIO.get(tag, str(tag)).strip().upper()
        if t in ("O", "0"):
            converted.append("O")
        elif t in ("B", "B-SKILL", "1"):
            converted.append(f"B-{entity_name}")
        elif t in ("I", "I-SKILL", "2"):
            converted.append(f"I-{entity_name}")
        else:
            raise ValueError(f"Unsupported tag value: {tag}")
    return converted

def is_valid_bio(tags: List[str]) -> Tuple[bool, Optional[str]]:
    """Validate standard BIO sequence transition rules."""
    prev_tag = "O"
    for idx, tag in enumerate(tags):
        if tag == "O":
            prev_tag = "O"
            continue
        if not (tag.startswith("B-") or tag.startswith("I-")):
            return False, f"Invalid format at index {idx}: {tag}"
        prefix, entity = tag.split("-", 1)
        if prefix == "I":
            if prev_tag == "O":
                return False, f"Invalid transition: I-{entity} after O at index {idx}"
            _, prev_entity = prev_tag.split("-", 1)
            if prev_entity != entity:
                return False, f"Entity mismatch: I-{entity} after {prev_tag} at index {idx}"
        prev_tag = tag
    return True, None
