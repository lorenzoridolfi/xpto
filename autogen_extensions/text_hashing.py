import hashlib
import json
import re
from typing import Any, Dict


def hash_text(text: str) -> str:
    """
    Standardize and hash a text string using SHA256.
    - Lowercase
    - Replace multiple spaces/tabs/newlines with a single space
    - Strip leading/trailing whitespace
    Returns the hex digest.
    """
    standardized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(standardized.encode("utf-8")).hexdigest()


def hash_json(json_str: str) -> str:
    """
    Hash a JSON string after removing formatting/indentation.
    Loads the JSON, dumps it with sorted keys and no whitespace, then hashes.
    Returns the hex digest.
    """
    obj = json.loads(json_str)
    normalized = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def hash_dict(obj: Dict[Any, Any]) -> str:
    """
    Hash a Python dict by converting to a normalized JSON string, then hashing.
    Returns the hex digest.
    """
    normalized = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
