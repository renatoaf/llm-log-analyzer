"""
Utility functions for Log Analyzer.
"""

import os
from typing import List, Optional
from importlib.resources import files

MIN_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.9

def get_confidence_emoji(confidence: float) -> str:
    """Get emoji for confidence level."""
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        return "🔴"
    elif confidence < HIGH_CONFIDENCE_THRESHOLD:
        return "🟡"
    else:
        return "🟢"

def _find_data_file_path(file_path: str) -> Optional[str]:
    """Return path if the file exists on disk, else None."""
    if os.path.exists(file_path):
        return file_path
    return None

def _read_package_resource(file_path: str) -> str:
    """Read a package resource (e.g. patterns/unity.txt) using the modern importlib.resources API."""
    parts = file_path.replace("\\", "/").split("/")
    ref = files("llm_log_analyzer").joinpath(*parts)
    if not ref.is_file():
        raise FileNotFoundError(f"Package resource {file_path!r} not found")
    return ref.read_text(encoding="utf-8")

def load_list_from_file(file_path: str, include_comments: bool = False) -> List[str]:
    """Load list from a text file (local path or package resource)."""
    result = []
    try:
        found_path = _find_data_file_path(file_path)
        if found_path is not None:
            with open(found_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = _read_package_resource(file_path)
        for line in content.splitlines():
            line = line.strip()
            if line and (include_comments or not line.startswith("#")):
                result.append(line)
    except Exception as e:
        raise Exception(f"Error reading list from file {file_path}: {e}") from e
    return result


def load_string_from_file(file_path: str) -> str:
    """Load string from a text file (local path or package resource)."""
    try:
        found_path = _find_data_file_path(file_path)
        if found_path is not None:
            with open(found_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return _read_package_resource(file_path).strip()
    except Exception as e:
        raise Exception(f"Error reading string from file {file_path}: {e}") from e
