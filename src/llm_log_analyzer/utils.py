"""
Utility functions for Log Analyzer.
"""

import os
from typing import List
import importlib.resources as pkg_resources

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

def _find_data_file_path(file_path: str) -> str:
    """Find the actual path to a data file, checking multiple locations."""
    if os.path.exists(file_path):
        return file_path
    
    try:
        with pkg_resources.path("llm_log_analyzer", file_path) as p:
            return p
    except Exception as e:
        raise Exception(f"Error finding file {file_path}: {e}")

def load_list_from_file(file_path: str, include_comments: bool = False) -> List[str]:
    """Load list from a text file."""
    result = []
    
    try:
        found_path = _find_data_file_path(file_path)
        with open(found_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and (include_comments or not line.startswith('#')):
                    result.append(line)
    except Exception as e:
        raise Exception(f"Error reading list from file {file_path}: {e}")
    
    return result


def load_string_from_file(file_path: str) -> str:
    """Load string from a text file."""
    try:
        found_path = _find_data_file_path(file_path)
        with open(found_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise Exception(f"Error reading string from file {file_path}: {e}")
