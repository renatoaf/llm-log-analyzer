"""
Default constants and configuration values for Log Analyzer.
"""

from typing import List

# LLM Provider Configuration
DEFAULT_PROVIDER = "gemini" 

# Log Processing Constants
DEFAULT_TAIL_LINES = 10000
DEFAULT_CONTEXT_LINES = 50
DEFAULT_CHUNK_SIZE = 100000
DEFAULT_TOKENIZER = "cl100k_base"
DEFAULT_OVERLAP_SIZE = 200

# Default keywords for filtering
DEFAULT_FILTER_KEYWORDS: List[str] = [
    "error", "failed", "exception", "stderr", "fatal", "crash", "abort",
    "failure", "fail", "critical", "panic", "assert", "segmentation fault",
    "access violation", "null reference", "stack overflow"
]

# Performance Tuning Constants
DEFAULT_MAX_PARALLEL_CHUNKS = 4
DEFAULT_MAX_CHUNKS = 20

# Default output directory
DEFAULT_OUTPUT_DIR = "."

# Output File Names
JSON_OUTPUT_FILE = "analysis.json"
MARKDOWN_OUTPUT_FILE = "analysis.md"

FILTERED_LOG_FILE = "filtered_log.txt"
CHUNK_PROMPT_FILE = "chunk_prompt_{chunk_number}.txt"
CHUNK_RESPONSE_FILE = "chunk_response_{chunk_number}.txt"
AGGREGATION_PROMPT_FILE = "aggregation_prompt.txt"
AGGREGATION_RESPONSE_FILE = "aggregation_response.txt"

