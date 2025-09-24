"""
Preset configurations for Log Analyzer.

This module contains preset definitions that combine pattern and prompt files
for different types of log analysis. Presets make it easy to apply specialized
analysis configurations for different platforms or applications.

To add a new preset:
1. Create pattern and prompt files in their respective directories
2. Add an entry to the PRESETS dictionary with the preset name as key
3. Include patterns_file, prompt_file, and description fields

Example:
    PRESETS["my_preset"] = {
        "patterns_file": "patterns/my_preset.txt",
        "prompt_file": "prompts/my_preset.txt", 
        "description": "My custom log analysis preset"
    }
"""

from typing import Dict, Any

# Preset definitions - combines pattern and prompt files
PRESETS: Dict[str, Dict[str, Any]] = {
    "generic": {
        "patterns_file": "patterns/generic.txt",
        "prompt_file": "prompts/generic.txt",
        "description": "General purpose log analysis for any application"
    },
    "unity": {
        "patterns_file": "patterns/unity.txt", 
        "prompt_file": "prompts/unity.txt",
        "description": "Unity game engine build logs analysis"
    }
}

# Default preset configuration
DEFAULT_PRESET = "generic"

def get_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get a preset configuration by name.
    
    Args:
        preset_name: Name of the preset to retrieve
        
    Returns:
        Dictionary containing preset configuration
        
    Raises:
        KeyError: If the preset name is not found
    """
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(f"Preset '{preset_name}' not found. Available presets: {available}")
    return PRESETS[preset_name].copy()


def list_presets() -> Dict[str, str]:
    """
    Get a dictionary of all available presets with their descriptions.
    
    Returns:
        Dictionary mapping preset names to their descriptions
    """
    return {name: config["description"] for name, config in PRESETS.items()}


def register_preset(name: str, patterns_file: str, prompt_file: str, description: str) -> None:
    """
    Register a new preset configuration.
    
    Args:
        name: Name of the preset
        patterns_file: Path to the patterns file relative to patterns directory
        prompt_file: Path to the prompt file relative to prompts directory
        description: Human-readable description of the preset
        
    Example:
        register_preset("django", "patterns/django.txt", "prompts/django.txt", 
                       "Django web framework log analysis")
    """
    PRESETS[name] = {
        "patterns_file": patterns_file,
        "prompt_file": prompt_file,
        "description": description
    }


def is_valid_preset(preset_name: str) -> bool:
    """
    Check if a preset name is valid.
    
    Args:
        preset_name: Name to check
        
    Returns:
        True if the preset exists, False otherwise
    """
    return preset_name in PRESETS
