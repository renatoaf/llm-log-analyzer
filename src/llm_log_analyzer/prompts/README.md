# Prompt Files

This directory contains prompt files used by presets to guide the LLM analysis.

## Adding New Prompt Files

To create a new prompt file for a custom preset:

1. Create a new `.txt` file with your preset name (e.g., `django.txt`)
2. Write a detailed prompt that explains how to analyze logs for your specific use case
3. Include instructions for identifying issues, categorizing problems, and providing solutions
4. Make the prompt specific to your domain - do not be specific about structure, as this is handled by the log analyzer application

## Prompt File Structure

A good prompt file should include:

- **Context**: What type of logs are being analyzed
- **Objectives**: What kind of issues to look for
- **Analysis approach**: How to categorize and prioritize findings
- **Domain expertise**: Specific knowledge about the technology/platform

## Example Prompt Files

- `generic.txt` - General purpose log analysis prompt
- `unity.txt` - Unity game engine build log analysis prompt

## Tips for Writing Effective Prompts

1. Be specific about the domain and technology
2. Provide examples of common issues and their solutions
3. Include guidance on severity assessment
4. Suggest actionable remediation steps
5. Consider the target audience (developers, ops team, etc.)
