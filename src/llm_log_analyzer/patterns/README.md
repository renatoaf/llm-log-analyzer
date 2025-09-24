# Pattern Files

This directory contains pattern files used by presets to identify relevant log entries.

## Adding New Pattern Files

To create a new pattern file for a custom preset:

1. Create a new `.txt` file with your preset name (e.g., `django.txt`)
2. Add text patterns, one per line, that match log entries relevant to your use case
3. Comments can be added using `#` at the start of a line

## Pattern File Format

```
# This is a comment
exception
traceback
fatal
```

## Example Pattern Files

- `generic.txt` - General purpose patterns for common errors
- `unity.txt` - Unity game engine specific build log patterns
