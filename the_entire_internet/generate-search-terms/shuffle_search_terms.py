#!/usr/bin/env python3
"""
Shuffle and clean search_terms.md:
- Remove empty lines
- Shuffle the remaining lines
- Write back to the file
"""
import random
import sys
import os

def shuffle_and_clean_file(filepath: str) -> None:
    """Read file, remove empty lines, shuffle, and write back."""
    # Read all lines
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    # Remove empty lines and strip whitespace
    non_empty_lines = [line.strip() for line in lines if line.strip()]

    original_count = len(lines)
    cleaned_count = len(non_empty_lines)

    print(f"Original lines: {original_count}")
    print(f"After removing empty lines: {cleaned_count}")
    print(f"Removed {original_count - cleaned_count} empty lines")

    # Shuffle
    random.shuffle(non_empty_lines)
    print(f"Shuffled {cleaned_count} search terms")

    # Write back
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(non_empty_lines))
        f.write("\n")  # Add trailing newline

    print(f"Written to {filepath}")

if __name__ == "__main__":
    default_path = os.path.join(os.path.dirname(__file__), "search_terms.md")
    filepath = sys.argv[1] if len(sys.argv) > 1 else default_path
    shuffle_and_clean_file(filepath)
