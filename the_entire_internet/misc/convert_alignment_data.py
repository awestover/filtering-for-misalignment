#!/usr/bin/env python3
"""
Convert alignment_classifier_data JSONL to separate text files matching the Google folder format.
"""
import json
import os
import re
from typing import Dict


def _slugify_filename(text: str) -> str:
    """Create a safe filename from text."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if not text:
        text = "item"
    return text[:80]


def _wrap_text_every_n_words(text: str, n: int = 15) -> str:
    """
    Insert a newline after every n words.
    Collapses whitespace to single spaces between words.
    """
    if not text:
        return ""
    words = text.split()
    if not words:
        return ""
    lines = [" ".join(words[i:i + n]) for i in range(0, len(words), n)]
    return "\n".join(lines)


def _ensure_unique_path(base_dir: str, base_name: str, ext: str) -> str:
    """
    Ensure we don't overwrite files if titles collide.
    Returns a path like "<base_dir>/<base_name>.ext" or with a numeric suffix.
    """
    candidate = os.path.join(base_dir, f"{base_name}.{ext}")
    if not os.path.exists(candidate):
        return candidate
    i = 2
    while True:
        candidate = os.path.join(base_dir, f"{base_name}-{i}.{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def convert_jsonl_to_text_files(
    jsonl_path: str,
    output_base_dir: str
) -> None:
    """
    Convert JSONL file to separate text files organized by source.

    Args:
        jsonl_path: Path to the input JSONL file
        output_base_dir: Base directory for output files
    """
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Track statistics
    records_processed = 0
    files_written = 0
    errors = 0

    # Group records by formatted_source to organize into subdirectories
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())

                # Extract fields
                record_id = record.get("id", f"record_{line_num}")
                source = record.get("formatted_source", "Unknown")
                text = record.get("text", "")

                # Create subdirectory based on source
                source_slug = _slugify_filename(source)
                source_dir = os.path.join(output_base_dir, source_slug)
                os.makedirs(source_dir, exist_ok=True)

                # Create filename from record ID
                base_name = f"{files_written + 1:06d}_{_slugify_filename(record_id)}"

                # Write text file
                txt_path = _ensure_unique_path(source_dir, base_name, "txt")
                with open(txt_path, "w", encoding="utf-8") as txt_file:
                    # Wrap text at 15 words per line (matching Google folder format)
                    wrapped_text = _wrap_text_every_n_words(text, 15)
                    txt_file.write(wrapped_text)

                files_written += 1
                records_processed += 1

                # Progress updates
                if records_processed % 1000 == 0:
                    print(f"Processed {records_processed} records, written {files_written} files...")

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                errors += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                errors += 1

    # Final summary
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Records processed: {records_processed}")
    print(f"  Files written: {files_written}")
    print(f"  Errors: {errors}")
    print(f"  Output directory: {output_base_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    jsonl_path = "alignment_classifier_data/data.jsonl"
    output_dir = "alignment_classifier_data/text_files"

    print(f"Converting {jsonl_path} to text files in {output_dir}...")
    convert_jsonl_to_text_files(jsonl_path, output_dir)
