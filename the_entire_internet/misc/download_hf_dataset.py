#!/usr/bin/env python3
"""
Download the alignment-classifier-documents-unlabeled dataset from Hugging Face.
Dataset: Kyle1668/alignment-classifier-documents-unlabeled
Size: ~57.9k rows
Format: Parquet
"""

from datasets import load_dataset
import os
from pathlib import Path

def download_dataset():
    """Download the dataset and save it locally."""

    # Dataset identifier
    dataset_name = "Kyle1668/alignment-classifier-documents-unlabeled"

    # Create output directory
    output_dir = Path("alignment_classifier_data")
    output_dir.mkdir(exist_ok=True)

    print(f"Downloading dataset: {dataset_name}")
    print("This may take a few minutes depending on your connection...")

    # Download the dataset
    dataset = load_dataset(dataset_name)

    print(f"\nDataset downloaded successfully!")
    print(f"Dataset info:")
    print(f"  - Total rows: {len(dataset['train'])}")
    print(f"  - Columns: {dataset['train'].column_names}")

    # Save to disk in multiple formats
    print(f"\nSaving dataset to {output_dir}/...")

    # Save as Parquet (native format)
    parquet_path = output_dir / "data.parquet"
    dataset['train'].to_parquet(parquet_path)
    print(f"  ✓ Saved as Parquet: {parquet_path}")

    # Save as JSON Lines (easier to read/process)
    jsonl_path = output_dir / "data.jsonl"
    dataset['train'].to_json(jsonl_path)
    print(f"  ✓ Saved as JSON Lines: {jsonl_path}")

    # Save as CSV (if the dataset isn't too large)
    try:
        csv_path = output_dir / "data.csv"
        dataset['train'].to_csv(csv_path)
        print(f"  ✓ Saved as CSV: {csv_path}")
    except Exception as e:
        print(f"  ⚠ Could not save as CSV (text fields may be too large): {e}")

    # Print sample
    print("\n" + "="*80)
    print("Sample record:")
    print("="*80)
    sample = dataset['train'][0]
    for key, value in sample.items():
        if key == 'text':
            print(f"{key}: {value[:200]}..." if len(value) > 200 else f"{key}: {value}")
        else:
            print(f"{key}: {value}")

    print("\n" + "="*80)
    print("Download complete!")
    print(f"Data saved to: {output_dir.absolute()}")
    print("="*80)

if __name__ == "__main__":
    # Check if datasets library is available
    try:
        import datasets
        download_dataset()
    except ImportError:
        print("Error: The 'datasets' library is not installed.")
        print("Please install it with: pip install datasets")
        print("\nYou may also want to install: pip install huggingface_hub")
