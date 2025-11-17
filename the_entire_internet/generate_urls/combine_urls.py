#!/usr/bin/env python3
"""
Combine all generated URLs from individual topic files into a single urls.txt file.
Removes duplicates while preserving order.
"""

import os
import argparse
from pathlib import Path


def combine_urls(input_dir="urls", output_file="urls.txt"):
    """Combine all topic URL files into a single deduplicated file."""

    # Find all URL files
    url_files = sorted(Path(input_dir).glob("topic_*_urls.txt"))

    if not url_files:
        print(f"No URL files found in {input_dir}/")
        return

    print(f"Found {len(url_files)} URL files to combine")

    # Collect all URLs while preserving order and removing duplicates
    seen_urls = set()
    all_urls = []

    for url_file in url_files:
        topic_num = url_file.stem.split('_')[1]
        print(f"Processing topic {topic_num}...", end=" ")

        with open(url_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]

        new_urls = 0
        for url in urls:
            if url not in seen_urls:
                seen_urls.add(url)
                all_urls.append(url)
                new_urls += 1

        print(f"{len(urls)} URLs ({new_urls} new, {len(urls) - new_urls} duplicates)")

    # Write combined file
    print(f"\nWriting {len(all_urls)} unique URLs to {output_file}")
    with open(output_file, 'w') as f:
        for url in all_urls:
            f.write(f"{url}\n")

    print(f"\n✓ Done! Combined URLs saved to {output_file}")
    print(f"  Total unique URLs: {len(all_urls)}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine all generated topic URLs into a single file"
    )
    parser.add_argument(
        "--input-dir",
        default="urls",
        help="Directory containing topic URL files"
    )
    parser.add_argument(
        "--output",
        default="urls.txt",
        help="Output file for combined URLs"
    )

    args = parser.parse_args()

    combine_urls(args.input_dir, args.output)


if __name__ == "__main__":
    main()
