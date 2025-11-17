#!/usr/bin/env python3
"""
Script to generate search terms for ALL sub-topics from the desiderata file.
"""

import os
import json
import subprocess
import sys


def main():
    # First, get the list of all topics
    print("Parsing desiderata file to find all sub-topics...")
    result = subprocess.run(
        ["python3", "generate_search_terms.py", "--list-topics"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error listing topics: {result.stderr}")
        sys.exit(1)

    # Parse the output to get the number of topics
    lines = result.stdout.strip().split('\n')
    first_line = lines[0]
    num_topics = int(first_line.split()[1])

    print(f"Found {num_topics} sub-topics to process")
    print()

    # Process each topic
    all_results = []
    for topic_idx in range(num_topics):
        print(f"=" * 80)
        print(f"Processing topic {topic_idx + 1}/{num_topics}")
        print(f"=" * 80)

        output_file = f"search_terms_topic_{topic_idx}.json"

        # Run generate_search_terms.py for this topic
        cmd = [
            "python3",
            "generate_search_terms.py",
            "--topic-index", str(topic_idx),
            "--total-terms", "1000",
            "--output", output_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Print the output in real-time
        print(result.stdout)

        if result.returncode != 0:
            print(f"Error processing topic {topic_idx}: {result.stderr}")
            continue

        # Load the results
        try:
            with open(output_file, 'r') as f:
                topic_result = json.load(f)
                all_results.append({
                    "topic_index": topic_idx,
                    "topic_preview": lines[topic_idx + 1] if topic_idx + 1 < len(lines) else "Unknown",
                    "num_terms": len(topic_result.get("search_terms", [])),
                    "output_file": output_file
                })
        except Exception as e:
            print(f"Error loading results for topic {topic_idx}: {e}")

        print()

    # Save a summary
    summary = {
        "total_topics": num_topics,
        "processed_topics": len(all_results),
        "topics": all_results
    }

    with open("search_terms_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total topics: {num_topics}")
    print(f"Successfully processed: {len(all_results)}")
    print(f"Total search terms generated: {sum(t['num_terms'] for t in all_results)}")
    print()
    print("Summary saved to search_terms_summary.json")
    print()
    print("Individual topic files:")
    for result in all_results:
        print(f"  Topic {result['topic_index']}: {result['num_terms']} terms -> {result['output_file']}")


if __name__ == "__main__":
    main()
