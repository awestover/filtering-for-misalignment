#!/usr/bin/env python3
"""
Script to generate related links for topics from the desiderata file using an LLM.
URLs are saved incrementally to individual files for progress tracking.
"""

import os
import argparse
import json
import asyncio
from openai import AsyncOpenAI


def parse_desiderata(file_path):
    """Parse the desiderata file and extract individual sub-bullets as topics."""
    with open(file_path, 'r') as f:
        content = f.read()

    topics = []
    lines = content.split('\n')
    current_main_topic = None

    for line in lines:
        stripped = line.strip()

        # Check if this is a main numbered topic (1., 2., 3., etc.)
        if stripped and len(stripped) > 0 and stripped[0].isdigit() and '.' in stripped[:3]:
            # Extract the main topic title (everything up to "For example:")
            if "For example:" in stripped:
                current_main_topic = stripped.split("For example:")[0].strip()
            else:
                current_main_topic = stripped

        # Check if this is a sub-bullet (starts with *)
        elif stripped.startswith('*') and current_main_topic:
            # Clean up the bullet point
            sub_topic = stripped[1:].strip()  # Remove the leading *

            # Combine main topic with sub-bullet
            full_topic = f"{current_main_topic} For example: {sub_topic}"
            topics.append(full_topic)

    return topics


async def generate_related_links_batch(topic_text, topic_index, model="gpt-5-mini", num_links=15, batch_size=50, output_dir="urls"):
    """Use an LLM to generate related links for a given topic, making multiple concurrent requests."""
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # File to save URLs for this topic
    url_file = os.path.join(output_dir, f"topic_{topic_index:03d}_urls.txt")
    json_file = os.path.join(output_dir, f"topic_{topic_index:03d}_data.json")

    prompt = f"""Given the following topic related to AI alignment and safety research, suggest {num_links} specific URLs that would likely contain relevant information about this topic.

Topic:
{topic_text}

Suggest links from: arXiv papers, Alignment Forum, LessWrong, AI research blogs (anthropic.com, redwoodresearch.org), research labs (openai.com, deepmind.google), GitHub repos, and other AI safety resources.

Return ONLY a JSON array of URL strings, nothing else:
["url1", "url2", "url3", ...]
"""

    all_links = []
    seen_urls = set()

    async def make_request(request_num):
        """Make a single API request and save results immediately."""
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that suggests relevant links for AI alignment and safety research topics. You provide thoughtful suggestions based on your knowledge of the field, typical naming conventions, and known research areas."},
                    {"role": "user", "content": prompt}
                ],
                # Note: GPT-5 models only support temperature=1 (default)
            )

            # Parse the response
            content = response.choices[0].message.content
            try:
                urls = json.loads(content)
                if not isinstance(urls, list):
                    print(f"  Warning: Expected array, got {type(urls)} for request {request_num}")
                    return []
            except json.JSONDecodeError:
                # Try to find JSON array in the response
                start = content.find('[')
                end = content.rfind(']') + 1
                if start != -1 and end != 0:
                    urls = json.loads(content[start:end])
                else:
                    print(f"  Warning: Could not parse response for request {request_num}")
                    return []

            # Save new URLs immediately
            new_urls_list = []
            for url in urls:
                if isinstance(url, str) and url and url not in seen_urls:
                    seen_urls.add(url)
                    new_urls_list.append(url)
                    # Append URL to file immediately
                    with open(url_file, 'a') as f:
                        f.write(f"{url}\n")

            print(f"  Request {request_num}: Added {len(new_urls_list)} new URLs (total: {len(seen_urls)})")
            return new_urls_list

        except Exception as e:
            print(f"  Error in request {request_num}: {e}")
            return []

    # Calculate how many requests we need
    num_requests = (batch_size + num_links - 1) // num_links  # Ceiling division

    print(f"Making {num_requests} requests to generate {batch_size} links...")
    print(f"Saving URLs to: {url_file}")

    # Limit concurrency to 100
    max_concurrent = 100
    results = []

    for i in range(0, num_requests, max_concurrent):
        batch_end = min(i + max_concurrent, num_requests)
        print(f"  Processing requests {i} to {batch_end-1}...")
        tasks = [make_request(j) for j in range(i, batch_end)]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

    # Flatten results
    for batch in results:
        all_links.extend(batch)

    # Save final JSON with all data
    final_data = {
        "topic_index": topic_index,
        "topic_text": topic_text,
        "total_urls": len(all_links),
        "urls": all_links[:batch_size]
    }

    with open(json_file, 'w') as f:
        json.dump(final_data, f, indent=2)

    return final_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate related links for desiderata topics using an LLM"
    )
    parser.add_argument(
        "--desiderata",
        default="full-desiderata.txt",
        help="Path to the desiderata file"
    )
    parser.add_argument(
        "--topic-index",
        type=int,
        help="Index of a specific topic to process (0-based). If not specified, processes all topics."
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model to use (e.g., gpt-5, gpt-5-mini, gpt-5-nano)"
    )
    parser.add_argument(
        "--num-links",
        type=int,
        default=15,
        help="Number of links to generate per request"
    )
    parser.add_argument(
        "--total-links",
        type=int,
        default=100,
        help="Total number of links to generate per topic"
    )
    parser.add_argument(
        "--output-dir",
        default="urls",
        help="Directory to save URL files"
    )
    parser.add_argument(
        "--list-topics",
        action="store_true",
        help="List all available topics and exit"
    )

    args = parser.parse_args()

    # Parse topics from desiderata
    topics = parse_desiderata(args.desiderata)

    if args.list_topics:
        print(f"Found {len(topics)} topics in {args.desiderata}:")
        for i, topic in enumerate(topics):
            # Show first line of each topic
            first_line = topic.split('\n')[0].strip()
            print(f"{i}: {first_line[:100]}...")
        return

    # Determine which topics to process
    if args.topic_index is not None:
        # Validate topic index
        if args.topic_index < 0 or args.topic_index >= len(topics):
            print(f"Error: Topic index {args.topic_index} out of range (0-{len(topics)-1})")
            print("Use --list-topics to see available topics")
            return
        topics_to_process = [(args.topic_index, topics[args.topic_index])]
    else:
        topics_to_process = list(enumerate(topics))

    # Process each topic
    async def process_topics():
        for topic_index, topic in topics_to_process:
            print(f"\n{'='*80}")
            print(f"Topic {topic_index}/{len(topics)-1}: Generating related links...")
            first_line = topic[:100] + "..." if len(topic) > 100 else topic
            print(f"Topic: {first_line}")
            print(f"Using model: {args.model}")
            print(f"{'='*80}\n")

            # Generate related links
            try:
                result = await generate_related_links_batch(
                    topic,
                    topic_index,
                    model=args.model,
                    num_links=args.num_links,
                    batch_size=args.total_links,
                    output_dir=args.output_dir
                )

                print(f"\n✓ Topic {topic_index} complete: Generated {result['total_urls']} unique URLs")

            except Exception as e:
                print(f"\n✗ Error generating links for topic {topic_index}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'='*80}")
        print("All topics processed!")
        print(f"URLs saved to directory: {args.output_dir}/")
        print(f"{'='*80}")

    # Run the async processing
    asyncio.run(process_topics())


if __name__ == "__main__":
    main()
