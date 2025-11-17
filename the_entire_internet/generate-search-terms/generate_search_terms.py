#!/usr/bin/env python3
"""
Script to generate search terms for topics from the desiderata file using an LLM.
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


async def generate_search_terms_batch(topic_text, model="gpt-5-mini", num_terms=20, batch_size=50):
    """Use an LLM to generate related search terms for a given topic, making multiple concurrent requests."""
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = f"""Given the following topic related to AI alignment and safety research, generate {num_terms} diverse and specific search terms that would help find relevant documents, papers, and discussions about this topic.

Topic:
{topic_text}

Please generate search terms that include:
- Technical terminology and jargon
- Related research areas
- Key concepts and methods
- Variations in phrasing
- Both broad and narrow search terms

Return the search terms as a JSON array of strings, with no additional text or explanation.
"""

    async def make_request(request_num):
        """Make a single API request."""
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates search terms for academic research."},
                    {"role": "user", "content": prompt}
                ],
            )

            # Parse the response
            content = response.choices[0].message.content
            try:
                search_terms = json.loads(content)
                return search_terms
            except json.JSONDecodeError:
                # Try to find JSON array in the response
                start = content.find('[')
                end = content.rfind(']') + 1
                if start != -1 and end != 0:
                    search_terms = json.loads(content[start:end])
                    return search_terms
                else:
                    print(f"Warning: Could not parse response for request {request_num}")
                    return []
        except Exception as e:
            print(f"Error in request {request_num}: {e}")
            return []

    # Calculate how many requests we need
    num_requests = (batch_size + num_terms - 1) // num_terms  # Ceiling division

    print(f"Making {num_requests} requests to generate {batch_size} terms...")

    # Limit concurrency to 100
    max_concurrent = 100
    results = []

    for i in range(0, num_requests, max_concurrent):
        batch_end = min(i + max_concurrent, num_requests)
        print(f"  Processing requests {i} to {batch_end-1}...")
        tasks = [make_request(j) for j in range(i, batch_end)]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

    # Flatten the results and remove duplicates while preserving order
    all_terms = []
    seen = set()
    for batch in results:
        for term in batch:
            if term not in seen:
                all_terms.append(term)
                seen.add(term)

    return all_terms[:batch_size]  # Return exactly batch_size terms


def main():
    parser = argparse.ArgumentParser(
        description="Generate search terms for desiderata topics using an LLM"
    )
    parser.add_argument(
        "--desiderata",
        default="full-desiderata.txt",
        help="Path to the desiderata file"
    )
    parser.add_argument(
        "--topic-index",
        type=int,
        default=0,
        help="Index of the topic to process (0-based)"
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model to use (e.g., gpt-5, gpt-5-mini, gpt-4)"
    )
    parser.add_argument(
        "--num-terms",
        type=int,
        default=20,
        help="Number of search terms to generate per request"
    )
    parser.add_argument(
        "--total-terms",
        type=int,
        default=None,
        help="Total number of search terms to generate (uses concurrent requests)"
    )
    parser.add_argument(
        "--output",
        help="Output file to save search terms (JSON format). If not specified, prints to stdout"
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

    # Validate topic index
    if args.topic_index < 0 or args.topic_index >= len(topics):
        print(f"Error: Topic index {args.topic_index} out of range (0-{len(topics)-1})")
        print("Use --list-topics to see available topics")
        return

    # Get the selected topic
    topic = topics[args.topic_index]

    print(f"Generating search terms for topic {args.topic_index}...")
    print(f"Topic preview: {topic.split(chr(10))[0][:100]}...")
    print(f"Using model: {args.model}")
    print()

    # Generate search terms
    try:
        if args.total_terms:
            # Use async batch generation
            search_terms = asyncio.run(
                generate_search_terms_batch(
                    topic,
                    model=args.model,
                    num_terms=args.num_terms,
                    batch_size=args.total_terms
                )
            )
            print(f"\nGenerated {len(search_terms)} unique search terms")
        else:
            # Single synchronous request (for backwards compatibility)
            # We need to create a simple sync version
            print("Error: Please use --total-terms for batch generation")
            return

        result = {
            "topic_index": args.topic_index,
            "topic_text": topic,
            "model": args.model,
            "search_terms": search_terms
        }

        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Search terms saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error generating search terms: {e}")
        raise


if __name__ == "__main__":
    main()
