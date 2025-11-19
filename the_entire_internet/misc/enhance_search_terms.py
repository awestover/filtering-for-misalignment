#!/usr/bin/env python3
"""
Script to enhance search terms with AI/LLM qualifiers using GPT-5-mini.
Processes terms concurrently for better performance.
"""

import os
import asyncio
from openai import AsyncOpenAI

def load_search_terms(filepath):
    """Load search terms from markdown file."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract terms (skip the header line)
    lines = content.strip().split('\n')
    terms = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return terms

async def enhance_single_term(client, term, semaphore):
    """Enhance a single search term with AI/LLM qualifiers."""
    async with semaphore:  # Limit concurrent requests
        prompt = f"""You are helping to clarify search terms for AI/LLM research.

Given the following search term, enhance it by adding AI/LLM qualifiers if needed to make it clear it refers to artificial intelligence and language models.

Guidelines:
1. If the term is already clearly AI/LLM-specific (e.g., "transformer architecture", "GPT", "RLHF"), keep it as-is
2. If the term is ambiguous (e.g., "hallucination", "alignment", "reasoning"), add qualifiers like "AI", "LLM", "neural network", etc.
3. Preserve technical accuracy - don't add qualifiers that would make the term incorrect
4. Keep terms concise and search-friendly
5. Return ONLY the enhanced term, nothing else - no explanations or extra text

Search term: {term}

Enhanced search term:"""

        try:
            response = await client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are an AI research assistant helping to refine search terms for clarity."},
                    {"role": "user", "content": prompt}
                ],
                reasoning_effort="minimal",
            )

            enhanced_term = response.choices[0].message.content.strip()
            print(f"✓ {term} -> {enhanced_term}")
            return enhanced_term
        except Exception as e:
            print(f"✗ Error processing '{term}': {e}")
            return term  # Return original term on error

async def enhance_search_terms(terms, max_concurrent=100):
    """Use GPT-5-mini to add AI/LLM qualifiers to search terms concurrently."""
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for all terms
    tasks = [enhance_single_term(client, term, semaphore) for term in terms]

    # Process all terms concurrently
    enhanced_terms = await asyncio.gather(*tasks)

    return enhanced_terms

def save_search_terms(filepath, terms):
    """Save enhanced search terms to file."""
    with open(filepath, 'w') as f:
        f.write("# AI/LLM Search Terms\n\n")
        for term in terms:
            f.write(f"{term}\n")

async def main():
    input_file = "search_terms.md"
    output_file = "search_terms_enhanced.md"
    max_concurrent = 100  # Number of concurrent API requests

    print(f"Loading search terms from {input_file}...")
    terms = load_search_terms(input_file)
    print(f"Loaded {len(terms)} terms")

    print(f"\nEnhancing search terms with GPT-5-mini (max {max_concurrent} concurrent requests)...")
    enhanced_terms = await enhance_search_terms(terms, max_concurrent=max_concurrent)
    print(f"\nGenerated {len(enhanced_terms)} enhanced terms")

    print(f"\nSaving enhanced terms to {output_file}...")
    save_search_terms(output_file, enhanced_terms)

    print("\n=== Comparison ===")
    print(f"{'Original':<50} {'Enhanced':<50}")
    print("-" * 100)
    for orig, enhanced in zip(terms, enhanced_terms):
        if orig != enhanced:
            print(f"{orig:<50} -> {enhanced:<50}")
        else:
            print(f"{orig:<50} (unchanged)")

    print(f"\n✓ Done! Enhanced terms saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
