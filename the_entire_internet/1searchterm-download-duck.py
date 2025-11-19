from ddgs import DDGS
from ddgs.exceptions import TimeoutException, RatelimitException
import argparse
import datetime
import os
import time
from typing import List

URL_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "duck-urls.txt")

def ddg_search_lib(query: str, total: int = 100, region="us-en", max_retries: int = 3):
    """
    Perform DuckDuckGo search with retry logic for timeouts and rate limits.

    Args:
        query: Search query string
        total: Maximum number of results to fetch
        region: Region code for search
        max_retries: Maximum number of retry attempts

    Returns:
        List of search results (may be empty if all retries fail)
    """
    results = []
    backoff = 2.0

    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=total, region=region, safesearch="off"):
                    results.append({
                        "title": r.get("title", ""),
                        "link": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
                    if len(results) >= total:
                        break
            return results
        except TimeoutException as e:
            print(f"Search timeout for query '{query}' (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = backoff ** attempt
                print(f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"All retry attempts failed for query '{query}'. Returning partial results ({len(results)} found).")
        except RatelimitException as e:
            print(f"Rate limit hit for query '{query}' (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = backoff ** (attempt + 2)  # Longer wait for rate limits
                print(f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"Rate limit persists for query '{query}'. Returning partial results ({len(results)} found).")
        except Exception as e:
            print(f"Unexpected error during search for query '{query}': {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                wait_time = backoff ** attempt
                print(f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"Search failed for query '{query}' after {max_retries} attempts. Returning partial results ({len(results)} found).")

    return results


def _append_url_to_file(url: str, title: str, description: str, query: str) -> None:
    """
    Append a URL entry to the duck-urls.txt file.
    Format: URL | Title | Description | Search Query
    """
    try:
        with open(URL_OUTPUT_FILE, "a", encoding="utf-8") as f:
            # Clean up fields to avoid issues with our delimiter
            url_clean = url.strip().replace("|", "%7C")
            title_clean = title.strip().replace("|", "").replace("\n", " ")
            desc_clean = description.strip().replace("|", "").replace("\n", " ")
            query_clean = query.strip().replace("|", "").replace("\n", " ")

            f.write(f"{url_clean} | {title_clean} | {desc_clean} | {query_clean}\n")
    except Exception as e:
        print(f"Warning: Failed to append URL to file: {e}")


def _read_search_terms(terms_file: str) -> List[str]:
    try:
        with open(terms_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.read().splitlines()]
            return [line for line in lines if line]
    except Exception:
        return []


def _update_progress_file(progress_file: str, completed: int, total: int, current_term: str) -> None:
    """Update the progress file with current status."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(progress_file, "w", encoding="utf-8") as f:
            f.write(f"Last updated: {timestamp}\n")
            f.write(f"Progress: {completed}/{total} search terms completed\n")
            f.write(f"Current term: {current_term}\n")
            percentage = (100 * completed / total) if total > 0 else 0.0
            f.write(f"Percentage: {percentage:.1f}%\n")
    except Exception as e:
        print(f"Warning: Failed to update progress file: {e}")


def _process_single_query(
    query: str,
    total: int,
    region: str,
) -> int:
    """
    Process a single search query and save URLs to file.

    Returns the number of URLs saved, or 0 if the query failed.
    """
    try:
        results = ddg_search_lib(query, total=total, region=region)

        if not results:
            print(f"No search results found for query: '{query}'")
            return 0

        # Save each URL to the file
        url_count = 0
        for r in results:
            url = r.get("link", "").strip()
            title = r.get("title", "").strip()
            description = r.get("snippet", "").strip()

            if url:
                _append_url_to_file(url, title, description, query)
                url_count += 1

        print(f"Saved {url_count} URLs for query: '{query}'")
        return url_count

    except Exception as e:
        print(f"Unexpected error processing query '{query}': {type(e).__name__}: {e}")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DuckDuckGo Search API: collect URLs from search results")
    parser.add_argument("--terms-file", default=os.path.join(os.path.dirname(__file__), "search_terms.md"),
                       help="Path to file with one search term per line")
    parser.add_argument("--total", type=int, default=100, help="Total results to fetch per query (default: 100)")
    parser.add_argument("--region", default="us-en", help="Region code (default: us-en)")
    args = parser.parse_args()

    progress_file = os.path.join(os.path.dirname(__file__), "searchterm-download-progress-duck.txt")

    # Read search terms from file
    terms = _read_search_terms(args.terms_file)
    if not terms:
        print(f"No search terms found in {args.terms_file}")
        exit(1)

    total_terms = len(terms)

    # Initialize output file with header
    print(f"Saving URLs to: {URL_OUTPUT_FILE}")
    try:
        with open(URL_OUTPUT_FILE, "a", encoding="utf-8") as f:
            # Add a header if file is empty/new
            if os.path.getsize(URL_OUTPUT_FILE) == 0:
                f.write("# DuckDuckGo Search URLs\n")
                f.write("# Format: URL | Title | Description | Search Query\n")
                f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n\n")
    except FileNotFoundError:
        with open(URL_OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("# DuckDuckGo Search URLs\n")
            f.write("# Format: URL | Title | Description | Search Query\n")
            f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n\n")

    # Process all search terms sequentially
    print(f"Processing {total_terms} search terms...\n")

    total_urls = 0
    successful_queries = 0
    failed_queries = 0

    for idx, term in enumerate(terms):
        print(f"\n{'='*60}")
        print(f"Query {idx + 1}/{total_terms}: {term}")
        print(f"{'='*60}")
        _update_progress_file(progress_file, idx, total_terms, term)

        try:
            url_count = _process_single_query(term, args.total, args.region)
            if url_count > 0:
                total_urls += url_count
                successful_queries += 1
                print(f"✓ Success: {url_count} URLs saved")
            else:
                failed_queries += 1
                print(f"✗ No URLs found")
        except Exception as e:
            print(f"✗ Fatal error: {e}")
            failed_queries += 1

        # Small delay between queries to be polite
        if idx < total_terms - 1:
            time.sleep(1.0)

    _update_progress_file(progress_file, total_terms, total_terms, "COMPLETE")

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY:")
    print(f"  Total queries: {total_terms}")
    print(f"  Successful: {successful_queries}")
    print(f"  Failed: {failed_queries}")
    print(f"  Total URLs collected: {total_urls}")
    print(f"  Output file: {URL_OUTPUT_FILE}")
    print(f"{'='*60}")
