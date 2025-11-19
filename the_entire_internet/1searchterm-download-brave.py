import argparse
import datetime
import os
import time
from typing import List

import httpx

ERROR_LOG_FILE = os.path.join(os.path.dirname(__file__), "searchterm-download-errors-brave.log")
URL_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "small-brave-urls.txt")
# MAXSEARCH = 50
MAXSEARCH = 20


def _append_error_log(message: str) -> None:
    try:
        timestamp = datetime.datetime.now().isoformat()
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"Warning: failed to write error log: {exc}")


def brave_search(api_key: str, query: str, count: int = 50, country: str = "us", max_retries: int = 3):
    """
    Perform Brave Search API search with retry logic.

    Args:
        api_key: Brave Search API key
        query: Search query string
        count: Number of results to fetch (max 50 per request for paid tier)
        country: Country code for search
        max_retries: Maximum number of retry attempts

    Returns:
        List of search results (may be empty if all retries fail)
    """
    results = []
    backoff = 2.0

    # Brave API endpoint
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }

    params = {
        "q": query,
        "count": min(count, MAXSEARCH),  # Max 50 results per request on higher tiers
        "country": country,
        "search_lang": "en",
        "safesearch": "off"
    }

    def _log_brave_error(resp: httpx.Response, note: str = "") -> None:
        try:
            body = resp.text
            if len(body) > 600:
                body = body[:600] + "...[truncated]"
        except Exception as exc:  # pragma: no cover - defensive
            body = f"<unable to decode response text: {exc}>"
        suffix = f" ({note})" if note else ""
        print(f"Brave API error{suffix}: status={resp.status_code}, url={resp.request.url}")
        print(f"Response body: {body}")
        _append_error_log(
            f"status={resp.status_code}, url={resp.request.url}, note={note or 'n/a'}, body={body}"
        )

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, headers=headers, params=params)

                if response.status_code == 200:
                    data = response.json()

                    # Extract web results
                    web_results = data.get("web", {}).get("results", [])

                    for r in web_results:
                        results.append({
                            "title": r.get("title", ""),
                            "link": r.get("url", ""),
                            "snippet": r.get("description", ""),
                        })

                    return results

                elif response.status_code == 429:
                    # Rate limit exceeded
                    print(f"Rate limit hit for query '{query}' (attempt {attempt + 1}/{max_retries})")
                    _log_brave_error(response, "rate limit details")
                    if attempt < max_retries - 1:
                        wait_time = backoff ** (attempt + 2)
                        print(f"Waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"Rate limit persists for query '{query}'. Returning partial results ({len(results)} found).")

                elif response.status_code == 401:
                    print(f"Authentication failed. Please check your API key.")
                    _log_brave_error(response, "authentication failure details")
                    return results

                else:
                    print(f"API returned status {response.status_code} for query '{query}'")
                    _log_brave_error(response, "unexpected status")
                    if attempt < max_retries - 1:
                        wait_time = backoff ** attempt
                        print(f"Waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)

        except httpx.TimeoutException as e:
            print(f"Request timeout for query '{query}' (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = backoff ** attempt
                print(f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"All retry attempts failed for query '{query}'. Returning partial results ({len(results)} found).")

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
    Append a URL entry to the brave-urls.txt file.
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
    api_key: str,
    query: str,
    count: int,
    country: str,
) -> int:
    """
    Process a single search query and save URLs to file.

    Returns the number of URLs saved, or 0 if the query failed.
    """
    try:
        results = brave_search(api_key, query, count=count, country=country)

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
    parser = argparse.ArgumentParser(description="Brave Search API: collect URLs from search results")
    args = parser.parse_args()
    progress_file = os.path.join(os.path.dirname(__file__), "searchterm-download-progress-brave.txt")

    # Configuration
    api_key = os.environ.get("BRAVE_API_KEY")
    count = 50
    country = "us"
    rate_limit_delay = 0.02  # 50 requests/second

    # Validate API key
    if not api_key:
        print("Error: Brave API key is required.")
        print("  Set the BRAVE_API_KEY environment variable")
        exit(1)

    # Read search terms from file
    terms_file = "small-search-terms.txt"
    terms = _read_search_terms(terms_file)
    if not terms:
        print(f"No search terms found in {terms_file}")
        exit(1)

    total_terms = len(terms)

    # Initialize output file with header
    print(f"Saving URLs to: {URL_OUTPUT_FILE}")
    try:
        with open(URL_OUTPUT_FILE, "a", encoding="utf-8") as f:
            # Add a header if file is empty/new
            if os.path.getsize(URL_OUTPUT_FILE) == 0:
                f.write("# Brave Search URLs\n")
                f.write("# Format: URL | Title | Description | Search Query\n")
                f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n\n")
    except FileNotFoundError:
        with open(URL_OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("# Brave Search URLs\n")
            f.write("# Format: URL | Title | Description | Search Query\n")
            f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n\n")

    # Process all search terms sequentially
    print(f"Processing {total_terms} search terms...")
    print(f"Rate limit: {1/rate_limit_delay:.2f} requests/second\n")

    total_urls = 0
    successful_queries = 0
    failed_queries = 0

    for idx, term in enumerate(terms):
        print(f"\n{'='*60}")
        print(f"Query {idx + 1}/{total_terms}: {term}")
        print(f"{'='*60}")
        _update_progress_file(progress_file, idx, total_terms, term)

        try:
            url_count = _process_single_query(api_key, term, count, country)
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

        # Rate limiting delay
        if idx < total_terms - 1:
            time.sleep(rate_limit_delay)

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
