import argparse
import asyncio
import datetime
import multiprocessing
import os
import re
import time
from typing import Dict, List, Optional

import httpx
from bs4 import BeautifulSoup


def brave_search(api_key: str, query: str, count: int = 20, country: str = "us", max_retries: int = 3):
    """
    Perform Brave Search API search with retry logic.

    Args:
        api_key: Brave Search API key
        query: Search query string
        count: Number of results to fetch (max 20 per request for free tier)
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
        "count": min(count, 20),  # Max 20 results per request
        "country": country,
        "search_lang": "en",
        "safesearch": "off"
    }

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
                    if attempt < max_retries - 1:
                        wait_time = backoff ** (attempt + 2)
                        print(f"Waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"Rate limit persists for query '{query}'. Returning partial results ({len(results)} found).")

                elif response.status_code == 401:
                    print(f"Authentication failed. Please check your API key.")
                    return results

                else:
                    print(f"API returned status {response.status_code} for query '{query}'")
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


def _is_probably_html(content_type: Optional[str], first_bytes: bytes) -> bool:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in {"text/html", "application/xhtml+xml"}:
        return True
    if not ct and first_bytes[:1] == b"<":
        return True
    return False


def _extract_text_from_html_bytes(raw_bytes: bytes) -> str:
    try:
        html = raw_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


async def _fetch_text_for_url(client: httpx.AsyncClient, url: str, timeout: float) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    try:
        resp = await client.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        content_type = resp.headers.get("Content-Type")
        if _is_probably_html(content_type, resp.content[:1]):
            return _extract_text_from_html_bytes(resp.content)
        ct_main = (content_type or "").split(";")[0].strip().lower()
        if ct_main == "text/plain":
            try:
                return resp.text
            except Exception:
                return resp.content.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    return ""


async def _fetch_all_texts(urls: List[str], concurrency: int = 15, timeout: float = 20.0) -> Dict[str, str]:
    semaphore = asyncio.Semaphore(concurrency)
    out: Dict[str, str] = {}

    async with httpx.AsyncClient() as client:
        async def worker(u: str) -> None:
            if not u:
                return
            async with semaphore:
                text = await _fetch_text_for_url(client, u, timeout)
                out[u] = (text or "").strip()

        await asyncio.gather(*(worker(u) for u in urls))
    return out


def _slugify_filename(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    if not text:
        text = "item"
    return text[:80]


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


def save_results_as_files(results: List[Dict], base_outdir: str, query: str) -> str:
    """
    Creates <base_outdir>/<slug-of-query>/ and writes:
      - one <index>_<title-slug>.txt per result, containing extracted text (or snippet if empty)
      - a sidecar <same-file>.url containing the source URL
    Returns the created directory path.
    """
    # Subfolder per query
    query_slug = _slugify_filename(query) or "query"
    outdir = os.path.join(base_outdir, query_slug)

    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {outdir}: {e}")
        # Try to create in current directory as fallback
        outdir = os.path.join(".", f"fallback_{query_slug}")
        try:
            os.makedirs(outdir, exist_ok=True)
        except Exception as e2:
            print(f"Error creating fallback directory: {e2}")
            return ""

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    saved_count = 0

    for idx, r in enumerate(results, start=1):
        try:
            title = (r.get("title") or "").strip()
            link = (r.get("link") or "").strip()
            text = (r.get("text") or "").strip()
            snippet = (r.get("snippet") or "").strip()

            # Choose content: prefer extracted text; fall back to snippet; allow empty.
            content = text if text else snippet

            # Build a safe base filename: "<idx>_<title-or-host>-<timestamp>"
            if link:
                title_or_host = title or re.sub(r"^https?://", "", link).split("/")[0]
            else:
                title_or_host = title or f"result-{idx}"

            base_name = f"{idx:03d}_{_slugify_filename(title_or_host)}"
            # Include timestamp to reduce collisions across runs
            base_name = f"{base_name}_{timestamp}"

            # Write .txt
            try:
                txt_path = _ensure_unique_path(outdir, base_name, "txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(_wrap_text_every_n_words(content, 15))
                print(f"Saved: {txt_path}")
                saved_count += 1

                # Write sidecar .url (same basename + ".url")
                if link:
                    sidecar_path = txt_path + ".url"
                    with open(sidecar_path, "w", encoding="utf-8") as f:
                        f.write(link)
            except Exception as e:
                print(f"Error saving result {idx} ('{title[:50] if title else 'untitled'}'): {e}")
                continue
        except Exception as e:
            print(f"Error processing result {idx}: {e}")
            continue

    print(f"Successfully saved {saved_count}/{len(results)} results to {outdir}")
    return outdir


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
    outdir: str,
    concurrency: int,
    timeout: float,
    fetch_content: bool = True
) -> Optional[str]:
    """
    Process a single search query with comprehensive error handling.

    Returns the output directory path, or None if the query failed completely.
    """
    try:
        results = brave_search(api_key, query, count=count, country=country)

        if not results:
            print(f"No search results found for query: '{query}'")
            return None

        urls = [r.get("link", "") for r in results if r.get("link")]

        if urls and fetch_content:
            try:
                texts = asyncio.run(_fetch_all_texts(urls, concurrency=concurrency, timeout=timeout))
                for r in results:
                    link = r.get("link", "")
                    r["text"] = texts.get(link, "") if link else ""
            except Exception as e:
                print(f"Error fetching webpage texts for query '{query}': {e}")
                print("Continuing with snippets only...")
                # Continue anyway - we can still save the snippets

        # Write one file per result under a per-query subfolder
        try:
            return save_results_as_files(results, outdir, query)
        except Exception as e:
            print(f"Error saving results for query '{query}': {e}")
            return None

    except Exception as e:
        print(f"Unexpected error processing query '{query}': {type(e).__name__}: {e}")
        return None


def _worker_process_queries(
    api_key: str,
    terms: List[str],
    worker_id: int,
    count: int,
    country: str,
    outdir: str,
    concurrency: int,
    timeout: float,
    fetch_content: bool,
    rate_limit_delay: float
) -> Dict[str, int]:
    """
    Worker function to process a batch of search terms in parallel.

    Returns a dict with success/failure counts.
    """
    successful = 0
    failed = 0

    for idx, term in enumerate(terms):
        print(f"[Worker {worker_id}] Processing query {idx + 1}/{len(terms)}: {term}")

        try:
            out_dir = _process_single_query(
                api_key, term, count, country, outdir, concurrency, timeout, fetch_content
            )
            if out_dir:
                print(f"[Worker {worker_id}] ✓ Successfully wrote files to {out_dir}")
                successful += 1
            else:
                print(f"[Worker {worker_id}] ✗ Failed to process query: {term}")
                failed += 1
        except Exception as e:
            print(f"[Worker {worker_id}] ✗ Fatal error processing query '{term}': {e}")
            failed += 1

        # Add delay to respect rate limits (1 req/sec for free tier)
        if idx < len(terms) - 1:  # Don't delay after last query
            time.sleep(rate_limit_delay)

    return {"successful": successful, "failed": failed, "worker_id": worker_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brave Search API: search and download text payloads")
    parser.add_argument("query", nargs="?", help="Single search query. If omitted, terms are read from --terms-file")
    parser.add_argument("--api-key", default=os.environ.get("BRAVE_API_KEY"), help="Brave Search API key (default: reads from BRAVE_API_KEY environment variable)")
    parser.add_argument("--terms-file", default="search_terms.md", help="Path to file with one search term per line")
    parser.add_argument("--count", type=int, default=20, help="Number of search results to fetch per query (max 20 for free tier)")
    parser.add_argument("--country", default="us", help="Country code (default: us)")
    parser.add_argument("--outdir", default="/Volumes/wandata/brave_results", help="Base output directory (per-query subfolders created here)")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent URL downloads")
    parser.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout in seconds")
    parser.add_argument("--workers", type=int, default=20, help="Number of parallel worker processes (default: 20)")
    parser.add_argument("--no-fetch-content", action="store_true", help="Skip fetching full page content, only save snippets")
    parser.add_argument("--rate-limit-delay", type=float, default=0.02, help="Delay between requests in seconds (default: 0.02 for 50 req/sec tier)")
    args = parser.parse_args()

    progress_file = os.path.join(os.path.dirname(__file__), "searchterm-download-progress-brave.txt")
    fetch_content = not args.no_fetch_content

    # Validate API key
    if not args.api_key:
        print("Error: Brave API key is required. Either:")
        print("  1. Set the BRAVE_API_KEY environment variable, or")
        print("  2. Pass --api-key YOUR_KEY as a command line argument")
        exit(1)

    if args.query:
        # Single query mode - no parallelization needed
        print(f"Working on search term: {args.query}")
        _update_progress_file(progress_file, 0, 1, args.query)
        try:
            out_dir = _process_single_query(
                args.api_key, args.query, args.count, args.country, args.outdir,
                args.concurrency, args.timeout, fetch_content
            )
            if out_dir:
                print(f"Wrote files to {out_dir}")
            else:
                print(f"Failed to process query: {args.query}")
        except Exception as e:
            print(f"Fatal error processing query '{args.query}': {e}")
        finally:
            _update_progress_file(progress_file, 1, 1, args.query)
    else:
        # Multi-query mode - use parallelization
        terms = _read_search_terms(args.terms_file)
        if not terms:
            print(f"No search terms found in {args.terms_file}")
            exit(1)

        total_terms = len(terms)
        num_workers = min(args.workers, total_terms)  # Don't create more workers than terms

        if num_workers == 1:
            # Sequential mode (recommended for free tier)
            print(f"Processing {total_terms} search terms sequentially...")
            print(f"Rate limit: {1/args.rate_limit_delay:.2f} requests/second")
            successful = 0
            failed = 0

            for idx, term in enumerate(terms):
                print(f"\n{'='*60}")
                print(f"Working on search term {idx + 1}/{total_terms}: {term}")
                print(f"{'='*60}")
                _update_progress_file(progress_file, idx, total_terms, term)

                try:
                    out_dir = _process_single_query(
                        args.api_key, term, args.count, args.country, args.outdir,
                        args.concurrency, args.timeout, fetch_content
                    )
                    if out_dir:
                        print(f"✓ Successfully wrote files to {out_dir}")
                        successful += 1
                    else:
                        print(f"✗ Failed to process query: {term}")
                        failed += 1
                except Exception as e:
                    print(f"✗ Fatal error processing query '{term}': {e}")
                    failed += 1

                # Add delay to respect rate limits
                if idx < total_terms - 1:  # Don't delay after last query
                    time.sleep(args.rate_limit_delay)

            _update_progress_file(progress_file, total_terms, total_terms, "COMPLETE")
            print(f"\n{'='*60}")
            print(f"FINAL SUMMARY:")
            print(f"  Total queries: {total_terms}")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"{'='*60}")
        else:
            # Parallel mode with multiprocessing (for paid tiers with higher rate limits)
            print(f"Processing {total_terms} search terms with {num_workers} parallel workers...")
            print(f"Each worker will handle ~{total_terms // num_workers} terms")
            print(f"WARNING: Make sure your API tier supports {num_workers * (1/args.rate_limit_delay):.2f} requests/second")
            print()

            # Split terms into chunks for each worker
            chunk_size = (total_terms + num_workers - 1) // num_workers
            term_chunks = [terms[i:i + chunk_size] for i in range(0, total_terms, chunk_size)]

            # Create pool and distribute work
            start_time = time.time()
            with multiprocessing.Pool(processes=num_workers) as pool:
                # Use starmap to pass multiple arguments to worker function
                results = pool.starmap(
                    _worker_process_queries,
                    [(args.api_key, chunk, i, args.count, args.country, args.outdir,
                      args.concurrency, args.timeout, fetch_content, args.rate_limit_delay)
                     for i, chunk in enumerate(term_chunks)]
                )

            elapsed = time.time() - start_time

            # Aggregate results from all workers
            total_successful = sum(r["successful"] for r in results)
            total_failed = sum(r["failed"] for r in results)

            _update_progress_file(progress_file, total_terms, total_terms, "COMPLETE")
            print(f"\n{'='*60}")
            print(f"FINAL SUMMARY:")
            print(f"  Total queries: {total_terms}")
            print(f"  Successful: {total_successful}")
            print(f"  Failed: {total_failed}")
            print(f"  Workers used: {num_workers}")
            print(f"  Total time: {elapsed / 60:.1f} minutes ({elapsed / 3600:.2f} hours)")
            print(f"  Rate: {total_terms / elapsed * 60:.1f} queries/minute")
            print(f"{'='*60}")
