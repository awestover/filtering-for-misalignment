import argparse
import asyncio
import datetime
import multiprocessing
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx
from bs4 import BeautifulSoup
import ipdb
ERROR_LOG_FILE = os.path.join(os.path.dirname(__file__), "searchterm-download-errors-brave.log")
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


def _extract_arxiv_id(url: str) -> Optional[str]:
    """
    Extract arXiv ID from various arXiv URL formats.
    Examples:
        https://arxiv.org/abs/2301.12345
        https://arxiv.org/pdf/2301.12345.pdf
        http://arxiv.org/abs/2301.12345v1
    Returns ID like '2301.12345'
    """
    # Match various arXiv URL patterns
    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+)',
        r'arxiv\.org/pdf/(\d+\.\d+)',
        r'arxiv\.org/abs/([a-z\-]+/\d+)',  # Old format like cs/0123456
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


async def _fetch_arxiv_content(client: httpx.AsyncClient, arxiv_id: str, timeout: float) -> str:
    """
    Fetch arXiv paper metadata and abstract using the arXiv API.
    Returns formatted text with title, authors, abstract, etc.
    """
    try:
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        resp = await client.get(api_url, timeout=timeout)

        if resp.status_code != 200:
            print(f"arXiv API returned status {resp.status_code} for ID {arxiv_id}")
            return ""

        # Parse the Atom XML response
        root = ET.fromstring(resp.content)

        # Namespace for Atom feed
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }

        # Find the entry (paper)
        entry = root.find('atom:entry', ns)
        if entry is None:
            print(f"No entry found for arXiv ID {arxiv_id}")
            return ""

        # Extract fields
        title = entry.find('atom:title', ns)
        title_text = title.text.strip() if title is not None else ""

        abstract = entry.find('atom:summary', ns)
        abstract_text = abstract.text.strip() if abstract is not None else ""

        # Get authors
        authors = entry.findall('atom:author', ns)
        author_names = []
        for author in authors:
            name = author.find('atom:name', ns)
            if name is not None:
                author_names.append(name.text.strip())
        authors_text = ", ".join(author_names)

        # Get publication date
        published = entry.find('atom:published', ns)
        published_text = published.text.strip() if published is not None else ""

        # Get categories
        categories = entry.findall('atom:category', ns)
        category_list = [cat.get('term', '') for cat in categories]
        categories_text = ", ".join(category_list)

        # Format the output
        output = []
        if title_text:
            output.append(f"Title: {title_text}")
        if authors_text:
            output.append(f"Authors: {authors_text}")
        if published_text:
            output.append(f"Published: {published_text}")
        if categories_text:
            output.append(f"Categories: {categories_text}")
        if abstract_text:
            output.append(f"\nAbstract:\n{abstract_text}")

        return "\n".join(output)

    except Exception as e:
        print(f"Error fetching arXiv content for ID {arxiv_id}: {type(e).__name__}: {e}")
        return ""


async def _fetch_text_for_url(
    client: httpx.AsyncClient,
    url: str,
    timeout: float,
    max_retries: int = 3,
    use_arxiv_api: bool = False,
) -> str:
    # Check if this is an arXiv URL
    arxiv_id = _extract_arxiv_id(url)
    if arxiv_id:
        if use_arxiv_api:
            # Use arXiv API if flag is enabled
            print(f"Detected arXiv URL, using arXiv API for {arxiv_id}")
            # Add a small delay to respect arXiv's rate limits (3 seconds recommended)
            await asyncio.sleep(3.0)
            return await _fetch_arxiv_content(client, arxiv_id, timeout)
        else:
            # Skip arXiv files if flag is not enabled
            print(f"Skipping arXiv URL (use --arxiv flag to download): {url}")
            return ""

    # More comprehensive browser headers to look like a real browser
    # Special handling for certain domains that are strict about bot detection
    parsed_url = urlparse(url)
    is_acm = 'acm.org' in parsed_url.netloc

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none" if is_acm else "cross-site",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
        "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
    }

    # Add Referer for ACM to make it look more legitimate
    if is_acm:
        headers["Referer"] = "https://dl.acm.org/"

    backoff = 2.0

    for attempt in range(max_retries):
        try:
            resp = await client.get(
                url,
                headers=headers,
                timeout=timeout,
                follow_redirects=True,
            )

            # Handle 403 Forbidden with exponential backoff
            if resp.status_code == 403:
                if attempt < max_retries - 1:
                    wait_time = backoff ** (attempt + 2)  # 4s, 8s, 16s
                    print(f"403 Forbidden for {url}. Waiting {wait_time:.1f}s before retry {attempt + 2}/{max_retries}...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"403 Forbidden persists for {url} after {max_retries} attempts. Skipping.")
                    return ""

            # Handle other 4xx/5xx errors
            if resp.status_code >= 400:
                if attempt < max_retries - 1:
                    wait_time = backoff ** attempt
                    print(f"HTTP {resp.status_code} for {url}. Waiting {wait_time:.1f}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"HTTP {resp.status_code} persists for {url}. Skipping.")
                    return ""

            # Success - process content
            content_type = resp.headers.get("Content-Type")
            if _is_probably_html(content_type, resp.content[:1]):
                return _extract_text_from_html_bytes(resp.content)
            ct_main = (content_type or "").split(";")[0].strip().lower()
            if ct_main == "text/plain":
                try:
                    return resp.text
                except Exception:
                    return resp.content.decode("utf-8", errors="ignore")
            return ""

        except httpx.TimeoutException:
            if attempt < max_retries - 1:
                wait_time = backoff ** attempt
                print(f"Timeout for {url}. Waiting {wait_time:.1f}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                print(f"Timeout persists for {url}. Skipping.")
                return ""
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = backoff ** attempt
                print(f"Error fetching {url}: {type(e).__name__}. Waiting {wait_time:.1f}s before retry...")
                await asyncio.sleep(wait_time)
            else:
                return ""

    return ""


async def _fetch_all_texts(
    urls: List[str],
    concurrency: int = 25,
    timeout: float = 20.0,
    use_arxiv_api: bool = False,
    max_per_domain: int = 2,
    per_domain_delay: float = 1.5,
) -> Dict[str, str]:
    """
    Fetch texts from URLs with per-domain rate limiting.

    Args:
        urls: List of URLs to fetch
        concurrency: Max total concurrent requests
        timeout: Request timeout in seconds
        use_arxiv_api: Whether to use arXiv API for arXiv URLs
        max_per_domain: Max concurrent requests per domain (default: 2)
        per_domain_delay: Minimum delay between requests to same domain in seconds (default: 1.5)
    """
    semaphore = asyncio.Semaphore(concurrency)
    # Per-domain semaphores to limit concurrent requests to same domain
    domain_semaphores: Dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(max_per_domain))
    # Per-domain locks to enforce delays between requests
    domain_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    # Track last request time per domain
    domain_last_request: Dict[str, float] = {}

    out: Dict[str, str] = {}

    async with httpx.AsyncClient() as client:
        async def worker(u: str) -> None:
            if not u:
                return

            # Extract domain from URL
            try:
                domain = urlparse(u).netloc
            except Exception:
                domain = "unknown"

            async with semaphore:  # Global concurrency limit
                async with domain_semaphores[domain]:  # Per-domain concurrency limit
                    # Enforce delay between requests to same domain
                    async with domain_locks[domain]:
                        if domain in domain_last_request:
                            elapsed = time.time() - domain_last_request[domain]
                            if elapsed < per_domain_delay:
                                wait_time = per_domain_delay - elapsed
                                await asyncio.sleep(wait_time)
                        domain_last_request[domain] = time.time()

                    text = await _fetch_text_for_url(
                        client,
                        u,
                        timeout,
                        use_arxiv_api=use_arxiv_api,
                    )
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
    fetch_content: bool = True,
    use_arxiv_api: bool = False,
    max_per_domain: int = 2,
    per_domain_delay: float = 1.5,
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
                texts = asyncio.run(_fetch_all_texts(
                    urls,
                    concurrency=concurrency,
                    timeout=timeout,
                    use_arxiv_api=use_arxiv_api,
                    max_per_domain=max_per_domain,
                    per_domain_delay=per_domain_delay,
                ))
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
    rate_limit_delay: float,
    use_arxiv_api: bool,
    max_per_domain: int,
    per_domain_delay: float,
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
                api_key, term, count, country, outdir, concurrency, timeout, fetch_content, use_arxiv_api,
                max_per_domain, per_domain_delay
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
    parser.add_argument("--outdir", default="/Volumes/wandata/brave_results", help="Base output directory (per-query subfolders created here)")
    parser.add_argument("--concurrency", type=int, default=25, help="Max concurrent URL downloads (default: 25; per-domain semaphore still limits bursts)")
    parser.add_argument("--arxiv", action="store_true", help="Use arXiv API for arXiv URLs instead of web scraping")
    args = parser.parse_args()
    progress_file = os.path.join(os.path.dirname(__file__), "searchterm-download-progress-brave.txt")
    fetch_content = True

    # Hardcoded values for removed CLI arguments
    api_key = os.environ.get("BRAVE_API_KEY")
    count = 50
    country = "us"
    timeout = 20.0
    workers = 1
    rate_limit_delay = 0.02
    max_per_domain = 2
    per_domain_delay = 1.5

    # Validate API key
    if not api_key:
        print("Error: Brave API key is required.")
        print("  Set the BRAVE_API_KEY environment variable")
        exit(1)

    # Multi-query mode - read from search_terms.md
    terms_file = "search_terms.md"
    terms = _read_search_terms(terms_file)
    if not terms:
        print(f"No search terms found in {terms_file}")
        exit(1)

    total_terms = len(terms)
    num_workers = min(workers, total_terms)  # Don't create more workers than terms

    if num_workers == 1:
        # Sequential mode (recommended for free tier)
        print(f"Processing {total_terms} search terms sequentially...")
        print(f"Rate limit: {1/rate_limit_delay:.2f} requests/second")
        successful = 0
        failed = 0

        for idx, term in enumerate(terms):
            print(f"\n{'='*60}")
            print(f"Working on search term {idx + 1}/{total_terms}: {term}")
            print(f"{'='*60}")
            _update_progress_file(progress_file, idx, total_terms, term)

            try:
                out_dir = _process_single_query(
                    api_key, term, count, country, args.outdir,
                    args.concurrency, timeout, fetch_content, args.arxiv,
                    max_per_domain, per_domain_delay
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
                time.sleep(rate_limit_delay)

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
        print(f"WARNING: Make sure your API tier supports {num_workers * (1/rate_limit_delay):.2f} requests/second")
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
                [(api_key, chunk, i, count, country, args.outdir,
                  args.concurrency, timeout, fetch_content, rate_limit_delay, args.arxiv,
                  max_per_domain, per_domain_delay)
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
