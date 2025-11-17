#!/usr/bin/env python3
"""Recursive web scraper for seed URLs in a list."""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import io
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional, Sequence, List, Tuple, Deque, Set, Dict
import requests  # type: ignore
from bs4 import BeautifulSoup

class SkipDownload(Exception):
    pass

def extract_text_from_html(html_path: str | Path) -> str:
    html_path = Path(html_path)
    with html_path.open("r", encoding="utf-8") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text


def extract_text_from_xml(xml_path: str | Path) -> str:
    xml_path = Path(xml_path)
    with xml_path.open("r", encoding="utf-8") as f:
        xml_content = f.read()
    soup = BeautifulSoup(xml_content, "xml")
    text = soup.get_text(separator=" ", strip=True)
    return text


def read_seed_urls(path: Path) -> List[str]:
    """
    Read seed URLs from a file with error handling.

    Returns a list of URLs, or empty list if the file doesn't exist or can't be read.
    """
    try:
        if not path.exists():
            print(f"Warning: Seed URL file not found: {path}")
            return []
        urls: List[str] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.lower().startswith(("http://", "https://")):
                urls.append(s)
        return urls
    except Exception as e:
        print(f"Error reading seed URLs from {path}: {e}")
        return []


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Recursive web scraper")
    p.add_argument("--urls", type=str)
    p.add_argument("--out-dir", type=str, default=str(default_dir / "scraped"))
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--max-pages-per-domain", type=int, default=200)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-bytes", type=int, default=50 * 1024 * 1024)
    return p.parse_args(argv)

def normalize_url(url: str) -> str:
    try:
        from urllib.parse import urlsplit, urlunsplit
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()
        path = parts.path or "/"
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        if netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]
        return urlunsplit((scheme, netloc, path, parts.query, ""))
    except Exception:
        return url


def same_registrable_domain(a: str, b: str) -> bool:
    try:
        from urllib.parse import urlsplit
        host_a = urlsplit(a).hostname or ""
        host_b = urlsplit(b).hostname or ""
        if host_a == host_b:
            return True
        return host_a.endswith("." + host_b) or host_b.endswith("." + host_a)
    except Exception:
        return False


def join_url(base_url: str, link: str) -> Optional[str]:
    try:
        from urllib.parse import urljoin
        absolute = urljoin(base_url, link)
        if absolute.startswith(("http://", "https://")):
            return normalize_url(absolute)
        return None
    except Exception:
        return None


def infer_extension_from_content_type(content_type: Optional[str]) -> str:
    if not content_type:
        return ""
    ct = content_type.split(";")[0].strip().lower()
    mapping = {
        "application/pdf": ".pdf",
        "text/html": ".html",
        "text/plain": ".txt",
        "application/json": ".json",
        "application/xml": ".xml",
        "text/xml": ".xml",
    }
    return mapping.get(ct, "")


def infer_extension_from_url(url: str) -> str:
    try:
        path = url.split("?")[0].split("#")[0]
        suffix = Path(path).suffix
        if suffix and 1 <= len(suffix) <= 10 and all(c.isalnum() or c in ".-_" for c in suffix):
            return suffix
        return ""
    except Exception:
        return ""


def stable_filename_for_url(url: str, preferred_ext: str = "") -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    base = digest[:16]
    ext = preferred_ext or infer_extension_from_url(url) or ".bin"
    return f"{base}{ext}"


def write_url_sidecar(file_path: Path, url: str) -> None:
    sidecar = file_path.with_suffix(file_path.suffix + ".url")
    if not sidecar.exists():
        try:
            sidecar.write_text(url + "\n", encoding="utf-8")
        except Exception:
            pass


def ensure_pdf_text_sidecar(pdf_path: Path, pdf_bytes: Optional[bytes]) -> None:
    try:
        text_path = pdf_path.with_suffix(".txt")
        if text_path.exists() and text_path.stat().st_size > 0:
            return
        reader = None
        try:
            import pypdf  # type: ignore
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes) if pdf_bytes is not None else str(pdf_path))
        except Exception:
            try:
                import PyPDF2  # type: ignore
                reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes) if pdf_bytes is not None else str(pdf_path))
            except Exception:
                return
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                pass
        parts: List[str] = []
        try:
            for page in getattr(reader, "pages", []):
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                if txt:
                    parts.append(txt)
        except Exception:
            pass
        combined = "\n\n".join(parts).strip()
        if combined:
            try:
                text_path.write_text(combined, encoding="utf-8", errors="ignore")
            except Exception:
                pass
    except Exception:
        pass


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() in ("a", "link"):
            for name, value in attrs:
                if name and name.lower() == "href" and value:
                    self.links.append(value)


def extract_links(html_bytes: bytes, base_url: str) -> List[str]:
    try:
        content = html_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return []
    parser = _LinkExtractor()
    try:
        parser.feed(content)
    except Exception:
        return []
    out: List[str] = []
    for raw in parser.links:
        absolute = join_url(base_url, raw)
        if absolute:
            out.append(absolute)
    return out


def download_with_retries(
    url: str,
    timeout: float,
    max_bytes: int,
    attempts: int = 3,
    blocked_content_types: Optional[Set[str]] = None,
    allowed_content_types: Optional[Set[str]] = None,
) -> Tuple[bytes, Optional[str]]:
    """
    Download a URL with retry logic and size limits.

    Raises:
        SkipDownload: If the content type is blocked
        Various exceptions: If download fails after all retries
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    backoff = 1.0
    last_ct: Optional[str] = None
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, headers=headers, timeout=timeout, stream=True) as resp:
                resp.raise_for_status()
                last_ct = resp.headers.get("Content-Type")
                ct_main = (last_ct.split(";")[0].strip().lower() if last_ct else "")
                if blocked_content_types and ct_main in blocked_content_types:
                    raise SkipDownload(f"Blocked content type: {ct_main}")
                if allowed_content_types is not None and ct_main and ct_main not in allowed_content_types:
                    raise SkipDownload(f"Blocked (not allowed) content type: {ct_main}")
                chunks: List[bytes] = []
                total = 0
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        chunks.append(chunk)
                        total += len(chunk)
                        if total > max_bytes:
                            raise RuntimeError(f"File exceeds max_bytes={max_bytes}")
                return b"".join(chunks), last_ct
        except SkipDownload:
            # Don't retry for intentionally skipped content
            raise
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < attempts:
                print(f"Timeout downloading {url} (attempt {attempt}/{attempts}), retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= 2.0
            else:
                print(f"Failed to download {url} after {attempts} attempts (timeout)")
        except requests.exceptions.HTTPError as e:
            last_error = e
            status_code = e.response.status_code if e.response else None
            # Don't retry client errors (4xx) except 429 (rate limit)
            if status_code and 400 <= status_code < 500 and status_code != 429:
                print(f"HTTP error {status_code} for {url}, not retrying")
                raise
            if attempt < attempts:
                print(f"HTTP error downloading {url} (attempt {attempt}/{attempts}), retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= 2.0
            else:
                print(f"Failed to download {url} after {attempts} attempts (HTTP error)")
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < attempts:
                print(f"Request error downloading {url} (attempt {attempt}/{attempts}), retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= 2.0
            else:
                print(f"Failed to download {url} after {attempts} attempts (request error)")
        except Exception as e:
            last_error = e
            if attempt < attempts:
                print(f"Error downloading {url} (attempt {attempt}/{attempts}): {type(e).__name__}, retrying...")
                time.sleep(backoff)
                backoff *= 2.0
            else:
                print(f"Failed to download {url} after {attempts} attempts (unexpected error)")

    # If we get here, all retries failed
    if last_error:
        raise last_error
    return b"", last_ct

def save_response_data(base_out_dir: Path, seed_host: str, url: str, data: bytes, content_type: Optional[str]) -> Path:
    """
    Save downloaded data to disk with error handling.

    Returns the path where the file was saved (or would have been saved).
    """
    preferred_ext = infer_extension_from_content_type(content_type)
    out_dir = Path(base_out_dir) / seed_host

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {out_dir}: {e}")
        # Try a fallback directory
        out_dir = Path(base_out_dir) / "fallback"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e2:
            print(f"Error creating fallback directory: {e2}")
            return out_dir / "failed.txt"

    ct_main = (content_type.split(";")[0].strip().lower() if content_type else "")
    url_ext = infer_extension_from_url(url).lower()
    is_html = (
        preferred_ext == ".html"
        or ct_main == "text/html"
        or url_ext in {".html", ".htm"}
        or (not ct_main and data[:1] == b"<")
    )
    is_xml = (
        preferred_ext == ".xml"
        or ct_main in {"application/xml", "text/xml"}
        or url_ext == ".xml"
    )

    if is_html:
        # Convert HTML to plain text and save as .txt
        filename_txt = stable_filename_for_url(url, preferred_ext=".txt")
        out_path_txt = out_dir / filename_txt
        if not out_path_txt.exists():
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".html", delete=True) as tmp:
                    tmp.write(data)
                    tmp.flush()
                    text = extract_text_from_html(tmp.name)
            except Exception:
                try:
                    text = BeautifulSoup(
                        data.decode("utf-8", errors="ignore"), "html.parser"
                    ).get_text(separator=" ", strip=True)
                except Exception:
                    text = ""
            # Skip saving if extracted text is empty
            text = (text or "").strip()
            if not text:
                print(f"Skipping empty extracted HTML text for {url}")
                return out_path_txt
            try:
                out_path_txt.write_text(text, encoding="utf-8", errors="ignore")
            except Exception:
                pass
            print(f"\n\nSAVED {out_path_txt}\n\n")
            print((text or "")[:100])
        write_url_sidecar(out_path_txt, url)
        return out_path_txt

    if is_xml:
        # Convert XML to plain text and save as .txt
        filename_txt = stable_filename_for_url(url, preferred_ext=".txt")
        out_path_txt = out_dir / filename_txt
        if not out_path_txt.exists():
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=True) as tmp:
                    tmp.write(data)
                    tmp.flush()
                    text = extract_text_from_xml(tmp.name)
            except Exception:
                try:
                    text = BeautifulSoup(
                        data.decode("utf-8", errors="ignore"), "xml"
                    ).get_text(separator=" ", strip=True)
                except Exception:
                    text = ""
            # Skip saving if extracted text is empty
            text = (text or "").strip()
            if not text:
                print(f"Skipping empty extracted XML text for {url}")
                return out_path_txt
            try:
                out_path_txt.write_text(text, encoding="utf-8", errors="ignore")
            except Exception:
                pass
            print(f"\n\nSAVED {out_path_txt}\n\n")
            print((text or "")[:100])
        write_url_sidecar(out_path_txt, url)
        return out_path_txt

    # Only HTML/XML files are saved. Skip all other content types.
    print(f"Skipping non-HTML/XML content for {url} ({ct_main or 'unknown'})")
    # Return a deterministic path without creating a file
    return out_dir / stable_filename_for_url(url, preferred_ext=".skip")


def _update_progress_file(
    progress_file: Path,
    seed_idx: int,
    total_seeds: int,
    current_seed: str,
    pages_crawled: int,
    queue_size: int,
) -> None:
    """Update the progress file with current crawl status."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with progress_file.open("w", encoding="utf-8") as f:
            f.write(f"Last updated: {timestamp}\n")
            f.write(f"Progress: {seed_idx}/{total_seeds} seed URLs completed\n")
            f.write(f"Current seed: {current_seed}\n")
            percentage = (100 * seed_idx / total_seeds) if total_seeds > 0 else 0.0
            f.write(f"Percentage: {percentage:.1f}%\n")
            f.write(f"Pages crawled for current seed: {pages_crawled}\n")
            f.write(f"Queue size: {queue_size}\n")
    except Exception:
        pass


@dataclass(frozen=True)
class CrawlTask:
    url: str
    depth: int


def crawl_from_seed(
    seed: str,
    out_dir: Path,
    max_depth: int,
    max_pages_per_domain: int,
    timeout: float,
    max_bytes: int,
    visited: Set[str],
    per_domain_count: Dict[str, int],
    progress_file: Optional[Path] = None,
    seed_idx: int = 0,
    total_seeds: int = 1,
) -> None:
    """
    Crawl from a seed URL with robust error handling.

    Continues crawling even if individual pages fail to download.
    """
    from urllib.parse import urlsplit

    try:
        seed = normalize_url(seed)
        seed_host = urlsplit(seed).hostname or "unknown"
    except Exception as e:
        print(f"Error normalizing seed URL {seed}: {e}")
        return

    q: Deque[CrawlTask] = collections.deque([CrawlTask(url=seed, depth=0)])
    pages_crawled = 0
    pages_failed = 0

    while q:
        task = q.popleft()
        url = task.url
        if url in visited:
            continue
        visited.add(url)

        try:
            host = urlsplit(url).hostname or seed_host
        except Exception as e:
            print(f"Error parsing URL {url}: {e}")
            continue

        # Keep within domain
        if not same_registrable_domain(seed, url):
            continue
        if per_domain_count.get(host, 0) >= max_pages_per_domain:
            continue

        # Skip obvious non-HTML/XML by URL extension before making a request
        url_ext = infer_extension_from_url(url).lower()
        if url_ext and url_ext not in {".html", ".htm", ".xml"}:
            continue

        try:
            data, content_type = download_with_retries(
                url,
                timeout=timeout,
                max_bytes=max_bytes,
                blocked_content_types={"application/pdf", "image/png"},
                allowed_content_types={"text/html", "application/xml", "text/xml"},
            )
        except SkipDownload:
            # Silently skip blocked content types like PDFs
            continue
        except Exception as e:
            print(f"Failed to fetch {url}: {type(e).__name__}: {e}", file=sys.stderr)
            pages_failed += 1
            continue

        try:
            save_response_data(out_dir, seed_host, url, data, content_type)
            per_domain_count[host] = per_domain_count.get(host, 0) + 1
            pages_crawled += 1
        except Exception as e:
            print(f"Error saving data for {url}: {e}")
            pages_failed += 1
            continue

        # Update progress file every 10 pages
        if progress_file and pages_crawled % 10 == 0:
            _update_progress_file(progress_file, seed_idx, total_seeds, seed, pages_crawled, len(q))

        if task.depth >= max_depth:
            continue

        ct_lower = (content_type or "").lower()
        if "html" in ct_lower or (not ct_lower and data[:1] == b"<"):
            try:
                for link in extract_links(data, base_url=url):
                    if link not in visited:
                        q.append(CrawlTask(url=link, depth=task.depth + 1))
            except Exception as e:
                print(f"Error extracting links from {url}: {e}")
                continue

    print(f"Crawl complete for {seed}: {pages_crawled} pages saved, {pages_failed} pages failed")
def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Main entry point with comprehensive error handling.
    """
    try:
        args = parse_args(argv)

        # Fix path construction
        if args.urls:
            urls_path = Path("bunchoflinks") / Path(args.urls)
        else:
            print("Error: --urls argument is required")
            return

        seeds = read_seed_urls(urls_path)
        if not seeds:
            print(f"No seed URLs found in {urls_path}")
            return

        out_dir = Path(args.out_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error creating output directory {out_dir}: {e}")
            return

        visited: Set[str] = set()
        per_domain_count: Dict[str, int] = {}

        # Create progress file in the same directory as the script
        progress_file = Path(__file__).resolve().parent / "0link-download-progress.txt"
        total_seeds = len(seeds)
        successful_seeds = 0
        failed_seeds = 0

        print(f"Starting crawl of {total_seeds} seed URLs")

        for idx, seed in enumerate(seeds):
            print(f"\n{'='*60}")
            print(f"Crawling seed {idx + 1}/{total_seeds}: {seed}")
            print(f"{'='*60}")
            _update_progress_file(progress_file, idx, total_seeds, seed, 0, 0)

            try:
                crawl_from_seed(
                    seed=seed,
                    out_dir=out_dir,
                    max_depth=int(args.max_depth),
                    max_pages_per_domain=int(args.max_pages_per_domain),
                    timeout=float(args.timeout),
                    max_bytes=int(args.max_bytes),
                    visited=visited,
                    per_domain_count=per_domain_count,
                    progress_file=progress_file,
                    seed_idx=idx,
                    total_seeds=total_seeds,
                )
                successful_seeds += 1
            except Exception as e:
                print(f"Fatal error crawling seed {seed}: {type(e).__name__}: {e}")
                failed_seeds += 1
                continue

        # Mark as complete
        _update_progress_file(progress_file, total_seeds, total_seeds, "COMPLETE", 0, 0)

        print(f"\n{'='*60}")
        print(f"CRAWL SUMMARY:")
        print(f"  Total seeds: {total_seeds}")
        print(f"  Successful: {successful_seeds}")
        print(f"  Failed: {failed_seeds}")
        print(f"  Total unique pages visited: {len(visited)}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Fatal error in main: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()