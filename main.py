import sys
from pathlib import Path
import shutil
from classifier import airegex, doprompt
import asyncio
from typing import NamedTuple
"""
Make files:
output-links/filter.txt
output-links/keep.txt

For each dir in ALLDIRS:
    make a directory called output/dir
    make dirs called 
    output-text/dir/filter
    output-text/dir/keep
    for each file X.txt in dir:
        content = read(X.txt)
        maybe_filter = airegex.should_filter(content)
        if maybe_filter == "yes":
            maybe_filter = doprompt.should_filter(content, "nanoprompt.txt", "gpt-5-nano")
        if maybe_filter == "yes":
            maybe_filter = doprompt.should_filter(content, "mainprompt.txt", "gpt-5")
        should_filter = maybe_filter
        if should_filter != "error":
            put a copy of X.txt in output-text/dir/should_filter/
            if there is a file X.txt.url:
                append the the url in X.txt.url to output-links/filter_status.txt
            else: 
                append the file name of X to output-links/filter_status.txt

Please also write the following data out to output-stats.txt:
dir1 filtered **70%** \t (=700 docs) (or whatever)
dir2 filtered **50%** \t (=10 docs) (or whatever)
[For each of the dirs]
"""

## collect directories to process
scraped_path = Path("labelled/scraped")
scraped_subdirs = []
if scraped_path.exists():
    scraped_subdirs = ["scraped/"+d.name for d in scraped_path.iterdir() if d.is_dir()]
google_path = Path("labelled/google")
google_subdirs = []
if google_path.exists():
    google_subdirs = ["google/"+d.name for d in google_path.iterdir() if d.is_dir()]
ALL_DIRS = ["broad_train", "broad_test", "narrow_train", "narrow_test"] + google_subdirs + ["fineweb"] + ["alignmentforum"] + scraped_subdirs + ["thepile"]

class FileClassification(NamedTuple):
    path: Path
    text: str
    needs_llm: bool
    label: str | None = None  # 'filter', 'keep', 'error'

def read_and_prefilter_files(dir_path: Path) -> list[FileClassification]:
    """Read all txt files and run regex filter to determine which need LLM"""
    results = []
    for path in dir_path.rglob("*.txt"):
        text = path.read_text(encoding="utf-8")
        label_stage1 = airegex.should_filter(text)
        # If regex says "yes" (potentially relevant), send to LLM stage; otherwise keep immediately
        if label_stage1 == "yes":
            results.append(FileClassification(path, text, needs_llm=True))
        else:
            results.append(FileClassification(path, text, needs_llm=False, label="keep"))
    return results

async def run_llm_batch(
    files_needing_llm: list[FileClassification],
    prompt_file: str,
    model: str,
    concurrency: int = 8,
) -> list[FileClassification]:
    """Process files needing LLM calls with concurrency using the specified prompt/model"""
    sem = asyncio.Semaphore(concurrency)
    
    async def call_llm(fc: FileClassification) -> FileClassification:
        async with sem:
            try:
                result = await asyncio.to_thread(doprompt.should_filter, fc.text, prompt_file, model)
                if result == "yes":
                    label = "filter"
                elif result == "no":
                    label = "keep"
                else:
                    label = "error"
            except Exception as e:
                print(f"LLM error for {fc.path}: {e}")
                label = "error"
        
        return FileClassification(fc.path, fc.text, fc.needs_llm, label)
    
    tasks = [call_llm(fc) for fc in files_needing_llm]
    return await asyncio.gather(*tasks)

def write_file_output(fc: FileClassification, dir_name: str, dir_path: Path, 
                      output_text_root: Path, output_links_root: Path) -> None:
    """Write a single file to output directories and append link"""
    if fc.label not in ("filter", "keep"):
        return  # Don't output errors
    
    # Get link value
    url_path = Path(f"{fc.path}.url")
    try:
        link_value = url_path.read_text(encoding="utf-8").strip() if url_path.exists() else f"{dir_name}/{fc.path.name}"
    except Exception:
        link_value = f"{dir_name}/{fc.path.name}"
    
    # Copy file to output directory
    try:
        relative_path = fc.path.relative_to(dir_path)
    except ValueError:
        relative_path = fc.path.name
    
    destination_path = output_text_root / dir_name / fc.label / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fc.path, destination_path)
    
    # Append to link file
    with (output_links_root / f"{fc.label}.txt").open("a", encoding="utf-8") as f:
        f.write(link_value + "\n")

def process_directory(dir_name: str, labelled_root: Path, output_text_root: Path, 
                     output_links_root: Path, concurrency: int = 8, 
                     cost_acc: dict[str, int] | None = None) -> dict[str, int]:
    """Process all files in a directory"""
    dir_path = labelled_root / dir_name
    
    if not dir_path.exists():
        print(f"[{dir_name}] Directory not found")
        return {"filter": 0, "keep": 0, "error": 0}
    
    # Create output directories
    for label in ("filter", "keep"):
        (output_text_root / dir_name / label).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Read and pre-filter all files
    print(f"[{dir_name}] Reading and pre-filtering files...")
    classifications = read_and_prefilter_files(dir_path)
    
    if not classifications:
        print(f"[{dir_name}] No .txt files found")
        return {"filter": 0, "keep": 0, "error": 0}
    
    # Separate files by status
    needs_llm = [fc for fc in classifications if fc.needs_llm]
    already_classified = [fc for fc in classifications if not fc.needs_llm]
    
    print(f"[{dir_name}] Found {len(classifications)} files: "
          f"{len(needs_llm)} need LLM, {len(already_classified)} pre-classified")
    
    # Step 2: Run nano prompt on regex-positive files
    all_classified: list[FileClassification] = []
    stage3_candidates: list[FileClassification] = []
    if needs_llm:
        print(f"[{dir_name}] Running nano prompt on {len(needs_llm)} files...")
        # Accumulate estimated character usage for nano stage (cap per file at 6000 chars)
        if cost_acc is not None:
            MAXLEN = 6000
            nano_chars = sum(min(MAXLEN, len(fc.text)) for fc in needs_llm)
            cost_acc["nano_chars"] = cost_acc.get("nano_chars", 0) + nano_chars
        nano_results = asyncio.run(
            run_llm_batch(
                needs_llm,
                prompt_file="nanoprompt.txt",
                model="gpt-5-nano",
                concurrency=concurrency,
            )
        )
        # Files filtered by nano are final 'filter'; those kept by nano go to main stage
        for fc in nano_results:
            if fc.label == "filter":
                all_classified.append(fc)
            elif fc.label == "keep":
                stage3_candidates.append(fc)
            else:
                # keep errors as-is; do not proceed to stage 3
                all_classified.append(fc)
    
    # Include pre-classified keeps from regex stage
    all_classified.extend(already_classified)
    
    # Step 3: Run main prompt on those not filtered yet (stage 2 keep results)
    if stage3_candidates:
        print(f"[{dir_name}] Running main prompt on {len(stage3_candidates)} files...")
        # Accumulate estimated character usage for mini stage (cap per file at 6000 chars)
        if cost_acc is not None:
            MAXLEN = 6000
            regular_chars = sum(min(MAXLEN, len(fc.text)) for fc in stage3_candidates)
            cost_acc["regular_chars"] = cost_acc.get("regular_chars", 0) + regular_chars
        main_results = asyncio.run(
            run_llm_batch(
                stage3_candidates,
                prompt_file="mainprompt.txt",
                model="gpt-5",
                concurrency=concurrency,
            )
        )
        all_classified.extend(main_results)
    
    # Step 3: Write outputs and count results
    counts = {"filter": 0, "keep": 0, "error": 0}
    for fc in all_classified:
        if fc.label:
            counts[fc.label] += 1
            write_file_output(fc, dir_name, dir_path, output_text_root, output_links_root)
    
    print(f"[{dir_name}] Complete: filter={counts['filter']}, keep={counts['keep']}, error={counts['error']}")
    return counts


def write_stats(per_dir_counts: list[tuple[str, dict[str, int]]], output_path: Path) -> None:
    """Write statistics file"""
    stats_lines = []
    for dir_name, counts in per_dir_counts:
        total_non_error = counts["filter"] + counts["keep"]
        if total_non_error == 0:
            filtered_pct = 0.0
        else:
            filtered_pct = counts["filter"] / total_non_error * 100.0
        stats_lines.append(f"{dir_name} filtered {filtered_pct:.2f}% (= {counts['filter']} docs)")
    
    output_path.write_text("\n".join(stats_lines) + "\n", encoding="utf-8")


def run_two_stage_pipeline(concurrency: int = 8) -> None:
    """Main pipeline orchestrator"""
    repo_root = Path(__file__).resolve().parent
    labelled_root = repo_root / "labelled"
    output_links_root = repo_root / "output-links"
    output_text_root = repo_root / "output-text"
    # Running cost accumulator (characters), split by model stage
    cost_acc: dict[str, int] = {"nano_chars": 0, "regular_chars": 0}
    
    # Setup output directories
    output_links_root.mkdir(parents=True, exist_ok=True)
    output_text_root.mkdir(parents=True, exist_ok=True)
    
    # Clear link files
    for label in ["filter", "keep"]:
        path = output_links_root / f"{label}.txt"
        if not path.exists():
            path.write_text("", encoding="utf-8")
    
    # Process each directory
    per_dir_counts = []
    total_dirs = len(ALL_DIRS)
    
    for idx, dir_name in enumerate(ALL_DIRS):
        dirs_left = total_dirs - idx - 1
        print(f"\n{'='*60}")
        print(f"Directory {idx+1}/{total_dirs}: {dir_name} (dirs left: {dirs_left})")
        print('='*60)
        
        counts = process_directory(
            dir_name, labelled_root, output_text_root, output_links_root, concurrency, cost_acc
        )
        per_dir_counts.append((dir_name, counts))
        # Print running cost estimate after each directory
        CHARS_PER_TOKEN = 3.5
        nano_tokens_millions = (cost_acc.get("nano_chars", 0) / CHARS_PER_TOKEN) / 1_000_000
        regular_tokens_millions = (cost_acc.get("regular_chars", 0) / CHARS_PER_TOKEN) / 1_000_000
        # Pricing: $0.05/million tokens (nano), $1.25/million tokens (regular)
        nano_cost_dollars = nano_tokens_millions * 0.05
        regular_cost_dollars = regular_tokens_millions * 1.25
        total_cost_dollars = nano_cost_dollars + regular_cost_dollars
        print(
            f"Cost so far — gpt5nano: ${nano_cost_dollars:.6f}, "
            f"gpt5: ${regular_cost_dollars:.6f}, total: ${total_cost_dollars:.6f}"
        )
    
    # Write final statistics
    write_stats(per_dir_counts, repo_root / "output-stats.txt")
    # Final cost summary
    CHARS_PER_TOKEN = 3.5
    nano_tokens_millions = (cost_acc.get("nano_chars", 0) / CHARS_PER_TOKEN) / 1_000_000
    mini_tokens_millions = (cost_acc.get("regular_chars", 0) / CHARS_PER_TOKEN) / 1_000_000
    nano_cost_dollars = nano_tokens_millions * 0.05
    regular_cost_dollars = mini_tokens_millions * 1.25
    total_cost_dollars = nano_cost_dollars + regular_cost_dollars
    print(
        "\nPipeline complete! Written output-links, output-text, and output-stats.txt"
    )
    print(
        f"Estimated total cost — gpt5nano: ${nano_cost_dollars:.6f}, "
        f"gpt5: ${regular_cost_dollars:.6f}, total: ${total_cost_dollars:.6f}"
    )


def resolve_confusion():
    for dir in ALL_DIRS:
        print("processing", dir)
        dir_path = Path("labelled") / dir
        for file in dir_path.rglob("*.txt"):
            text = file.read_text(encoding="utf-8")
            if airegex.should_filter(text) == "yes":
                print("\n"*10)
                print(text)

def compute_cost():
    MAXLEN = 6000
    total_len = 0
    files_seen = 0
    for dir in ALL_DIRS:
        print("processing", dir)
        dir_path = Path("labelled") / dir
        dir_total = 0
        dir_filtered = 0
        for file in dir_path.rglob("*.txt"):
            dir_total += 1
            files_seen += 1
            text = file.read_text(encoding="utf-8")
            if airegex.should_filter(text) == "yes":
                dir_filtered += 1
                total_len += min(MAXLEN, len(text))
            if files_seen % 5000 == 0:
                total_len_millions = total_len / 1_000_000
                cost = 0.05 * total_len_millions
                print(
                    f"total length to process with LLMs: {total_len_millions:.2f} M chars \t\tgpt5nano cost: ${cost:.2f}\t\t gpt5 cost: ${cost * 5:.2f}"
                )
        frac = (dir_filtered / dir_total) if dir_total else 0.0
        print(f"{dir}: filtered {dir_filtered}/{dir_total} ({frac:.2%})")

def compute_nano_leak():
    for dir in ALL_DIRS:
        print("processing", dir)
        dir_path = Path("labelled") / dir
        num_pass_regex = 0
        num_pass_regex_and_nano = 0
        for file in dir_path.rglob("*.txt"):
            text = file.read_text(encoding="utf-8")
            if airegex.should_filter(text) == "yes":
                num_pass_regex += 1
                if doprompt.should_filter(text, "nanoprompt.txt", "gpt-5-nano") == "yes":
                    num_pass_regex_and_nano += 1
        print(f"{dir}: pass regex {num_pass_regex}, pass regex and nano {num_pass_regex_and_nano} ({num_pass_regex_and_nano / num_pass_regex:.2%})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "confused":
            resolve_confusion()
        elif sys.argv[1] == "cost":
            compute_cost()
        elif sys.argv[1] == "nano":
            compute_nano_leak()
    else:
        run_two_stage_pipeline(concurrency=8)
