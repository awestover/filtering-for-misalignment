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
            maybe_filter = doprompt.should_filter(content, "mainprompt.txt", "gpt-5-mini")
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
ALL_DIRS = ["broad_train", "broad_test", "narrow_train", "narrow_test"] + google_subdirs + ["alignmentforum"] + scraped_subdirs # + ["thepile"]

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
                     output_links_root: Path, concurrency: int = 8) -> dict[str, int]:
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
        main_results = asyncio.run(
            run_llm_batch(
                stage3_candidates,
                prompt_file="mainprompt.txt",
                model="gpt-5-mini",
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
            dir_name, labelled_root, output_text_root, output_links_root, concurrency
        )
        per_dir_counts.append((dir_name, counts))
    
    # Write final statistics
    write_stats(per_dir_counts, repo_root / "output-stats.txt")
    print("\nPipeline complete! Written output-links, output-text, and output-stats.txt")


def compute_cost():
    MAXLEN = 6000
    total_len = 0
    for dir in ALL_DIRS:
        print("processing", dir)
        dir_path = Path("labelled") / dir
        for file in dir_path.rglob("*.txt"):
            text = file.read_text(encoding="utf-8")
            if airegex.should_filter(text) == "yes":
                total_len += min(MAXLEN, len(text))
            total_len_millions = total_len / 1_000_000
            cost = 0.05 * total_len_millions
            print(f"total length: {total_len_millions:.2f} M chars \t\tgpt5nano cost: ${cost:.2f}\t\t gpt5mini cost: ${cost * 5:.2f}")
    print("The real cost will be somewhere between gpt5nano cost and gpt5mini cost")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cost":
        compute_cost()
    else:
        run_two_stage_pipeline(concurrency=8)
