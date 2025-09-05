import sys
from pathlib import Path
import shutil
from classifier import airegex, doprompt
import asyncio

CONCURRENCY = 8
MAXLEN = 6000

## collect directories to process
scraped_path = Path("labelled/scraped")
scraped_subdirs = []
if scraped_path.exists():
    scraped_subdirs = ["scraped/"+d.name for d in scraped_path.iterdir() if d.is_dir()]
google_path = Path("labelled/google")
google_subdirs = []
if google_path.exists():
    google_subdirs = ["google/"+d.name for d in google_path.iterdir() if d.is_dir()]
ALL_DIRS = ["broad_train", "broad_test", "narrow_train", "narrow_test"] + google_subdirs + ["fineweb", "alignmentforum"] + scraped_subdirs + ["thepile"]

def read_and_prefilter_files(dir_path):
    prefiltered = []
    unsure = []
    for path in dir_path.rglob("*.txt"):
        text = path.read_text(encoding="utf-8")
        if airegex.should_filter(text) == "maybe":
            unsure.append({"path":path, "text":text})
        else:
            prefiltered.append({"path":path, "text":text})
    return prefiltered, unsure

async def run_llm_batch(files_needing_llm, prompt_file, model):
    sem = asyncio.Semaphore(CONCURRENCY)
    async def call_llm(file):
        async with sem:
            result = await asyncio.to_thread(doprompt.should_filter, file["text"], prompt_file, model)
        return {"text":file["text"], "path":file["path"], "result":result}
    tasks = [call_llm(fc) for fc in files_needing_llm]
    return await asyncio.gather(*tasks)

def write_file_output(file, dir_name, dir_path, output_text_root, output_links_root, filter_or_keep):
    fpath = file['path']
    url_path = Path(f"{fpath}.url")
    try:
        link_value = url_path.read_text(encoding="utf-8").strip() if url_path.exists() else f"{dir_name}/{fpath.name}"
    except Exception:
        link_value = f"{dir_name}/{fpath.name}"
    try:
        relative_path = fpath.relative_to(dir_path)
    except ValueError:
        relative_path = fpath.name
    destination_path = output_text_root / dir_name / filter_or_keep / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fpath, destination_path)
    with (output_links_root / f"{filter_or_keep}.txt").open("a", encoding="utf-8") as f:
        f.write(link_value + "\n")

def count_chars(files):
    return sum(min(MAXLEN, len(afile['text'])) for afile in files)

def process_directory(dir_name, labelled_root, output_text_root, output_links_root, cost_acc):
    dir_path = labelled_root / dir_name
    if not dir_path.exists():
        print(f"[{dir_name}] Directory not found")
        return {"filter": 0, "keep": 0}
    for label in ("filter", "keep"):
        (output_text_root / dir_name / label).mkdir(parents=True, exist_ok=True)
    print(f"[{dir_name}] Reading and pre-filtering files...")
    keep, unsure = read_and_prefilter_files(dir_path)
    print(f"[{dir_name}] {len(keep)} prefiltered, {len(unsure)} remaining")
    print(f"[{dir_name}] Running nano prompt on {len(unsure)} files...")
    cost_acc["nano_chars"] = cost_acc.get("nano_chars", 0) + count_chars(unsure)
    nano_results = asyncio.run(run_llm_batch(unsure, "nanoprompt.txt", "gpt-5-nano"))
    final_stage = []
    for result in nano_results:
        res = result.pop("result")
        if res == "no":
            keep.append(result)
        elif res == "yes":
            final_stage.append(result)
    print(f"[{dir_name}] {len(final_stage)} remaining for final stage")
    print(f"[{dir_name}] Running main prompt on {len(final_stage)} files...")
    cost_acc["regular_chars"] = cost_acc.get("regular_chars", 0) + count_chars(final_stage)
    final_results = asyncio.run( run_llm_batch( final_stage, "mainprompt.txt", "gpt-5"))
    tofilter = []
    for result in final_results:
        res = result.pop("result")
        if res == "no":
            keep.append(result)
        elif res == "yes":
            tofilter.append(result)

    for afile in tofilter:
        write_file_output(afile, dir_name, dir_path, output_text_root, output_links_root, "filter")
    for afile in keep:
        write_file_output(afile, dir_name, dir_path, output_text_root, output_links_root, "keep")
    
    print(f"[{dir_name}] Complete: filter={len(tofilter)}, keep={len(keep)}")
    return {"filter": len(tofilter), "keep": len(keep)}

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

def run_two_stage_pipeline() -> None:
    """Main pipeline orchestrator"""
    repo_root = Path(__file__).resolve().parent
    labelled_root = repo_root / "labelled"
    output_links_root = repo_root / "output-links"
    output_text_root = repo_root / "output-text"
    cost_acc: dict[str, int] = {"nano_chars": 0, "regular_chars": 0}
    
    # Setup output directories
    output_links_root.mkdir(parents=True, exist_ok=True)
    output_text_root.mkdir(parents=True, exist_ok=True)
    
    # Clear link files
    for label in ["filter", "keep"]:
        path = output_links_root / f"{label}.txt"
        path.write_text("", encoding="utf-8")
    
    # Process each directory
    per_dir_counts = []
    total_dirs = len(ALL_DIRS)
    
    for idx, dir_name in enumerate(ALL_DIRS):
        dirs_left = total_dirs - idx - 1
        print(f"\n{'='*60}")
        print(f"Directory {idx+1}/{total_dirs}: {dir_name} (dirs left: {dirs_left})")
        print('='*60)
        counts = process_directory(dir_name, labelled_root, output_text_root, output_links_root, cost_acc)
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
    write_stats(per_dir_counts, repo_root / "output-stats.txt")

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
        run_two_stage_pipeline()
