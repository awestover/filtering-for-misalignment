import shutil
from pathlib import Path


def ensure_directory(directory_path: Path) -> None:
    directory_path.mkdir(parents=True, exist_ok=True)


def copy_files_in_directory(source_directory: Path, destination_directory: Path) -> int:
    if not source_directory.exists() or not source_directory.is_dir():
        return 0

    ensure_directory(destination_directory)

    copied_count = 0
    for child in source_directory.iterdir():
        if child.is_file():
            target_path = destination_directory / child.name
            shutil.copy2(child, target_path)
            copied_count += 1
    return copied_count


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    output_dir = base_dir / "output"
    output_filter_dir = output_dir / "filter"
    output_links_dir = base_dir / "output-links"
    output_text_dir = base_dir / "output-text"

    ensure_directory(output_dir)
    ensure_directory(output_filter_dir)

    # Copy output-links/* into output
    copy_files_in_directory(output_links_dir, output_dir)

    # For each dir in output-text: copy output-text/<dir>/filter/* into output/filter
    if output_text_dir.exists() and output_text_dir.is_dir():
        for dataset_dir in output_text_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_filter_dir = dataset_dir / "filter"
            copy_files_in_directory(dataset_filter_dir, output_filter_dir)


if __name__ == "__main__":
    main()
    print("zip -e -r filter-keep-dataset.zip output")