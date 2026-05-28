#!/usr/bin/env python3
"""Download olmocr_science_pdfs data from allenai/dolma3_pool with progress tracking."""

import os
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ID = "allenai/dolma3_pool"
REPO_TYPE = "dataset"
LOCAL_DIR = "/data/hulk/yaxin/data_anatomy"

# Limit files per directory
MAX_FILES_PER_DIR = 10

# Directories to download
DIRECTORIES = [
    "data/olmocr_science_pdfs-adult_content",
    "data/olmocr_science_pdfs-art_and_design",
    "data/olmocr_science_pdfs-entertainment",
    "data/olmocr_science_pdfs-games",
    "data/olmocr_science_pdfs-sports_and_fitness",
]


def get_files_for_directory(all_files: list, directory: str, max_files: int = MAX_FILES_PER_DIR) -> list:
    """Filter files that belong to a specific directory, limited to max_files."""
    dir_files = sorted([f for f in all_files if f.startswith(directory + "/")])
    return dir_files[:max_files]


def download_file(filename: str) -> str:
    """Download a single file and return its path."""
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
        local_dir=LOCAL_DIR,
    )
    return local_path


def main():
    print("Fetching file list from allenai/dolma3_pool...")
    all_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    
    # Collect all files to download (first 100 per directory)
    files_to_download = []
    for directory in DIRECTORIES:
        all_dir_files = sorted([f for f in all_files if f.startswith(directory + "/")])
        dir_files = all_dir_files[:MAX_FILES_PER_DIR]
        files_to_download.extend(dir_files)
        print(f"  {directory}: {len(dir_files)}/{len(all_dir_files)} files")
    
    print(f"\nTotal files to download: {len(files_to_download)}")
    
    if not files_to_download:
        print("No files found to download!")
        return
    
    # Download with progress bar
    print("\nDownloading files...")
    failed = []
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_file, f): f for f in files_to_download}
        
        with tqdm(total=len(files_to_download), unit="file", desc="Downloading") as pbar:
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    future.result()
                except Exception as e:
                    failed.append((filename, str(e)))
                pbar.update(1)
    
    # Summary
    print(f"\n✓ Successfully downloaded: {len(files_to_download) - len(failed)} files")
    if failed:
        print(f"✗ Failed: {len(failed)} files")
        for f, err in failed[:5]:
            print(f"  - {f}: {err}")
    
    print(f"\nFiles saved to: {LOCAL_DIR}/data/olmocr_science_pdfs-*/")


if __name__ == "__main__":
    main()

