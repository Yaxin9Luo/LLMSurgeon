import os
import glob
import argparse
from typing import List
from fetch_category_samples import efficient_sample_from_jsonl_zst, write_jsonl

# Define the source directories
DATA_ROOT = '/data/hulk/yaxin/data_anatomy/data'
SOURCES = {
    'finemath': os.path.join(DATA_ROOT, 'finemath-3plus'),
    'olmpdf': os.path.join(DATA_ROOT, 'olmPDF'),
}

# Output filenames
FILENAME_MAP = {
    'finemath': 'finemath.jsonl',
    'olmpdf': 'olmpdf.jsonl',
}

def get_zst_files(directory: str) -> List[str]:
    """Recursively find .jsonl.zst files in a directory."""
    return sorted(glob.glob(os.path.join(directory, '**', '*.jsonl.zst'), recursive=True))

def main():
    parser = argparse.ArgumentParser(description='Fetch samples for Olmo3 categories.')
    parser.add_argument('--n_per_category', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_chars', type=int, default=300)
    parser.add_argument('--max_chars', type=int, default=2000)
    parser.add_argument('--out_dir', type=str, default='/data/hulk/yaxin/data_anatomy/data_samples/olmo3')
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    for category, path in SOURCES.items():
        print(f"Processing category: {category} from {path}")
        
        files = get_zst_files(path)
        if not files:
            print(f"No .jsonl.zst files found for {category} in {path}")
            continue
            
        print(f"Found {len(files)} files for {category}")
        
        try:
            samples = efficient_sample_from_jsonl_zst(
                file_paths=files,
                n_samples=args.n_per_category,
                seed=args.seed,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                text_key='text' # Assuming 'text' is the key, verify if needed
            )
            
            out_file = os.path.join(args.out_dir, FILENAME_MAP.get(category, f"{category}.jsonl"))
            write_jsonl(out_file, samples, append=False) # Overwrite or append? Default to overwrite for fresh run
            print(f"Saved {len(samples)} samples to {out_file}")
            
        except Exception as e:
            print(f"Error processing {category}: {e}")

if __name__ == '__main__':
    main()

