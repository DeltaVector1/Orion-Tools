"""Unified data loading and writing utilities."""
import json
from pathlib import Path
from typing import Iterator, Any, Optional
from tqdm import tqdm


def load_jsonl(file_path: str | Path, show_progress: bool = False) -> Iterator[dict]:
    """
    Load JSONL file line by line.
    
    Args:
        file_path: Path to JSONL file
        show_progress: Show progress bar
        
    Yields:
        Parsed JSON objects
    """
    file_path = Path(file_path)
    
    # Count lines if showing progress
    total = None
    if show_progress:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            total = sum(1 for _ in f)
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        iterator = tqdm(f, total=total, desc="Loading") if show_progress else f
        for line in iterator:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                if show_progress:
                    tqdm.write(f"Skipping invalid JSON: {e}")
                continue


def write_jsonl(file_path: str | Path, data: Iterator[dict], show_progress: bool = False, total: Optional[int] = None):
    """
    Write data to JSONL file.
    
    Args:
        file_path: Output file path
        data: Iterator of dictionaries to write
        show_progress: Show progress bar
        total: Total number of items (for progress bar)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        iterator = tqdm(data, total=total, desc="Writing") if show_progress else data
        for obj in iterator:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def count_lines(file_path: str | Path) -> int:
    """Count lines in a file."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return sum(1 for _ in f)


def get_output_path(input_path: str | Path, output_dir: str | Path, suffix: str) -> Path:
    """
    Generate output path based on input file and suffix.
    
    Args:
        input_path: Input file path
        output_dir: Output directory
        suffix: Suffix to add to filename (e.g., "filtered", "deduplicated")
        
    Returns:
        Path object for output file
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}_{suffix}.jsonl"
