"""Filter JSONL conversations containing specific phrases."""
import argparse
from pathlib import Path
from typing import List

from orion_tools.common.data_loader import load_jsonl, write_jsonl, get_output_path


def load_filter_phrases(filter_files: List[str]) -> List[str]:
    """Load filter phrases from one or more files."""
    phrases = []
    for filter_file in filter_files:
        with open(filter_file, 'r', encoding='utf-8', errors='replace') as f:
            phrases.extend(line.strip() for line in f if line.strip())
    return phrases


def count_matches(conversation: dict, phrases: List[str]) -> int:
    """Count how many filter phrases appear in GPT responses."""
    count = 0
    for msg in conversation.get("conversations", []):
        if msg.get("from") == "gpt" and msg.get("value"):
            for phrase in phrases:
                if phrase in msg["value"]:
                    count += 1
    return count


def filter_conversations(input_file: str, output_dir: str, filter_files: List[str], threshold: int = 0):
    """
    Filter conversations based on phrase matches.
    
    Args:
        input_file: Input JSONL file
        output_dir: Output directory
        filter_files: List of files containing filter phrases
        threshold: Max allowed matches (default: 0 = remove any match)
    """
    phrases = load_filter_phrases(filter_files)
    print(f"Loaded {len(phrases)} filter phrases")
    
    output_file = get_output_path(input_file, output_dir, "filtered")
    
    kept = 0
    removed = 0
    total_matches = 0
    
    def process():
        nonlocal kept, removed, total_matches
        for conversation in load_jsonl(input_file, show_progress=True):
            matches = count_matches(conversation, phrases)
            total_matches += matches
            
            if matches <= threshold:
                kept += 1
                yield conversation
            else:
                removed += 1
    
    write_jsonl(output_file, process(), show_progress=True)
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Kept:     {kept:,}")
    print(f"  Removed:  {removed:,}")
    print(f"  Total:    {kept + removed:,}")
    print(f"  Matches:  {total_matches:,}")
    print(f"\nOutput: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter conversations containing specific phrases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run phrase-filter data.jsonl output/ --filters slop.txt
  uv run phrase-filter data.jsonl output/ --filters f1.txt f2.txt --threshold 2
        """
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--filters", required=True, nargs="+", help="Filter phrase files")
    parser.add_argument("--threshold", type=int, default=0, help="Max allowed matches (default: 0)")
    
    args = parser.parse_args()
    filter_conversations(args.input, args.output_dir, args.filters, args.threshold)


if __name__ == "__main__":
    main()
