"""Analyze and report frequent n-grams in conversations."""
import argparse
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from orion_tools.common.data_loader import load_jsonl, count_lines


def extract_ngrams(text: str, n: int, no_punctuation: bool = False) -> list[tuple]:
    """Extract n-grams from text."""
    if no_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    words = text.lower().split()
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def analyze_ngrams(
    input_file: str,
    min_ngram: int = 2,
    max_ngram: int = 3,
    min_count: int = 5,
    no_punctuation: bool = False,
    top_n: int = 50
):
    """
    Analyze frequent n-grams in conversations.
    
    Args:
        input_file: Input JSONL file
        min_ngram: Minimum n-gram size
        max_ngram: Maximum n-gram size
        min_count: Minimum occurrence count
        no_punctuation: Remove punctuation before analysis
        top_n: Number of top n-grams to show per size
    """
    total = count_lines(input_file)
    
    # Collect n-grams for each size
    ngram_counters = {n: Counter() for n in range(min_ngram, max_ngram + 1)}
    
    for conversation in tqdm(load_jsonl(input_file), total=total, desc="Analyzing n-grams"):
        for turn in conversation.get("conversations", []):
            if turn.get("value"):
                text = turn["value"]
                for n in range(min_ngram, max_ngram + 1):
                    ngrams = extract_ngrams(text, n, no_punctuation)
                    ngram_counters[n].update(ngrams)
    
    # Print results
    print("\n" + "="*80)
    print("N-gram Analysis Results")
    print("="*80)
    
    for n in range(min_ngram, max_ngram + 1):
        counter = ngram_counters[n]
        filtered = {ngram: count for ngram, count in counter.items() if count >= min_count}
        
        print(f"\n{n}-grams (showing top {top_n} with count >= {min_count}):")
        print("-"*80)
        
        for ngram, count in counter.most_common(top_n):
            if count < min_count:
                break
            ngram_str = ' '.join(ngram)
            print(f"  {count:6,}x  {ngram_str}")
        
        print(f"\nTotal unique {n}-grams with count >= {min_count}: {len(filtered):,}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze frequent n-grams in conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run ngram-analyzer data.jsonl
  uv run ngram-analyzer data.jsonl --min-ngram 2 --max-ngram 4
  uv run ngram-analyzer data.jsonl --no-punctuation --min-count 10
        """
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--min-ngram", type=int, default=2,
                       help="Minimum n-gram size (default: 2)")
    parser.add_argument("--max-ngram", type=int, default=3,
                       help="Maximum n-gram size (default: 3)")
    parser.add_argument("--min-count", type=int, default=5,
                       help="Minimum occurrence count (default: 5)")
    parser.add_argument("--no-punctuation", action="store_true",
                       help="Remove punctuation before analysis")
    parser.add_argument("--top-n", type=int, default=50,
                       help="Number of top n-grams to show (default: 50)")
    
    args = parser.parse_args()
    analyze_ngrams(
        args.input,
        args.min_ngram,
        args.max_ngram,
        args.min_count,
        args.no_punctuation,
        args.top_n
    )


if __name__ == "__main__":
    main()
