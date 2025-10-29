"""Hash-based deduplication for conversations."""
import argparse
import hashlib
import multiprocessing as mp
from functools import partial
from pathlib import Path

from tqdm import tqdm

from orion_tools.common.data_loader import load_jsonl, write_jsonl, get_output_path, count_lines


def generate_sha256_hash(text: str) -> str:
    """Generate SHA256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def generate_minhash_hash(text: str, num_perm: int = 128):
    """Generate MinHash hash of text (requires rensa)."""
    try:
        from rensa import RMinHash
    except ImportError:
        raise ImportError("MinHash deduplication requires 'rensa' package. Install with: uv pip install rensa")
    
    minhash = RMinHash(num_perm=num_perm, seed=42)
    minhash.update(text.split())
    return tuple(minhash.digest())


def get_conversation_text(conversation: dict) -> str:
    """Extract all text from a conversation."""
    return ''.join(
        turn['value']
        for turn in conversation.get('conversations', [])
        if turn.get('value') is not None
    )


def process_chunk(chunk: list, method: str, num_perm: int):
    """Process a chunk of conversations and return unique ones with their hashes."""
    results = []
    local_hashes = set()
    
    for conversation in chunk:
        text = get_conversation_text(conversation)
        
        if method == 'sha256':
            hash_val = generate_sha256_hash(text)
        else:  # minhash
            hash_val = generate_minhash_hash(text, num_perm)
        
        if hash_val not in local_hashes:
            local_hashes.add(hash_val)
            results.append((conversation, hash_val))
    
    return results


def deduplicate(input_file: str, output_dir: str, method: str = "sha256", num_processes: int | None = None, num_perm: int = 128):
    """
    Deduplicate conversations using hash-based methods.
    
    Args:
        input_file: Input JSONL file
        output_dir: Output directory
        method: 'sha256' or 'minhash'
        num_processes: Number of parallel processes
        num_perm: Number of permutations for MinHash
    """
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    
    output_file = get_output_path(input_file, output_dir, f"dedup_{method}")
    
    # Load all data
    print("Loading data...")
    data = list(load_jsonl(input_file, show_progress=True))
    total = len(data)
    
    # Split into chunks for parallel processing
    chunk_size = max(100, total // (num_processes * 10))
    chunks = [data[i:i + chunk_size] for i in range(0, total, chunk_size)]
    
    print(f"Processing {total:,} conversations with {num_processes} workers...")
    
    # Process chunks in parallel
    with mp.Pool(num_processes) as pool:
        process_func = partial(process_chunk, method=method, num_perm=num_perm)
        chunk_results = list(tqdm(
            pool.imap(process_func, chunks),
            total=len(chunks),
            desc=f"Deduplicating ({method})"
        ))
    
    # Combine results and ensure global uniqueness
    global_hashes = set()
    unique_conversations = []
    
    for chunk_result in chunk_results:
        for conversation, hash_val in chunk_result:
            if hash_val not in global_hashes:
                global_hashes.add(hash_val)
                unique_conversations.append(conversation)
    
    # Write results
    write_jsonl(output_file, iter(unique_conversations), show_progress=True, total=len(unique_conversations))
    
    duplicates = total - len(unique_conversations)
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Original:    {total:,}")
    print(f"  Unique:      {len(unique_conversations):,}")
    print(f"  Duplicates:  {duplicates:,} ({100*duplicates/total:.1f}%)")
    print(f"\nOutput: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate conversations using hash-based methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run deduplication data.jsonl output/
  uv run deduplication data.jsonl output/ --method minhash
  uv run deduplication data.jsonl output/ --processes 8
        """
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--method", choices=['sha256', 'minhash'], default='sha256',
                       help="Deduplication method (default: sha256)")
    parser.add_argument("--processes", type=int, default=None,
                       help="Number of processes (default: CPU count - 1)")
    parser.add_argument("--num-perm", type=int, default=128,
                       help="MinHash permutations (default: 128)")
    
    args = parser.parse_args()
    deduplicate(args.input, args.output_dir, args.method, args.processes, args.num_perm)


if __name__ == "__main__":
    main()
