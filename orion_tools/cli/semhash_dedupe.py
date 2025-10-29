"""Semantic deduplication using SemHash."""
import argparse
import sys

from tqdm import tqdm

from orion_tools.common.data_loader import load_jsonl, write_jsonl, get_output_path, count_lines

try:
    from semhash import SemHash
except ImportError:
    print("Error: SemHash not installed. Install with: uv pip install semhash")
    sys.exit(1)


def extract_conversation_text(conversation: dict, mode: str = "full") -> str:
    """
    Extract text from conversation based on mode.
    
    Args:
        conversation: Conversation object
        mode: 'full', 'human_only', 'assistant_only', or 'first_turn'
    """
    texts = []
    for turn in conversation.get("conversations", []):
        from_role = turn.get("from", "")
        value = turn.get("value", "")
        
        if mode == "full":
            texts.append(f"{from_role}: {value}")
        elif mode == "human_only" and from_role in ["human", "user"]:
            texts.append(value)
        elif mode == "assistant_only" and from_role in ["gpt", "assistant"]:
            texts.append(value)
        elif mode == "first_turn" and len(texts) == 0:
            texts.append(value)
    
    return " ".join(texts).strip()


def deduplicate(
    input_file: str,
    output_dir: str,
    threshold: float = 0.85,
    mode: str = "full",
    min_length: int = 10
):
    """
    Semantically deduplicate conversations using SemHash.
    
    Args:
        input_file: Input JSONL file
        output_dir: Output directory
        threshold: Similarity threshold (default: 0.85)
        mode: Text extraction mode
        min_length: Minimum text length to consider
    """
    total = count_lines(input_file)
    
    # Load and prepare data
    print(f"Loading conversations (mode: {mode})...")
    conversations = []
    texts = []
    
    for conversation in tqdm(load_jsonl(input_file), total=total, desc="Loading"):
        if "conversations" not in conversation:
            continue
        
        text = extract_conversation_text(conversation, mode)
        if len(text) < min_length:
            continue
        
        conversations.append(conversation)
        texts.append(text)
    
    print(f"Prepared {len(texts):,} valid conversations")
    
    if not texts:
        print("Error: No valid conversations found")
        return
    
    # Perform deduplication
    print("Running SemHash deduplication...")
    semhash = SemHash.from_records(records=texts)
    result = semhash.self_deduplicate(threshold=threshold)
    deduplicated_texts = result.selected
    
    # Map back to original conversations
    print("Mapping deduplicated texts back to conversations...")
    deduplicated_conversations = []
    deduplicated_set = set(deduplicated_texts)
    
    for i, text in enumerate(texts):
        if text in deduplicated_set:
            deduplicated_conversations.append(conversations[i])
            deduplicated_set.remove(text)  # Remove to handle duplicates correctly
    
    # Write results
    output_file = get_output_path(input_file, output_dir, f"semhash_{threshold}")
    write_jsonl(output_file, iter(deduplicated_conversations), show_progress=True, total=len(deduplicated_conversations))
    
    removed = len(texts) - len(deduplicated_conversations)
    reduction = (removed / len(texts) * 100) if texts else 0
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Original:     {len(texts):,}")
    print(f"  Deduplicated: {len(deduplicated_conversations):,}")
    print(f"  Removed:      {removed:,} ({reduction:.1f}%)")
    print(f"\nOutput: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic deduplication using SemHash",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run semhash-dedupe data.jsonl output/
  uv run semhash-dedupe data.jsonl output/ --threshold 0.9
  uv run semhash-dedupe data.jsonl output/ --mode human_only
        """
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.85,
                       help="Similarity threshold (default: 0.85)")
    parser.add_argument("--mode", choices=["full", "human_only", "assistant_only", "first_turn"],
                       default="full", help="Text extraction mode (default: full)")
    parser.add_argument("--min-length", type=int, default=10,
                       help="Minimum text length (default: 10)")
    
    args = parser.parse_args()
    deduplicate(args.input, args.output_dir, args.threshold, args.mode, args.min_length)


if __name__ == "__main__":
    main()
