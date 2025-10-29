"""FineWeb-style quality scoring using vLLM."""
import argparse
import json
import os

# Set multiprocessing method for CUDA compatibility BEFORE any imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
from tqdm import tqdm

from orion_tools.common.data_loader import get_output_path, count_lines
from orion_tools.common.vllm_classifier import VLLMClassifier


DEFAULT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"


def clean_output(obj: dict) -> dict:
    """Remove extra fields from conversation object."""
    if not obj or "conversations" not in obj:
        return obj
    
    clean_obj = {"conversations": []}
    for turn in obj["conversations"]:
        if turn.get("from") in ["human", "gpt", "system"]:
            clean_turn = {
                "from": turn["from"],
                "value": turn["value"]
            }
            clean_obj["conversations"].append(clean_turn)
    return clean_obj


def score_conversations(
    input_file: str,
    output_dir: str,
    model: str = DEFAULT_MODEL,
    batch_size: int = 64,
    tensor_parallel_size: int = 1,
    filter_threshold: float | None = None
):
    """
    Score conversations using vLLM classifier.
    
    Args:
        input_file: Input JSONL file
        output_dir: Output directory
        model: vLLM model name
        batch_size: Processing batch size
        tensor_parallel_size: vLLM tensor parallel size
        filter_threshold: If set, also write filtered file with score >= threshold
    """
    print("Loading model...")
    classifier = VLLMClassifier(model_name=model, tensor_parallel_size=tensor_parallel_size)
    
    rated_file = get_output_path(input_file, output_dir, "rated")
    total = count_lines(input_file)
    
    all_scores = []
    
    print("Scoring conversations...")
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(rated_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total, desc="Rating"):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            
            if "conversations" not in obj or not isinstance(obj["conversations"], list):
                obj["score"] = 0.0
            else:
                # Extract all message texts
                texts = [msg.get("value", "") for msg in obj["conversations"]]
                
                # Score each message
                results = classifier.classify_texts(
                    texts,
                    labels=["positive", "negative"],
                    system_hint="You are a strict content classifier.",
                    instruction="Return only the label.",
                    batch_size=batch_size,
                )
                
                # Convert to numeric scores and average
                scores = [1.0 if r.get("label") == "positive" else 0.0 for r in results] or [0.0]
                obj["score"] = float(np.mean(scores))
            
            f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
            all_scores.append(obj["score"])
    
    print(f"\n{'='*60}")
    print(f"Rating Statistics ({len(all_scores):,} conversations):")
    print(f"  Min:  {min(all_scores):.3f}")
    print(f"  Max:  {max(all_scores):.3f}")
    print(f"  Mean: {np.mean(all_scores):.3f}")
    print(f"\nOutput: {rated_file}")
    
    # Optional filtering
    if filter_threshold is not None:
        print(f"\nFiltering with threshold >= {filter_threshold:.2f}...")
        filtered_file = get_output_path(input_file, output_dir, f"filtered_{filter_threshold:.2f}")
        
        kept = 0
        with open(rated_file, 'r', encoding='utf-8') as f_in, \
             open(filtered_file, 'w', encoding='utf-8') as f_out:
            
            for line in tqdm(f_in, total=len(all_scores), desc="Filtering"):
                try:
                    obj = json.loads(line)
                    if obj.get("score", -1) >= filter_threshold:
                        clean_obj = clean_output(obj)
                        f_out.write(json.dumps(clean_obj, ensure_ascii=False) + '\n')
                        kept += 1
                except Exception:
                    continue
        
        print(f"\n{'='*60}")
        print(f"Filtering Results:")
        print(f"  Kept:    {kept:,}")
        print(f"  Removed: {len(all_scores) - kept:,}")
        print(f"\nOutput: {filtered_file}")


def main():
    import multiprocessing
    # Set spawn method for CUDA compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    parser = argparse.ArgumentParser(
        description="FineWeb-style quality scoring using vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run fineweb-scorer data.jsonl output/
  uv run fineweb-scorer data.jsonl output/ --filter-threshold 0.5
  uv run fineweb-scorer data.jsonl output/ --model mymodel --batch-size 128
        """
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                       help=f"vLLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Processing batch size (default: 64)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="vLLM tensor parallel size (default: 1)")
    parser.add_argument("--filter-threshold", type=float, default=None,
                       help="Also write filtered file with score >= threshold")
    
    args = parser.parse_args()
    score_conversations(
        args.input,
        args.output_dir,
        args.model,
        args.batch_size,
        args.tensor_parallel_size,
        args.filter_threshold
    )


if __name__ == "__main__":
    main()
