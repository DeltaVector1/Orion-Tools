"""Analyze and filter conversations by linguistic complexity."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import spacy
from tqdm import tqdm

from orion_tools.common.data_loader import get_output_path


MAX_PARTICIPLES = 2
MAX_NESTED_CLAUSES = 2


def analyze_complexity(text: str, nlp) -> dict:
    """Analyze linguistic complexity of text."""
    doc = nlp(text)
    analysis = {
        "has_passive_voice": False,
        "max_nested_clauses": 0,
        "longest_sentence": 0,
        "num_participle_phrases": 0,
        "num_sentences": len(list(doc.sents)),
        "is_complex": False,
        "reasons": [],
        "score": 0,
    }
    
    for sent in doc.sents:
        # Check for passive voice
        for token in sent:
            if token.dep_ in ["nsubjpass", "auxpass"]:
                if not analysis["has_passive_voice"]:
                    analysis["reasons"].append("Contains passive voice")
                    analysis["has_passive_voice"] = True
                    analysis["score"] += 1
                break
        
        # Count participles and clauses
        sent_participles = sum(1 for token in sent if token.tag_ in ["VBG", "VBN"] and token.dep_ in ["ROOT", "advcl", "acl"])
        sent_clauses = sum(1 for token in sent if token.dep_ in ["ccomp", "xcomp", "advcl", "relcl"])
        
        analysis["max_nested_clauses"] = max(analysis["max_nested_clauses"], sent_clauses)
        analysis["longest_sentence"] = max(analysis["longest_sentence"], len(sent))
        analysis["num_participle_phrases"] = max(analysis["num_participle_phrases"], sent_participles)
        
        if sent_participles > MAX_PARTICIPLES:
            analysis["reasons"].append(f"Too many participle phrases ({sent_participles}) in one sentence")
            analysis["score"] += (sent_participles - MAX_PARTICIPLES)
        
        if sent_clauses > MAX_NESTED_CLAUSES:
            analysis["reasons"].append(f"Deeply nested clauses ({sent_clauses}) in one sentence")
            analysis["score"] += (sent_clauses - MAX_NESTED_CLAUSES)
    
    analysis["is_complex"] = analysis["score"] > 0
    return analysis


def process_line(args):
    """Process a single line (for multiprocessing)."""
    line, nlp_model = args
    obj = json.loads(line)
    
    # Load spacy in each worker
    nlp = spacy.load(nlp_model, disable=["ner", "textcat"])
    
    conversation_complexity = 0
    
    if "conversations" in obj:
        for turn in obj["conversations"]:
            if turn["from"] in ["human", "gpt"] and turn["value"]:
                turn["complexity_analysis"] = analyze_complexity(turn["value"], nlp)
                conversation_complexity += turn["complexity_analysis"]["score"]
        
        obj["total_conversation_complexity"] = conversation_complexity
        obj["is_complex"] = False
    
    return json.dumps(obj, ensure_ascii=False)


def analyze_dataset(input_file: str, output_dir: str, num_workers: int | None = None, threshold: int | None = None):
    """
    Analyze conversation complexity.
    
    Args:
        input_file: Input JSONL file
        output_dir: Output directory
        num_workers: Number of worker processes
        threshold: If set, filter conversations with complexity <= threshold
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    output_file = get_output_path(input_file, output_dir, "analyzed")
    
    print(f"Analyzing with {num_workers} workers...")
    
    # Count lines first
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    # Process with multiprocessing
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        with mp.Pool(num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_line, ((line, "en_core_web_sm") for line in f_in), chunksize=10),
                total=total_lines,
                desc="Analyzing complexity"
            ):
                if result:
                    if threshold is not None:
                        obj = json.loads(result)
                        if obj.get("total_conversation_complexity", 0) <= threshold:
                            f_out.write(result + '\n')
                    else:
                        f_out.write(result + '\n')
    
    # Generate report if no threshold
    if threshold is None:
        print("\n" + "="*60)
        print("Complexity Distribution:")
        print("="*60)
        print("Threshold\tKept\t\tPercentage")
        print("-"*60)
        
        with open(output_file, 'r') as f:
            lines = [json.loads(line) for line in f]
            total = len(lines)
            for t in range(15, 32):
                kept = sum(1 for obj in lines if obj.get("total_conversation_complexity", 0) <= t)
                percentage = (kept / total) * 100 if total > 0 else 0
                print(f"{t}\t\t{kept:,}\t\t{percentage:.1f}%")
    
    print(f"\nOutput: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and filter conversations by linguistic complexity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run complexity-analyzer data.jsonl output/
  uv run complexity-analyzer data.jsonl output/ --threshold 20
  uv run complexity-analyzer data.jsonl output/ --workers 8
        """
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of workers (default: CPU count)")
    parser.add_argument("--threshold", type=int, default=None,
                       help="Filter by complexity threshold")
    
    args = parser.parse_args()
    analyze_dataset(args.input, args.output_dir, args.workers, args.threshold)


if __name__ == "__main__":
    main()
