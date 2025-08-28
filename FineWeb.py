import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

from common.vllm_classifier import VLLMClassifier

# So i dont' have to pass through dataset converter again
def clean_output(obj):
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

def main():
    parser = argparse.ArgumentParser(
        description="Use a vLLM classifier to score and optionally filter ShareGPT-style data."
    )
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='mistralai/Mixtral-8x7B-Instruct-v0.1')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--filter-threshold', type=float, default=None, help='If set, write filtered file with items >= threshold')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: file not found at '{input_path}'")
        return
    
    output_dir_path = Path(args.output_dir)
    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error : {e}")
        return
    
    cls = VLLMClassifier(model_name=args.model, tensor_parallel_size=args.tensor_parallel_size)
    base_name = input_path.stem
    rated_file_path = output_dir_path / f"{base_name}_rated.jsonl"
    
    all_scores = []
    with open(input_path, 'r', encoding='utf-8') as f_in, open(rated_file_path, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc="Rating conversations"):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "conversations" not in obj or not isinstance(obj["conversations"], list):
                obj["score"] = 0.0
            else:
                texts = [mes.get("value", "") for mes in obj["conversations"]]
                # Score each message then average, similar to original
                results = cls.classify_texts(
                    texts,
                    labels=["positive", "negative"],
                    system_hint="You are a strict content classifier.",
                    instruction="Return only the label.",
                    batch_size=args.batch_size,
                )
                # Map labels to numeric score {positive:1, negative:0}
                scores = [1.0 if r.get("label") == "positive" else 0.0 for r in results] or [0.0]
                obj["score"] = float(np.mean(scores))
            f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
            all_scores.append(obj["score"])
    
    print(f"\nRating stats ({len(all_scores)} processed):")
    print(f"Min: {min(all_scores):.2f}, Max: {max(all_scores):.2f}, Mean: {np.mean(all_scores):.2f}")
    
    if args.filter_threshold is not None:
        threshold = args.filter_threshold
        filtered_file = output_dir_path / f"{base_name}_filtered_{threshold:.1f}.jsonl"
        kept_count = 0
        with open(rated_file_path, 'r', encoding='utf-8') as f_in, open(filtered_file, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, desc=f"Filtering (threshold ≥ {threshold:.1f})"):
                try:
                    obj = json.loads(line)
                    if obj.get("score", -1) >= threshold:
                        clean_obj = clean_output(obj)
                        f_out.write(json.dumps(clean_obj, ensure_ascii=False) + '\n')
                        kept_count += 1
                except Exception:
                    continue
        print(f"Saved {kept_count} filtered items")

if __name__ == "__main__":
    main()