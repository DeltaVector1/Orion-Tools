"""Extract seed prompts from ShareGPT datasets with configurable ratios."""
import argparse
import json
import random
import yaml
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset


def extract_sharegpt_prompt(row, extract_system=False, conversation_key="conversations"):
    """Extract the first human prompt (and optionally system) from a ShareGPT conversation."""
    conversations = row.get(conversation_key, [])
    if not conversations:
        return None

    system_prompt = ""
    human_prompt = None

    for msg in conversations:
        role = msg.get("from", msg.get("role", ""))
        value = msg.get("value", msg.get("content", ""))

        if role == "system" and extract_system:
            system_prompt = value or ""
        elif role in ("human", "user"):
            human_prompt = value
            break

    if human_prompt is None:
        return None

    return {"SYSTEM": system_prompt, "PROMPT": human_prompt}


def process_sharegpt_dataset(dataset_name, extract_system, ratio, label=None,
                             subset=None, split="train", conversation_key="conversations", seed=42):
    """Process a ShareGPT-style dataset and yield extracted prompts (streaming)."""
    print(f"\nStreaming {dataset_name}" + (f" ({subset})" if subset else ""))

    try:
        if subset:
            ds = load_dataset(dataset_name, subset, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return

    rng = random.Random(seed)
    extracted = 0
    total_seen = 0

    for row in tqdm(ds, desc="  Processing", leave=False):
        total_seen += 1

        if ratio < 1.0 and rng.random() > ratio:
            continue

        result = extract_sharegpt_prompt(row, extract_system, conversation_key)
        if result:
            result["SOURCE"] = dataset_name
            result["LABEL"] = label or ""
            extracted += 1
            yield result

    print(f"  Processed: {total_seen:,} rows, Extracted: {extracted:,} prompts")


def process_wildguard_dataset(ratio, label=None, seed=42):
    """Process wildguardmix dataset with filtering for harmful prompts (streaming)."""
    print("\nStreaming allenai/wildguardmix (wildguardtrain)")

    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train", streaming=True)

    rng = random.Random(seed)
    extracted = 0
    total_seen = 0
    filtered_count = 0

    for row in tqdm(ds, desc="  Processing", leave=False):
        total_seen += 1

        if not (row.get("prompt_harm_label") == "harmful" and
                row.get("response_refusal_label") == "refusal" and
                row.get("subcategory") != "benign"):
            continue

        filtered_count += 1

        if ratio < 1.0 and rng.random() > ratio:
            continue

        prompt = row.get("prompt", "")
        if prompt:
            yield {"SYSTEM": "", "PROMPT": prompt, "SOURCE": "allenai/wildguardmix", "LABEL": label or ""}
            extracted += 1

    print(f"  Processed: {total_seen:,} rows, Matched filter: {filtered_count:,}, Extracted: {extracted:,} prompts")


def analyze_wildguard_subcategories():
    """Analyze and print subcategory distribution in wildguardmix."""
    print("\nAnalyzing allenai/wildguardmix (wildguardtrain) subcategories (streaming)...")

    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train", streaming=True)

    subcategory_counts = Counter()
    total_harmful_refusal = 0
    total_rows = 0

    for row in tqdm(ds, desc="Analyzing"):
        total_rows += 1
        if row.get("prompt_harm_label") == "harmful" and row.get("response_refusal_label") == "refusal":
            subcategory_counts[row.get("subcategory", "unknown")] += 1
            total_harmful_refusal += 1

    print(f"\nTotal rows: {total_rows:,}")
    print(f"\n{'='*60}")
    print("Subcategory distribution (harmful + refusal only):")
    print(f"{'='*60}")
    print(f"{'Subcategory':<40} {'Count':>10} {'%':>8}")
    print(f"{'-'*60}")

    for subcat, count in sorted(subcategory_counts.items(), key=lambda x: -x[1]):
        pct = (count / total_harmful_refusal) * 100 if total_harmful_refusal > 0 else 0
        print(f"{subcat:<40} {count:>10,} {pct:>7.1f}%")

    print(f"{'-'*60}")
    print(f"{'TOTAL':<40} {total_harmful_refusal:>10,} {'100.0%':>8}")
    print(f"{'='*60}")

    return subcategory_counts


def extract_prompts(config_path, output_path, seed=42, shuffle=True):
    """Extract seed prompts from multiple datasets based on config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    all_prompts = []

    for ds_config in config.get("datasets", []):
        dataset_name = ds_config.get("name")
        if not dataset_name:
            continue

        if dataset_name == "allenai/wildguardmix":
            for prompt in process_wildguard_dataset(
                ds_config.get("ratio", 1.0),
                ds_config.get("label"),
                seed
            ):
                all_prompts.append(prompt)
        else:
            for prompt in process_sharegpt_dataset(
                dataset_name,
                ds_config.get("extract_system", False),
                ds_config.get("ratio", 1.0),
                ds_config.get("label"),
                ds_config.get("subset"),
                ds_config.get("split", "train"),
                ds_config.get("conversation_key", "conversations"),
                seed
            ):
                all_prompts.append(prompt)

    if shuffle:
        print(f"\nShuffling {len(all_prompts):,} prompts...")
        random.seed(seed)
        random.shuffle(all_prompts)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for prompt in tqdm(all_prompts, desc="Writing"):
            f.write(json.dumps(prompt, ensure_ascii=False) + '\n')

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Total prompts: {len(all_prompts):,}")

    source_counts = Counter(p["SOURCE"] for p in all_prompts)
    print("\nBy source:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(all_prompts)) * 100
        print(f"  {source}: {count:,} ({pct:.1f}%)")

    system_count = sum(1 for p in all_prompts if p["SYSTEM"])
    print(f"\nWith system prompt: {system_count:,} ({system_count/len(all_prompts)*100:.1f}%)")
    print(f"\nOutput: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract seed prompts from ShareGPT datasets (streaming)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run seed-prompt-extractor --config config.yaml --output prompts.jsonl
  uv run seed-prompt-extractor --analyze-wildguard
        """
    )

    parser.add_argument("--config", "-c", help="YAML config file path")
    parser.add_argument("--output", "-o", help="Output JSONL file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-shuffle", action="store_true", help="Don't shuffle output")
    parser.add_argument("--analyze-wildguard", action="store_true",
                       help="Analyze wildguardmix subcategories and exit")

    args = parser.parse_args()

    if args.analyze_wildguard:
        analyze_wildguard_subcategories()
        return

    if not args.config or not args.output:
        parser.error("--config and --output are required (unless using --analyze-wildguard)")

    extract_prompts(args.config, args.output, args.seed, not args.no_shuffle)


if __name__ == "__main__":
    main()
