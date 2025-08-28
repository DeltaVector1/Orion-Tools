#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def clean(data):
    # This function is supposed to remove empty GPT turns.
    # Let's make sure it actually does that.
    cleaned_items = []
    for item in data:
        conversations = item.get("conversations", [])
        filtered_conversations = []
        for msg in conversations:
            # Only keep messages if they aren't from 'gpt' and empty
            # or if they aren't from 'gpt' at all
            if not (msg.get("from") == "gpt" and not msg.get("value", "").strip()):
                filtered_conversations.append(msg)
        
        item["conversations"] = filtered_conversations
        cleaned_items.append(item)
    return cleaned_items

def main():
    parser = argparse.ArgumentParser(description="Remove empty GPT turns from ShareGPT dataset")
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("output_file", help="Path to output JSONL file")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read input file
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): # Only process non-empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line: {line.strip()} - Error: {e}")
                    continue

    # Clean data using the 'clean' function you already defined
    cleaned_data = clean(data)

    # Write cleaned data to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Processed {len(data)} original entries. Wrote {len(cleaned_data)} cleaned entries to {output_path}")

if __name__ == "__main__":
    main()