#!/usr/bin/env python3
"""
ShareGPT Semantic Deduplication Script using SemHash
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

try:
	from semhash import SemHash
except ImportError:
	print("Error: SemHash not installed. Please run: pip install semhash")
	sys.exit(1)


def setup_logging(verbose: bool = False):
	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_conversation_text(conversation: List[Dict[str, str]], mode: str = "full") -> str:
	texts = []
	for turn in conversation:
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


def load_sharegpt_jsonl(file_path: str) -> List[Dict[str, Any]]:
	conversations = []
	with open(file_path, 'r', encoding='utf-8') as f:
		for line_num, line in enumerate(f, 1):
			line = line.strip()
			if not line:
				continue
			try:
				data = json.loads(line)
				conversations.append(data)
			except json.JSONDecodeError as e:
				logging.warning(f"Skipping invalid JSON on line {line_num}: {e}")
				continue
	logging.info(f"Loaded {len(conversations)} conversations from {file_path}")
	return conversations


def save_conversations(conversations: List[Dict[str, Any]], output_path: str):
	with open(output_path, 'w', encoding='utf-8') as f:
		for conv in conversations:
			f.write(json.dumps(conv, ensure_ascii=False) + '\n')


def main():
	parser = argparse.ArgumentParser(description="Semantically deduplicate ShareGPT JSONL files using SemHash")
	parser.add_argument("input_file", help="Input ShareGPT JSONL file")
	parser.add_argument("output_dir", help="Output directory for deduplicated files")
	parser.add_argument("--mode", choices=["full", "human_only", "assistant_only", "first_turn"], default="full")
	parser.add_argument("--threshold", type=float, default=0.85)
	parser.add_argument("--min-length", type=int, default=10)
	parser.add_argument("--max-conversations", type=int)
	parser.add_argument("--verbose", "-v", action="store_true")
	parser.add_argument("--save-explanations", action="store_true")
	args = parser.parse_args()

	setup_logging(args.verbose)
	if not os.path.exists(args.input_file):
		logging.error(f"Input file not found: {args.input_file}")
		sys.exit(1)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	logging.info("Loading conversations...")
	conversations = load_sharegpt_jsonl(args.input_file)
	if args.max_conversations:
		conversations = conversations[:args.max_conversations]
		logging.info(f"Limited to {len(conversations)} conversations")
	logging.info(f"Extracting text using mode: {args.mode}")
	texts = []
	valid_indices = []
	for i, conv in enumerate(conversations):
		if "conversations" not in conv:
			logging.warning(f"Skipping conversation {i}: missing 'conversations' field")
			continue
		text = extract_conversation_text(conv["conversations"], args.mode)
		if len(text) < args.min_length:
			logging.debug(f"Skipping conversation {i}: text too short ({len(text)} chars)")
			continue
		texts.append(text)
		valid_indices.append(i)
	logging.info(f"Prepared {len(texts)} conversations for deduplication")
	if not texts:
		logging.error("No valid conversations found for deduplication")
		sys.exit(1)
	logging.info("Initializing SemHash...")
	try:
		semhash = SemHash.from_records(records=texts)
		logging.info("Performing self-deduplication...")
		result = semhash.self_deduplicate(threshold=args.threshold)
		deduplicated_texts = result.selected
		deduplicated_conversations = []
		for dedup_text in deduplicated_texts:
			for i, original_text in enumerate(texts):
				if original_text == dedup_text:
					deduplicated_conversations.append(conversations[valid_indices[i]])
					break
		logging.info(f"Deduplication complete:")
		logging.info(f"  Original: {len(texts)} conversations")
		logging.info(f"  Deduplicated: {len(deduplicated_conversations)} conversations")
		logging.info(f"  Removed: {len(texts) - len(deduplicated_conversations)} conversations")
		logging.info(f"  Reduction: {((len(texts) - len(deduplicated_conversations)) / len(texts) * 100):.1f}%")
	except Exception as e:
		logging.error(f"Deduplication failed: {e}")
		sys.exit(1)
	output_file = output_dir / "deduplicated.jsonl"
	logging.info(f"Saving deduplicated conversations to {output_file}")
	save_conversations(deduplicated_conversations, str(output_file))
	stats_file = output_dir / "deduplication_stats.json"
	stats = {
		"input_file": args.input_file,
		"mode": args.mode,
		"threshold": args.threshold,
		"min_length": args.min_length,
		"original_count": None,
		"valid_count": len(texts),
		"deduplicated_count": len(deduplicated_conversations),
		"removed_count": len(texts) - len(deduplicated_conversations),
		"reduction_percentage": ((len(texts) - len(deduplicated_conversations)) / len(texts) * 100) if texts else 0,
	}
	with open(stats_file, 'w', encoding='utf-8') as f:
		json.dump(stats, f, indent=2, ensure_ascii=False)
	logging.info(f"Statistics saved to {stats_file}")

if __name__ == "__main__":
	main()


