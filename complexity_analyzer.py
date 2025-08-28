import json
import argparse
import spacy
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

MAX_PARTICIPLES = 2
MAX_NESTED_CLAUSES = 2

nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

def analyze_complexity(text):
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
		for token in sent:
			if token.dep_ in ["nsubjpass", "auxpass"]:
				if not analysis["has_passive_voice"]:
					analysis["reasons"].append("Contains passive voice")
					analysis["has_passive_voice"] = True
					analysis["score"] += 1
				break

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
	line, _, _ = args
	obj = json.loads(line)
	conversation_complexity = 0  

	if "conversations" in obj:
		for turn in obj["conversations"]:
			if turn["from"] in ["human", "gpt"] and turn["value"]:
				turn["complexity_analysis"] = analyze_complexity(turn["value"])
				conversation_complexity += turn["complexity_analysis"]["score"]

		obj["total_conversation_complexity"] = conversation_complexity
		obj["is_complex"] = False

	return json.dumps(obj, ensure_ascii=False)

def process_jsonl(input_file, output_file, num_workers, threshold=None):
	input_path = Path(input_file)
	with input_path.open('r') as f_in, open(output_file, 'w') as f_out:
		total_lines = sum(1 for _ in f_in)
		f_in.seek(0)
		with Pool(num_workers) as pool:
			for result in tqdm(pool.imap_unordered(process_line, ((line, False, 0) for line in f_in), chunksize=10), total=total_lines, desc="Processing"):
				if result:
					obj = json.loads(result)
					if threshold is None or obj["total_conversation_complexity"] <= threshold:
						f_out.write(result + '\n')

def main():
	parser = argparse.ArgumentParser(description="Analyze conversation complexity and filter by threshold.")
	parser.add_argument('input_file', type=str, help='Input JSONL file path')
	parser.add_argument('-w', '--workers', type=int, default=cpu_count(), help="Number of worker processes")
	parser.add_argument('-t', '--threshold', type=int, help="Filter by complexity threshold (skips report generation)")
	args = parser.parse_args()

	input_path = Path(args.input_file)
	output_path = input_path.parent / f"{input_path.stem}_analyzed.jsonl"
	
	print("Processing file...")
	if args.threshold is not None:
		print(f"Filtering conversations with complexity score <= {args.threshold}")
	process_jsonl(input_path, output_path, args.workers, args.threshold)

	if args.threshold is None:
		print("\nGenerating threshold report...")
		print("\nThreshold\tKept Lines\tPercentage")
		print("----------------------------------------")
		with open(output_path, 'r') as f:
			lines = [json.loads(line) for line in f]
			total_lines = len(lines)
			for threshold in range(15, 32):
				kept = sum(1 for obj in lines if obj["total_conversation_complexity"] <= threshold)
				percentage = (kept / total_lines) * 100
				print(f"{threshold}\t\t{kept}\t\t{percentage:.1f}%")

if __name__ == "__main__":
	main()


