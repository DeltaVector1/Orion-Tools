import json
import argparse
import spacy
from tqdm import tqdm
from pathlib import Path
from text_utils import extract_texts


nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])


def analyze_complexity(text):
	doc = nlp(text)
	analysis = {
		"has_passive_voice": any(tok.dep_ in ["nsubjpass", "auxpass"] for tok in doc),
		"longest_sentence": max((len(s) for s in doc.sents), default=0),
	}
	return int(analysis["has_passive_voice"]) + (1 if analysis["longest_sentence"] > 60 else 0)


def main():
	parser = argparse.ArgumentParser(description='Analyze complexity over --text path and optionally filter by threshold')
	parser.add_argument('--input', required=True)
	parser.add_argument('--output', required=True)
	parser.add_argument('--text', required=True)
	parser.add_argument('--threshold', type=int, default=None)
	args = parser.parse_args()

	Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)
	kept = 0
	with open(args.input, 'r', encoding='utf-8') as f_in, open(args.output, 'w', encoding='utf-8') as f_out:
		for raw in tqdm(f_in, desc='Analyzing'):
			raw = raw.strip()
			if not raw:
				continue
			try:
				obj = json.loads(raw)
			except Exception:
				continue
			texts = extract_texts(obj, args.text)
			if not texts:
				continue
			scores = [analyze_complexity(t) for t in texts]
			obj['complexity_score'] = sum(scores)
			if args.threshold is None or obj['complexity_score'] <= args.threshold:
				f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
				kept += 1
	print(f"Kept {kept} items")


if __name__ == '__main__':
	main()


