import json
import argparse
import string
from collections import Counter
from tqdm import tqdm
from text_utils import extract_texts


def tokenize(text: str, remove_punct: bool) -> list:
	text = ' '.join(text.split())
	if remove_punct:
		return [t for t in text.lower().split() if t and all(c not in string.punctuation for c in t)]
	return text.lower().split()


def count_ngrams(lines, min_n, max_n, min_count, remove_punct):
	ngram_counts = {n: Counter() for n in range(min_n, max_n + 1)}
	for text in tqdm(lines, desc='Counting ngrams'):
		tokens = tokenize(text, remove_punct)
		for n in range(min_n, max_n + 1):
			for i in range(0, max(0, len(tokens) - n + 1)):
				ngram = tuple(tokens[i:i+n])
				ngram_counts[n][ngram] += 1
	return {n: {" ".join(k): v for k, v in cnt.items() if v >= min_count} for n, cnt in ngram_counts.items()}


def main():
	parser = argparse.ArgumentParser(description='Extract n-grams from JSONL at --text path')
	parser.add_argument('--input', required=True)
	parser.add_argument('--text', required=True)
	parser.add_argument('--min-ngram', type=int, default=3)
	parser.add_argument('--max-ngram', type=int, default=5)
	parser.add_argument('--min-count', type=int, default=2)
	parser.add_argument('--no-punctuation', action='store_true')
	args = parser.parse_args()

	lines = []
	with open(args.input, 'r', encoding='utf-8') as f:
		for raw in f:
			raw = raw.strip()
			if not raw:
				continue
			try:
				obj = json.loads(raw)
			except Exception:
				continue
			texts = extract_texts(obj, args.text)
			lines.extend(texts)

	result = count_ngrams(lines, args.min_ngram, args.max_ngram, args.min_count, args.no_punctuation)
	for n, table in result.items():
		print(f"\nTop {n}-grams:")
		for ngram, cnt in sorted(table.items(), key=lambda x: -x[1])[:50]:
			print(f"{ngram}: {cnt}")


if __name__ == '__main__':
	main()


