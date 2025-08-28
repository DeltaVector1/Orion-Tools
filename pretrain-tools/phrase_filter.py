import json
import argparse
from pathlib import Path
from text_utils import extract_texts


def filter_line(obj, phrases, threshold):
	counts = 0
	for text in phrases['texts']:
		counts += sum(1 for p in phrases['phrases'] if p in text)
	return counts < threshold if threshold is not None else counts == 0


def main():
	parser = argparse.ArgumentParser(description="Filter JSONL by removing lines containing phrases in --filters within --text path")
	parser.add_argument('--input', required=True, help='Input JSONL file')
	parser.add_argument('--output', required=True, help='Output JSONL file')
	parser.add_argument('--text', required=True, help='Dot-path to text field(s), e.g., data.story or messages')
	parser.add_argument('--filters', required=True, nargs='+', help='Path(s) to text files with phrases (one per line)')
	parser.add_argument('--threshold', type=int, default=None, help='Max allowed phrase matches per line (default: 0)')
	args = parser.parse_args()

	phrase_list = []
	for fp in args.filters:
		with open(fp, 'r', encoding='utf-8', errors='replace') as f:
			for line in f:
				line = line.strip()
				if line:
					phrase_list.append(line)

	Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)
	kept = 0
	total = 0
	with open(args.input, 'r', encoding='utf-8') as f_in, open(args.output, 'w', encoding='utf-8') as f_out:
		for raw in f_in:
			raw = raw.strip()
			if not raw:
				continue
			total += 1
			try:
				obj = json.loads(raw)
			except Exception:
				continue
			texts = extract_texts(obj, args.text)
			if not texts:
				continue
			ok = filter_line(obj, {"texts": texts, "phrases": phrase_list}, args.threshold)
			if ok:
				f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
				kept += 1

	print(f"Kept {kept}/{total}")


if __name__ == '__main__':
	main()


