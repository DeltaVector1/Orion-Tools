import json
import argparse
from pathlib import Path
from text_utils import extract_texts


def ends_with_letter_number_comma(text: str) -> bool:
	import re
	return bool(re.search(r'[a-zA-Z0-9,]$', text.strip())) if isinstance(text, str) else False


def main():
	parser = argparse.ArgumentParser(description='Rule-based filter for JSONL using --text path')
	parser.add_argument('--input', required=True)
	parser.add_argument('--output', required=True)
	parser.add_argument('--text', required=True)
	parser.add_argument('--check-blank', action='store_true')
	parser.add_argument('--check-ending', action='store_true')
	args = parser.parse_args()

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
			bad = False
			for t in texts:
				if args.check_blank and (not isinstance(t, str) or not t.strip()):
					bad = True
					break
				if args.check_ending and t and ends_with_letter_number_comma(t):
					bad = True
					break
			if not bad:
				f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
				kept += 1
	print(f"Kept {kept}/{total}")


if __name__ == '__main__':
	main()


