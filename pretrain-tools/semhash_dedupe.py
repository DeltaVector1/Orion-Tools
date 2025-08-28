import json
import argparse
from pathlib import Path
from typing import List
from text_utils import extract_texts

try:
	from semhash import SemHash
except Exception:
	SemHash = None


def main():
	parser = argparse.ArgumentParser(description='Semantic dedupe JSONL using SemHash over --text path')
	parser.add_argument('--input', required=True)
	parser.add_argument('--output', required=True)
	parser.add_argument('--text', required=True)
	parser.add_argument('--threshold', type=float, default=0.85)
	args = parser.parse_args()

	if SemHash is None:
		raise SystemExit('Please pip install semhash')

	Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)
	texts: List[str] = []
	objects: List[dict] = []
	with open(args.input, 'r', encoding='utf-8') as f:
		for raw in f:
			raw = raw.strip()
			if not raw:
				continue
			try:
				obj = json.loads(raw)
			except Exception:
				continue
			content = extract_texts(obj, args.text)
			if not content:
				continue
			texts.append(" \n".join(content))
			objects.append(obj)

	sem = SemHash.from_records(records=texts)
	res = sem.self_deduplicate(threshold=args.threshold)
	selected = set(res.selected)

	kept = 0
	with open(args.output, 'w', encoding='utf-8') as f_out:
		for text, obj in zip(texts, objects):
			if text in selected:
				f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
				kept += 1

	print(f"Kept {kept}/{len(objects)} unique records")


if __name__ == '__main__':
	main()


