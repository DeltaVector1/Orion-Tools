import json
import argparse
import hashlib
from pathlib import Path
from typing import List
from text_utils import extract_texts

try:
	from rensa import RMinHash
except Exception:
	RMinHash = None


def sha256(text: str) -> str:
	return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
	parser = argparse.ArgumentParser(description='Deduplicate JSONL using sha256 or MinHash over --text path')
	parser.add_argument('--input', required=True)
	parser.add_argument('--output', required=True)
	parser.add_argument('--text', required=True)
	parser.add_argument('--method', choices=['sha256', 'minhash'], default='sha256')
	args = parser.parse_args()

	Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)
	seen = set()
	kept = 0
	with open(args.input, 'r', encoding='utf-8') as f_in, open(args.output, 'w', encoding='utf-8') as f_out:
		for raw in f_in:
			raw = raw.strip()
			if not raw:
				continue
			try:
				obj = json.loads(raw)
			except Exception:
				continue
			texts: List[str] = extract_texts(obj, args.text)
			if not texts:
				continue
			joined = " \n".join(texts)
			if args.method == 'sha256':
				h = sha256(joined)
				if h in seen:
					continue
				seen.add(h)
			else:
				if RMinHash is None:
					raise SystemExit('Please pip install rensa for MinHash support')
				mh = RMinHash(num_perm=128, seed=42)
				mh.update(joined.split())
				dig = tuple(mh.digest())
				if dig in seen:
					continue
				seen.add(dig)
			f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
			kept += 1
	print(f"Kept {kept} unique records")


if __name__ == '__main__':
	main()


