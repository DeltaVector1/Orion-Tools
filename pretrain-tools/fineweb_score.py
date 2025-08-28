import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from text_utils import extract_texts
from vllm_classifier import VLLMClassifier


def main():
	parser = argparse.ArgumentParser(description='Score JSONL using vLLM classifier over --text path and optionally filter')
	parser.add_argument('--input', required=True)
	parser.add_argument('--output', required=True)
	parser.add_argument('--text', required=True)
	parser.add_argument('--model', default='mistralai/Mixtral-8x7B-Instruct-v0.1')
	parser.add_argument('--tensor-parallel-size', type=int, default=1)
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--filter-threshold', type=float, default=None)
	args = parser.parse_args()

	Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)
	cls = VLLMClassifier(model_name=args.model, tensor_parallel_size=args.tensor_parallel_size)

	rated_path = Path(args.output)

	all_scores = []
	with open(args.input, 'r', encoding='utf-8') as f_in, open(rated_path, 'w', encoding='utf-8') as f_out:
		for raw in tqdm(f_in, desc='Scoring'):
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
			results = cls.classify_texts(
				texts,
				labels=['positive', 'negative'],
				system_hint='You are a strict content classifier.',
				instruction='Return only the label.',
				batch_size=args.batch_size,
			)
			scores = [1.0 if r.get('label') == 'positive' else 0.0 for r in results] or [0.0]
			obj['score'] = float(np.mean(scores))
			f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
			all_scores.append(obj['score'])

	print(f"Min: {min(all_scores):.2f}, Max: {max(all_scores):.2f}, Mean: {np.mean(all_scores):.2f}")

	if args.filter_threshold is not None:
		filtered_file = rated_path.parent / f"{rated_path.stem}_filtered_{args.filter_threshold:.2f}.jsonl"
		kept = 0
		with open(rated_path, 'r', encoding='utf-8') as f_in, open(filtered_file, 'w', encoding='utf-8') as f_out:
			for line in tqdm(f_in, desc='Filtering'):
				try:
					obj = json.loads(line)
				except Exception:
					continue
				if obj.get('score', -1) >= args.filter_threshold:
					f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
					kept += 1
		print(f"Saved {kept} filtered items to {filtered_file}")


if __name__ == '__main__':
	main()


