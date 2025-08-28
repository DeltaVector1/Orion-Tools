import json
import argparse
from pathlib import Path
import language_tool_python
from tqdm import tqdm
from text_utils import extract_texts, set_texts


def correct_text(tool, text: str) -> str:
	matches = tool.check(text)
	return language_tool_python.utils.correct(text, matches)


def main():
	parser = argparse.ArgumentParser(description='Grammar correct text at --text path in JSONL')
	parser.add_argument('--input', required=True)
	parser.add_argument('--output', required=True)
	parser.add_argument('--text', required=True, help='Dot-path to text field(s) e.g. data.story or messages')
	parser.add_argument('--disable-grammar', action='store_true')
	args = parser.parse_args()

	tool = None if args.disable_grammar else language_tool_python.LanguageTool('en-US')
	Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)

	with open(args.input, 'r', encoding='utf-8') as f_in, open(args.output, 'w', encoding='utf-8') as f_out:
		for raw in tqdm(f_in, desc='Correcting'):
			raw = raw.strip()
			if not raw:
				continue
			try:
				obj = json.loads(raw)
			except Exception:
				continue
			texts = extract_texts(obj, args.text)
			if not texts:
				f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')
				continue
			if tool:
				corrected = [correct_text(tool, t) for t in texts]
				set_texts(obj, args.text, corrected)
			f_out.write(json.dumps(obj, ensure_ascii=False) + '\n')


if __name__ == '__main__':
	main()


