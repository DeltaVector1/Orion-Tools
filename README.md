## Orion-Tools (Pretraining Utilities)

Small, fast scripts to curate and filter small pretraining JSONL datasets. All tools accept a `--text` dot-path to locate the content field(s) (e.g., `--text data.story`, `--text messages`, `--text samples.text`). If a list is found, all strings are used.

### Tools (in `pretrain-tools/`)

- phrase_filter.py: Remove lines that contain phrases from one or more files at `--filters`. Optional `--threshold` caps allowed matches per line (default: 0). Usage:
  - `python pretrain-tools/phrase_filter.py --input in.jsonl --output out.jsonl --text data.story --filters phrases.txt`

- grammar_maxxer.py: Grammar-correct text at `--text` using LanguageTool. Use `--disable-grammar` to skip correction. Usage:
  - `python pretrain-tools/grammar_maxxer.py --input in.jsonl --output out.jsonl --text data.story`

- ngram_analyzer.py: Print frequent n-grams from text at `--text`. Configure `--min-ngram`, `--max-ngram`, `--min-count`, `--no-punctuation`. Usage:
  - `python pretrain-tools/ngram_analyzer.py --input in.jsonl --text data.story`

- semhash_dedupe.py: Semantic deduplication (SemHash) over concatenated text at `--text`. Keep unique items by similarity `--threshold`. Usage:
  - `python pretrain-tools/semhash_dedupe.py --input in.jsonl --output dedup.jsonl --text data.story --threshold 0.85`

- fineweb_score.py: Score lines via vLLM classifier over `--text`, write scores, optionally filter with `--filter-threshold`. Usage:
  - `python pretrain-tools/fineweb_score.py --input in.jsonl --output rated.jsonl --text data.story --filter-threshold 0.5`

- filterer.py: Lightweight rule-based filter over `--text`. Flags: `--check-blank`, `--check-ending`. Usage:
  - `python pretrain-tools/filterer.py --input in.jsonl --output out.jsonl --text data.story --check-blank --check-ending`

- deduplication.py: Hash-based dedup over `--text` using `--method sha256|minhash` (for MinHash install `rensa`). Usage:
  - `python pretrain-tools/deduplication.py --input in.jsonl --output dedup.jsonl --text data.story --method sha256`

- complexity_analyzer.py: Simple complexity scoring over `--text` (passive voice, length); optional `--threshold` to filter. Usage:
  - `python pretrain-tools/complexity_analyzer.py --input in.jsonl --output out.jsonl --text data.story --threshold 2`

### Notes

- Dot-paths traverse dicts and lists (e.g., `root.messages` applies to each item if `messages` is a list).
- Dependencies vary by tool: `language_tool_python`, `spacy` (with `en_core_web_sm`), `vllm`, `semhash`, `rensa`, `tqdm`, `numpy`. Install per your needs.
## ShareGPT Tools V2

A Small set of scripts i use for data-filtering and annotating. 

### Quick start

1) Create and activate a UV venv
```
uv venv
. .venv/bin/activate
```

2) Install deps
```
uv pip install -r requirements.txt
```

3) Run a tool 
```
# Slop classifier (vLLM)
python regular-tools/DeslopTool_classifier.py --input data.jsonl --output-dir out

# Binary conversation filter (vLLM)
python regular-tools/binary_classification.py --input data.jsonl --output-dir out --threshold 0.55

# FineWeb-style scoring (vLLM) with optional filtering
python regular-tools/FineWeb.py --input data.jsonl --output-dir out --filter-threshold 0.5

# Phrase filter
python regular-tools/phrase_filter.py data.jsonl out --filters filters.txt

# Semantic dedupe (SemHash)
python regular-tools/semhash_dedupe.py data.jsonl out --threshold 0.85

# Complexity analyzer
python regular-tools/complexity_analyzer.py data.jsonl -w 8 -t 20
```


