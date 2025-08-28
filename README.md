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


