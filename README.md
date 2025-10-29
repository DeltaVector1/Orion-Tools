# Orion Tools

Fast, simple data filtering tools for JSONL datasets. Built with UV for easy installation and usage.

## Quick Start

1. **Install UV** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Install the project**:
```bash
cd Orion-Tools
uv sync
```

3. **Run a tool**:
```bash
uv run phrase-filter data.jsonl output/ --filters slop.txt
```

## Available Tools

All tools follow a consistent pattern:
```bash
uv run <tool-name> <input.jsonl> <output-dir> [options]
```

### ?? Phrase Filter
Remove conversations containing specific phrases.

```bash
# Basic usage
uv run phrase-filter data.jsonl output/ --filters phrases.txt

# Multiple filter files
uv run phrase-filter data.jsonl output/ --filters f1.txt f2.txt

# Allow up to 2 matches before filtering
uv run phrase-filter data.jsonl output/ --filters slop.txt --threshold 2
```

**Options:**
- `--filters` (required): One or more text files with phrases (one per line)
- `--threshold`: Max allowed matches (default: 0)

---

### ? Dataset Filter
Rule-based conversation quality filtering.

```bash
# Basic usage (all checks enabled)
uv run dataset-filter data.jsonl output/

# Disable specific checks
uv run dataset-filter data.jsonl output/ --no-check-blank-turns
```

**Options:**
- `--no-check-blank-turns`: Disable blank turn filtering
- `--no-check-invalid-endings`: Disable invalid ending filtering
- `--no-check-null-gpt`: Disable null GPT filtering
- `--no-check-duplicate-system`: Disable duplicate system message filtering
- `--no-allow-empty-system-role`: Disallow empty system roles

**Checks performed:**
- Blank or empty turns
- Invalid sentence endings
- Null GPT responses
- Duplicate system messages
- Empty system roles

---

### ?? Deduplication
Hash-based conversation deduplication.

```bash
# SHA256 hashing (fast, exact matches)
uv run deduplication data.jsonl output/

# MinHash (fuzzy matching, requires rensa)
uv run deduplication data.jsonl output/ --method minhash

# Custom worker count
uv run deduplication data.jsonl output/ --processes 8
```

**Options:**
- `--method`: `sha256` or `minhash` (default: sha256)
- `--processes`: Number of parallel processes (default: CPU count - 1)
- `--num-perm`: MinHash permutations (default: 128)

**Note:** MinHash requires: `uv pip install rensa`

---

### ?? Complexity Analyzer
Analyze and filter by linguistic complexity (passive voice, nested clauses, etc.).

```bash
# Analyze and generate report
uv run complexity-analyzer data.jsonl output/

# Filter by complexity threshold
uv run complexity-analyzer data.jsonl output/ --threshold 20

# Custom worker count
uv run complexity-analyzer data.jsonl output/ --workers 8
```

**Options:**
- `--threshold`: Filter conversations with complexity ? threshold
- `--workers`: Number of parallel workers (default: CPU count)

**Requires:** `uv pip install spacy && python -m spacy download en_core_web_sm`

---

### ?? Grammar Corrector
Correct grammar in GPT responses using LanguageTool.

```bash
# Basic usage
uv run grammar-maxxer data.jsonl output/

# Verbose mode (show corrections)
uv run grammar-maxxer data.jsonl output/ --verbose

# Disable correction (just copy)
uv run grammar-maxxer data.jsonl output/ --disable-grammar
```

**Options:**
- `--verbose`: Show corrections as they happen
- `--disable-grammar`: Disable correction

**Requires:** `uv pip install language-tool-python`

---

### ?? N-gram Analyzer
Analyze and report frequent n-grams in conversations.

```bash
# Basic usage
uv run ngram-analyzer data.jsonl

# Custom n-gram range
uv run ngram-analyzer data.jsonl --min-ngram 2 --max-ngram 4

# Filter by count and remove punctuation
uv run ngram-analyzer data.jsonl --no-punctuation --min-count 10
```

**Options:**
- `--min-ngram`: Minimum n-gram size (default: 2)
- `--max-ngram`: Maximum n-gram size (default: 3)
- `--min-count`: Minimum occurrence count (default: 5)
- `--no-punctuation`: Remove punctuation before analysis
- `--top-n`: Number of top n-grams to show (default: 50)

---

### ?? Binary Classifier
Filter conversations using vLLM-based binary classification.

```bash
# Basic usage
uv run binary-classifier data.jsonl output/ --threshold 0.55

# Custom model
uv run binary-classifier data.jsonl output/ --threshold 0.7 \
  --model mymodel --tensor-parallel-size 2
```

**Options:**
- `--threshold` (required): Rejection threshold (0.0 to 1.0)
- `--model`: vLLM model name
- `--batch-size`: Processing batch size (default: 32)
- `--tensor-parallel-size`: vLLM tensor parallel size (default: 1)

**Requires:** `uv pip install vllm spacy && python -m spacy download en_core_web_sm`

---

### ? FineWeb Scorer
Score conversations using vLLM classifier (FineWeb-style).

```bash
# Score only
uv run fineweb-scorer data.jsonl output/

# Score and filter
uv run fineweb-scorer data.jsonl output/ --filter-threshold 0.5

# Custom model
uv run fineweb-scorer data.jsonl output/ --model mymodel --batch-size 128
```

**Options:**
- `--model`: vLLM model name (default: Mixtral-8x7B-Instruct)
- `--batch-size`: Processing batch size (default: 64)
- `--tensor-parallel-size`: vLLM tensor parallel size (default: 1)
- `--filter-threshold`: Also write filtered file with score ? threshold

**Requires:** `uv pip install vllm`

---

### ?? SemHash Dedupe
Semantic deduplication using SemHash.

```bash
# Basic usage
uv run semhash-dedupe data.jsonl output/

# Custom threshold
uv run semhash-dedupe data.jsonl output/ --threshold 0.9

# Only compare human messages
uv run semhash-dedupe data.jsonl output/ --mode human_only
```

**Options:**
- `--threshold`: Similarity threshold (default: 0.85)
- `--mode`: Text extraction mode (default: full)
  - `full`: All messages
  - `human_only`: Only human messages
  - `assistant_only`: Only assistant messages
  - `first_turn`: Only first message
- `--min-length`: Minimum text length (default: 10)

**Requires:** `uv pip install semhash`

---

## Data Format

All tools expect JSONL files with ShareGPT-style conversations:

```json
{"conversations": [
  {"from": "human", "value": "Hello!"},
  {"from": "gpt", "value": "Hi there!"}
]}
```

## Output Structure

All tools create output files in the specified directory with descriptive suffixes:
- `phrase-filter` ? `{input}_filtered.jsonl`
- `dataset-filter` ? `{input}_filtered.jsonl`
- `deduplication` ? `{input}_dedup_{method}.jsonl`
- `complexity-analyzer` ? `{input}_analyzed.jsonl`
- `grammar-maxxer` ? `{input}_corrected.jsonl`
- `binary-classifier` ? `{input}_classified_{threshold}.jsonl`
- `fineweb-scorer` ? `{input}_rated.jsonl` (+ optional filtered file)
- `semhash-dedupe` ? `{input}_semhash_{threshold}.jsonl`

## Adding New Tools

The codebase is designed to be simple and extensible:

1. Create a new file in `orion_tools/cli/your_tool.py`
2. Use the common utilities from `orion_tools.common.data_loader`:
   ```python
   from orion_tools.common.data_loader import load_jsonl, write_jsonl, get_output_path
   ```
3. Add your tool to `pyproject.toml` under `[project.scripts]`:
   ```toml
   your-tool = "orion_tools.cli.your_tool:main"
   ```
4. Run `uv sync` to register the new command

## Development

```bash
# Install in development mode
uv sync

# Add a new dependency
uv add package-name

# Run tests (if you add them)
uv run pytest
```

## License

MIT
