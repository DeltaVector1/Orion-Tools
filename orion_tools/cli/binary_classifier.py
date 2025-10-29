"""Binary classification filtering using vLLM."""
import argparse
import re

import spacy
from tqdm import tqdm

from orion_tools.common.data_loader import load_jsonl, write_jsonl, get_output_path, count_lines
from orion_tools.common.vllm_classifier import VLLMClassifier


DEFAULT_MODEL = "Dans-DiscountModels/Dans-Classifier-RP-Validity-V1.0.0-396m"


def clean_text(text: str) -> str:
    """Clean text for processing."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = ''.join(c for c in text if c.isprintable())
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text


def extract_sentences(doc) -> list[str]:
    """Extract sentences from spacy doc."""
    return [clean_text(sent.text.strip()) for sent in doc.sents if sent.text.strip()]


def classify_conversations(
    input_file: str,
    output_dir: str,
    threshold: float,
    model: str = DEFAULT_MODEL,
    batch_size: int = 32,
    tensor_parallel_size: int = 1
):
    """
    Filter conversations using binary classification.
    
    Removes conversations where any sentence scores above threshold.
    
    Args:
        input_file: Input JSONL file
        output_dir: Output directory
        threshold: Rejection threshold (0.0 to 1.0)
        model: vLLM model name
        batch_size: Processing batch size
        tensor_parallel_size: vLLM tensor parallel size
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("Threshold must be between 0.0 and 1.0")
    
    output_file = get_output_path(input_file, output_dir, f"classified_{threshold:.2f}")
    
    # Initialize models
    print("Loading models...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.add_pipe("sentencizer")
    
    classifier = VLLMClassifier(model_name=model, tensor_parallel_size=tensor_parallel_size)
    
    total = count_lines(input_file)
    kept = 0
    removed = 0
    
    def process():
        nonlocal kept, removed
        
        # Process in batches
        batch = []
        for conversation in tqdm(load_jsonl(input_file), total=total, desc="Classifying"):
            batch.append(conversation)
            
            if len(batch) >= batch_size:
                for conv, keep in process_batch(batch, classifier, nlp, threshold):
                    if keep:
                        kept += 1
                        yield conv
                    else:
                        removed += 1
                batch = []
        
        # Process remaining
        if batch:
            for conv, keep in process_batch(batch, classifier, nlp, threshold):
                if keep:
                    kept += 1
                    yield conv
                else:
                    removed += 1
    
    write_jsonl(output_file, process())
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Kept:    {kept:,}")
    print(f"  Removed: {removed:,}")
    print(f"  Total:   {kept + removed:,}")
    print(f"\nOutput: {output_file}")


def process_batch(batch: list, classifier: VLLMClassifier, nlp, threshold: float):
    """Process a batch of conversations."""
    # Extract all sentences from all conversations
    all_sentences = []
    sentence_indices = []
    
    for conversation in batch:
        sentences = []
        for turn in conversation.get('conversations', []):
            if turn.get('from') == 'gpt':
                value = clean_text(turn.get('value', ''))
                doc = nlp(value)
                sentences.extend(extract_sentences(doc))
        
        if sentences:
            start_idx = len(all_sentences)
            all_sentences.extend(sentences)
            end_idx = len(all_sentences)
            sentence_indices.append((start_idx, end_idx))
        else:
            sentence_indices.append(None)
    
    # Classify all sentences at once
    if all_sentences:
        classifications = classifier.classify_texts(
            all_sentences,
            labels=["positive", "negative"],
            system_hint="You are a strict content classifier.",
            instruction="Return only the label.",
            max_tokens=3,
            temperature=0.0,
            batch_size=128,
        )
    else:
        classifications = []
    
    # Determine which conversations to keep
    results = []
    for i, indices in enumerate(sentence_indices):
        if indices is None:
            # No sentences to check, keep
            results.append((batch[i], True))
            continue
        
        start_idx, end_idx = indices
        conv_classifications = classifications[start_idx:end_idx]
        
        # Remove if any sentence is classified as positive above threshold
        has_positive = any(
            1.0 if c.get("label") == "positive" else 0.0 > threshold
            for c in conv_classifications
        )
        
        results.append((batch[i], not has_positive))
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Binary classification filtering using vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run binary-classifier data.jsonl output/ --threshold 0.55
  uv run binary-classifier data.jsonl output/ --threshold 0.7 --model mymodel
        """
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--threshold", type=float, required=True,
                       help="Rejection threshold (0.0 to 1.0)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                       help=f"vLLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Processing batch size (default: 32)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="vLLM tensor parallel size (default: 1)")
    
    args = parser.parse_args()
    classify_conversations(
        args.input,
        args.output_dir,
        args.threshold,
        args.model,
        args.batch_size,
        args.tensor_parallel_size
    )


if __name__ == "__main__":
    main()
