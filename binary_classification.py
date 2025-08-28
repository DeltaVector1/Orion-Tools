import spacy
import jsonlines
import json
import os
import re
import argparse
from tqdm import tqdm

from common.vllm_classifier import VLLMClassifier

DEFAULT_MODEL = "Dans-DiscountModels/Dans-Classifier-RP-Validity-V1.0.0-396m"
nlp = None
vllm_cls = None

def process_file(input_file, output_dir, threshold, batch_size):
    if not input_file.endswith('.jsonl'):
        raise ValueError("Input file must be a .jsonl file")
    
    threshold = float(threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("Threshold must be between 0.0 and 1.0.")
    
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}-classified.jsonl")
    
    # Count total lines for progress bar
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with jsonlines.open(input_file, mode='r') as reader, \
         jsonlines.open(output_file, mode='w') as writer, \
         tqdm(total=total_lines, desc="Processing", unit=" conversations") as pbar:
        
        batch = []
        for conversation in reader:
            if validate_json(conversation):
                batch.append(conversation)
                if len(batch) >= batch_size:
                    process_batch(batch, threshold, writer)
                    pbar.update(len(batch))
                    batch = []
        
        # Process remaining items
        if batch:
            process_batch(batch, threshold, writer)
            pbar.update(len(batch))
    
    print(f"Processing complete. Output saved to {output_file}")

def validate_json(obj):
    # Simple validation to ensure we have a conversation object
    if not isinstance(obj, dict) or 'conversations' not in obj:
        return False
    return True

def process_batch(batch, threshold, writer):
    all_sentences = []
    sentence_indices = []
    conversation_map = []
    
    # Extract all sentences from all conversations into a single batch
    for i, conversation in enumerate(batch):
        sentences = []
        for turn in conversation.get('conversations', []):
            if turn.get('from') == 'gpt':
                value = clean_text(turn.get('value', ''))
                doc = nlp(value)
                extracted = extract_sentences(doc)
                sentences.extend(extracted)
        
        if sentences:
            start_idx = len(all_sentences)
            all_sentences.extend(sentences)
            end_idx = len(all_sentences)
            sentence_indices.append((start_idx, end_idx))
        else:
            # No sentences to process, mark as "keep"
            sentence_indices.append(None)
        
        conversation_map.append(conversation)
    
    # Batch process all sentences
    if all_sentences:
        classifications = predict(all_sentences)
    else:
        classifications = []
    
    # Now determine which conversations to keep
    for i, indices in enumerate(sentence_indices):
        if indices is None:
            # No sentences to check, keep the conversation
            writer.write(conversation_map[i])
            continue
            
        start_idx, end_idx = indices
        conversation_classifications = classifications[start_idx:end_idx]
        
        if not any(c['positive'] > threshold for c in conversation_classifications):
            writer.write(conversation_map[i])

def extract_sentences(doc):
    return [clean_text(sent.text.strip()) for sent in doc.sents if sent.text.strip()]

def predict(texts):
    if not texts:
        return []
    labels = ["positive", "negative"]
    raw = vllm_cls.classify_texts(
        texts,
        labels=labels,
        system_hint="You are a strict content classifier.",
        instruction="Return only the label.",
        max_tokens=3,
        temperature=0.0,
        batch_size=128,
    )
    # Map to old schema used by this script
    mapped = []
    for r in raw:
        pos = 1.0 if r.get("label") == "positive" else 0.0
        mapped.append({"positive": pos, "negative": 1.0 - pos})
    return mapped

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = ''.join(c for c in text if c.isprintable())
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text

def validate_utf8(text):
    try:
        if isinstance(text, str):
            text.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter JSONL conversations based on vLLM classification")
    parser.add_argument("--input", required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-dir", required=True, help="Directory to save the classified output")
    parser.add_argument("--threshold", type=float, required=True, help="Rejection threshold (0.0 to 1.0)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing (default: 32)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="vLLM model to use")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for vLLM")
    args = parser.parse_args()
    
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.add_pipe("sentencizer")
    vllm_cls = VLLMClassifier(model_name=args.model, tensor_parallel_size=args.tensor_parallel_size)
    process_file(args.input, args.output_dir, args.threshold, args.batch_size)
