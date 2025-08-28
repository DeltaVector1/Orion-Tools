import logging
import os
import json
import spacy
import asyncio
import aiofiles
import argparse
from tqdm import tqdm
from collections import defaultdict

from common.vllm_classifier import VLLMClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class CharacterSlopFilter:
    def __init__(self, model_name: str, batch_size: int = 64, confidence_margin: float = 0.1, tensor_parallel_size: int = 1):
        self.vllm = VLLMClassifier(model_name=model_name, tensor_parallel_size=tensor_parallel_size)
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")
        self.batch_size = batch_size
        self.confidence_margin = confidence_margin
        self.classification_cache = defaultdict(dict)
        logging.info("vLLM classifier initialized")

    def split_into_sentences(self, text):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def classify_sentences(self, sentences):
        to_classify = [sent for sent in sentences if sent not in self.classification_cache]
        
        if to_classify:
            labels = ["positive", "negative"]
            results = self.vllm.classify_texts(
                to_classify,
                labels=labels,
                system_hint="You are a strict content classifier.",
                instruction="Return only the label.",
                max_tokens=3,
                temperature=0.0,
                batch_size=self.batch_size,
            )
            sentences_combined = to_classify
            for sentence, result in zip(sentences_combined, results):
                self.classification_cache[sentence] = {"label": result.get("label", "negative"), "score": result.get("score", 0.0)}
                
        return [self.classification_cache[sent] for sent in sentences]

    async def filter_conversations(self, filepath, output_filepath):
        filtered_conversations = []
        try:
            async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
                lines = await f.readlines()
                
            for line in tqdm(lines, desc="Processing conversations"):
                try:
                    data = json.loads(line.strip())
                    conversations = data.get("conversations", [])
                    gpt_sentences = []
                    
                    for conversation in conversations:
                        if conversation.get("from") == "gpt":
                            text = conversation.get("value", "")
                            if text:
                                sentences = self.split_into_sentences(text)
                                gpt_sentences.extend(sentences)
                                
                    if gpt_sentences:
                        sentence_results = self.classify_sentences(gpt_sentences)
                        positive_count = sum(
                            1 for result in sentence_results if result["label"] == "positive" and result["score"] > 0.5 + self.confidence_margin
                        )
                        positive_ratio = positive_count / len(gpt_sentences)
                        
                        if positive_ratio <= 0.55:
                            filtered_conversations.append(data)
                            
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logging.error(f"Error decoding line: {e}")
                except Exception as e:
                    logging.error(f"Error processing line: {e}")
                    
            async with aiofiles.open(output_filepath, "w", encoding="utf-8") as f_out:
                for conversation in filtered_conversations:
                    await f_out.write(json.dumps(conversation, ensure_ascii=False) + "\n")
                    
            logging.info(f"Filtered conversations saved to {output_filepath}")
            
        except Exception as e:
            logging.error(f"Error processing file {filepath}: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Filter JSONL conversations using a vLLM text classifier.")
    parser.add_argument("--input", required=True, help="Path to the input JSONL file")
    parser.add_argument("--output-dir", required=True, help="Directory to save the filtered output")
    parser.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="vLLM model to use")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for classification")
    parser.add_argument("--confidence-margin", type=float, default=0.1, help="Confidence margin for classification")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for vLLM")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, os.path.basename(args.input).replace('.jsonl', '_filtered.jsonl'))
    
    slop_filter = CharacterSlopFilter(
        model_name=args.model,
        batch_size=args.batch_size,
        confidence_margin=args.confidence_margin,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    await slop_filter.filter_conversations(args.input, output_file)

if __name__ == "__main__":
    asyncio.run(main())