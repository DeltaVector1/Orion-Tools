"""Grammar correction using LanguageTool."""
import argparse
from pathlib import Path

import language_tool_python
from tqdm import tqdm

from orion_tools.common.data_loader import load_jsonl, write_jsonl, get_output_path, count_lines


def correct_text(text: str, tool: language_tool_python.LanguageTool) -> str:
    """Correct grammar using LanguageTool."""
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)
    return corrected.strip()


def correct_conversations(input_file: str, output_dir: str, enable_grammar: bool = True, verbose: bool = False):
    """
    Correct grammar in GPT responses.
    
    Args:
        input_file: Input JSONL file
        output_dir: Output directory
        enable_grammar: Enable grammar correction
        verbose: Show corrections as they happen
    """
    output_file = get_output_path(input_file, output_dir, "corrected")
    
    if not enable_grammar:
        print("Grammar correction is disabled, copying file...")
        def process():
            for conv in load_jsonl(input_file, show_progress=True):
                yield conv
        write_jsonl(output_file, process(), show_progress=True)
        print(f"Output: {output_file}")
        return
    
    tool = language_tool_python.LanguageTool('en-US')
    total = count_lines(input_file)
    corrections_made = 0
    
    def process():
        nonlocal corrections_made
        for conversation in tqdm(load_jsonl(input_file), total=total, desc="Correcting grammar"):
            for turn in conversation.get('conversations', []):
                if turn.get('from') == 'gpt' and turn.get('value'):
                    original = turn['value']
                    corrected = correct_text(original, tool)
                    
                    if original != corrected:
                        corrections_made += 1
                        if verbose:
                            tqdm.write(f"\nOriginal:  {original[:100]}...")
                            tqdm.write(f"Corrected: {corrected[:100]}...")
                            tqdm.write("-" * 60)
                    
                    turn['value'] = corrected
            yield conversation
    
    write_jsonl(output_file, process())
    
    print(f"\n{'='*60}")
    print(f"Made {corrections_made:,} corrections")
    print(f"Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Correct grammar in GPT responses using LanguageTool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run grammar-maxxer data.jsonl output/
  uv run grammar-maxxer data.jsonl output/ --disable-grammar
  uv run grammar-maxxer data.jsonl output/ --verbose
        """
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--disable-grammar", action="store_false", dest="enable_grammar",
                       help="Disable grammar correction")
    parser.add_argument("--verbose", action="store_true",
                       help="Show corrections as they happen")
    
    args = parser.parse_args()
    correct_conversations(args.input, args.output_dir, args.enable_grammar, args.verbose)


if __name__ == "__main__":
    main()
