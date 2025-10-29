"""Rule-based conversation quality filtering."""
import argparse
import re
from pathlib import Path

from orion_tools.common.data_loader import load_jsonl, write_jsonl, get_output_path


def ends_with_letter_number_comma(text: str) -> bool:
    """Check if text ends with letter, number, or comma."""
    if isinstance(text, str):
        return bool(re.search(r'[a-zA-Z0-9,]$', text.strip()))
    return False


def filter_conversation(
    conversation: dict,
    check_blank_turns: bool,
    check_invalid_endings: bool,
    check_null_gpt: bool,
    check_duplicate_system: bool,
    allow_empty_system_role: bool
) -> dict | None:
    """
    Filter a single conversation based on quality rules.
    
    Returns:
        Filtered conversation or None if should be removed
    """
    conversations = conversation.get("conversations", [])
    if not conversations:
        return None
    
    filtered_conversations = []
    
    for i, msg in enumerate(conversations):
        value = msg.get('value')
        role = msg.get('from')
        
        # Check blank turns
        if check_blank_turns:
            if role == "system":
                if value is not None and not isinstance(value, str):
                    return None
            else:
                if not (isinstance(value, str) and value.strip()):
                    return None
        
        # Check invalid endings
        if check_invalid_endings and value and ends_with_letter_number_comma(value):
            return None
        
        # Check null GPT responses
        if check_null_gpt and role == 'gpt' and value is None:
            return None
        
        # Check for duplicate system messages
        if check_duplicate_system and role == 'system' and i < len(conversations) - 1:
            next_msg = conversations[i + 1]
            next_value = next_msg.get('value')
            if (next_msg.get('from') == 'human' and value and next_value 
                and value.strip().lower() == next_value.strip().lower()):
                continue
        
        # Check empty system role
        if role == "system" and not allow_empty_system_role and not value:
            return None
        
        filtered_conversations.append(msg)
    
    # Ensure valid conversation structure
    roles = set(msg.get('from') for msg in filtered_conversations)
    if 'human' not in roles or 'gpt' not in roles:
        return None
    
    # Remove trailing human message
    if filtered_conversations and filtered_conversations[-1].get('from') == 'human':
        filtered_conversations = filtered_conversations[:-1]
    
    if not filtered_conversations:
        return None
    
    return {"conversations": filtered_conversations}


def filter_dataset(
    input_file: str,
    output_dir: str,
    check_blank_turns: bool = True,
    check_invalid_endings: bool = True,
    check_null_gpt: bool = True,
    check_duplicate_system: bool = True,
    allow_empty_system_role: bool = True
):
    """Filter dataset based on conversation quality rules."""
    output_file = get_output_path(input_file, output_dir, "filtered")
    
    kept = 0
    removed = 0
    
    def process():
        nonlocal kept, removed
        for conversation in load_jsonl(input_file, show_progress=True):
            filtered = filter_conversation(
                conversation,
                check_blank_turns,
                check_invalid_endings,
                check_null_gpt,
                check_duplicate_system,
                allow_empty_system_role
            )
            if filtered:
                kept += 1
                yield filtered
            else:
                removed += 1
    
    write_jsonl(output_file, process(), show_progress=True)
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Kept:    {kept:,}")
    print(f"  Removed: {removed:,}")
    print(f"  Total:   {kept + removed:,}")
    print(f"\nOutput: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter conversations based on quality rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run dataset-filter data.jsonl output/
  uv run dataset-filter data.jsonl output/ --no-check-blank-turns
        """
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--no-check-blank-turns", action="store_false", dest="check_blank_turns",
                       help="Disable blank turn filtering")
    parser.add_argument("--no-check-invalid-endings", action="store_false", dest="check_invalid_endings",
                       help="Disable invalid ending filtering")
    parser.add_argument("--no-check-null-gpt", action="store_false", dest="check_null_gpt",
                       help="Disable null GPT filtering")
    parser.add_argument("--no-check-duplicate-system", action="store_false", dest="check_duplicate_system",
                       help="Disable duplicate system message filtering")
    parser.add_argument("--no-allow-empty-system-role", action="store_false", dest="allow_empty_system_role",
                       help="Disallow empty system roles")
    
    args = parser.parse_args()
    filter_dataset(
        args.input,
        args.output_dir,
        args.check_blank_turns,
        args.check_invalid_endings,
        args.check_null_gpt,
        args.check_duplicate_system,
        args.allow_empty_system_role
    )


if __name__ == "__main__":
    main()
