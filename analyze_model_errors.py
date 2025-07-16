import argparse
import json
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from collections import defaultdict
import re
from typing import List, Dict, Tuple

def parse_sensitivity(text: str) -> Tuple[bool, str]:
    """
    Parses the generated text to find density sensitivity prediction.
    Returns (True/False/None, explanation) tuple.
    """
    # Clean up the text - take only the first line or first sentence
    text = text.split('\n')[0].split('.')[0].strip()
    
    # This regex looks for "Density sensitive" followed by optional modifiers and then "True" or "False"
    match = re.search(r"Density sensitive[^:]*:\s*(True|False)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'true', "Standard format"
    
    # Try alternative format with semicolon
    match = re.search(r"Density sensitive[^;]*;\s*(True|False)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'true', "Semicolon format"
    
    # Try without any separator
    match = re.search(r"Density sensitive[^a-zA-Z]*(True|False)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'true', "No separator format"
    
    return None, "No prediction found"

def analyze_errors(model_dir: str, test_file: str, max_length: int = 128, batch_size: int = 4):
    """
    Analyzes model predictions and errors in detail.
    """
    console = Console()
    
    # Load model and tokenizer
    console.print("[bold blue]Loading model and tokenizer...[/bold blue]")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load test data
    console.print("[bold blue]Loading test data...[/bold blue]")
    with open(test_file, 'r') as f:
        test_examples = [json.loads(line) for line in f]

    # Initialize error analysis containers
    error_patterns = defaultdict(int)
    error_examples = []
    confidence_analysis = defaultdict(int)
    format_analysis = defaultdict(int)

    # Process examples
    console.print("[bold blue]Analyzing predictions...[/bold blue]")
    for i in range(0, len(test_examples), batch_size):
        batch = test_examples[i:i + batch_size]
        prompts = [ex['prompt'] for ex in batch]
        
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                temperature=0.3,
                top_k=10,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, (example, gen_text) in enumerate(zip(batch, generated_texts)):
            completion_text = gen_text[len(example['prompt']):].strip()
            true_sensitivity = parse_sensitivity(example['completion'])[0]
            pred_sensitivity, format_type = parse_sensitivity(completion_text)
            
            # Record format usage
            format_analysis[format_type] += 1
            
            if pred_sensitivity is None:
                error_patterns["No prediction"] += 1
                error_examples.append({
                    "prompt": example['prompt'],
                    "true": true_sensitivity,
                    "predicted": "None",
                    "generated": completion_text,
                    "error_type": "No prediction"
                })
            elif pred_sensitivity != true_sensitivity:
                error_patterns["Wrong prediction"] += 1
                error_examples.append({
                    "prompt": example['prompt'],
                    "true": true_sensitivity,
                    "predicted": pred_sensitivity,
                    "generated": completion_text,
                    "error_type": "Wrong prediction"
                })

    # Display Analysis Results
    console.print("\n[bold green]Error Analysis Results[/bold green]")
    
    # Error Patterns Table
    error_table = Table(title="Error Patterns")
    error_table.add_column("Error Type", style="cyan")
    error_table.add_column("Count", justify="right", style="magenta")
    for error_type, count in error_patterns.items():
        error_table.add_row(error_type, str(count))
    console.print(error_table)

    # Format Analysis Table
    format_table = Table(title="Output Format Analysis")
    format_table.add_column("Format Type", style="cyan")
    format_table.add_column("Count", justify="right", style="magenta")
    for format_type, count in format_analysis.items():
        format_table.add_row(format_type, str(count))
    console.print(format_table)

    # Detailed Error Examples
    if error_examples:
        console.print("\n[bold yellow]Detailed Error Examples[/bold yellow]")
        for i, error in enumerate(error_examples[:5], 1):  # Show first 5 errors
            console.print(Panel(
                f"[bold]Example {i}[/bold]\n"
                f"Prompt: {error['prompt']}\n"
                f"True: {error['true']}\n"
                f"Predicted: {error['predicted']}\n"
                f"Generated: {error['generated']}\n"
                f"Error Type: {error['error_type']}",
                title=f"Error Example {i}",
                border_style="red"
            ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model errors and patterns")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the fine-tuned model"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to the test set JSONL file"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing"
    )
    args = parser.parse_args()
    
    analyze_errors(args.model_dir, args.test_file, args.max_length, args.batch_size) 