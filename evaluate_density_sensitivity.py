import argparse
import json
import torch
import re
from typing import Optional, List, Dict
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from rich.console import Console
from rich.table import Table
from rich.progress import track

def parse_sensitivity(text: str) -> Optional[bool]:
    """
    Parses the generated text to find density sensitivity prediction.
    Returns True, False, or None if not found or if the value is not a clear boolean.
    """
    # Clean up the text - take only the first line or first sentence
    text = text.split('\n')[0].split('.')[0].strip()
    
    # Debug logging
    print(f"\nParsing text: {text}")
    
    # This regex looks for "Density sensitive" followed by optional modifiers and then "True" or "False"
    # It handles variations like "Density sensitive:", "Density sensitive (low):", etc.
    match = re.search(r"Density sensitive[^:]*:\s*(True|False)", text, re.IGNORECASE)
    if match:
        result = match.group(1).lower() == 'true'
        print(f"Found match with colon: {result}")
        return result
    
    # Try alternative format with semicolon
    match = re.search(r"Density sensitive[^;]*;\s*(True|False)", text, re.IGNORECASE)
    if match:
        result = match.group(1).lower() == 'true'
        print(f"Found match with semicolon: {result}")
        return result
    
    # Try without any separator
    match = re.search(r"Density sensitive[^a-zA-Z]*(True|False)", text, re.IGNORECASE)
    if match:
        result = match.group(1).lower() == 'true'
        print(f"Found match without separator: {result}")
        return result
    
    print("No match found")
    return None

def load_test_data(test_file_path: str) -> List[Dict]:
    """Loads the JSONL test file and filters for scorable examples."""
    print(f"Loading test data from {test_file_path}...")
    with open(test_file_path, "r") as f:
        all_examples = [json.loads(line) for line in f]
    
    # Filter for examples where the ground truth is clearly True or False
    scorable_examples = []
    for ex in all_examples:
        true_sensitivity = parse_sensitivity(ex.get("completion", ""))
        if true_sensitivity is not None:
            ex['true_sensitivity'] = true_sensitivity
            scorable_examples.append(ex)
            
    print(f"Found {len(all_examples)} total examples, {len(scorable_examples)} of which are scorable (True/False).")
    return scorable_examples

def run_evaluation(model_dir: str, test_file_path: str, max_length: int = 128, batch_size: int = 4):
    """
    Loads a fine-tuned model, runs predictions on the test set,
    and reports accuracy for density sensitivity prediction.
    """
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    else:
        device = torch.device("cuda")
        print("Using CUDA.")

    try:
        print(f"Loading model and tokenizer from {model_dir}...")
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Set padding side to left for decoder-only architecture
        tokenizer.padding_side = 'left'
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    test_examples = load_test_data(test_file_path)
    if not test_examples:
        print("No scorable examples found in the test file. Aborting.")
        return

    correct_predictions = 0
    results_data = []

    prompts = [ex['prompt'] for ex in test_examples]

    for i in track(range(0, len(prompts), batch_size), description="Generating predictions..."):
        batch_prompts = prompts[i:i+batch_size]
        # Match finetuning tokenization settings
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            # Match finetuning generation settings
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,     # Reduced from 100 to focus on shorter, more precise outputs
                temperature=0.3,       # Reduced from 0.8 for more focused sampling
                top_k=10,             # Reduced from 50 to limit vocabulary choices
                top_p=0.9,            # Slightly reduced from 0.95
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, gen_text in enumerate(generated_texts):
            original_example_index = i + j
            example = test_examples[original_example_index]
            true_sensitivity = example['true_sensitivity']
            
            # The generated text includes the prompt, so we extract the completion part.
            completion_text = gen_text[len(example['prompt']):].strip()
            
            # Debug logging
            print(f"\nPrompt: {example['prompt']}")
            print(f"Generated completion: {completion_text}")
            
            predicted_sensitivity = parse_sensitivity(completion_text)

            is_correct = (true_sensitivity == predicted_sensitivity)
            if is_correct:
                correct_predictions += 1
            
            results_data.append({
                "prompt": example['prompt'],
                "true": str(true_sensitivity),
                "predicted": str(predicted_sensitivity) if predicted_sensitivity is not None else "[N/A]",
                "status": "✅ Correct" if is_correct else "❌ Incorrect"
            })

    # --- Display Results ---
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Prompt", style="dim", width=50)
    table.add_column("True Sensitivity", justify="center")
    table.add_column("LLM Prediction", justify="center")
    table.add_column("Result", justify="center")

    for result in results_data:
        table.add_row(
            result["prompt"],
            result["true"],
            result["predicted"],
            result["status"]
        )
    
    console.print(table)

    # --- Final Score ---
    accuracy = (correct_predictions / len(test_examples)) * 100 if test_examples else 0
    console.print(f"\n[bold]Final Accuracy Score[/bold]: {accuracy:.2f}% ({correct_predictions}/{len(test_examples)} correct)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned GPT-2 model on density sensitivity prediction.")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory where the fine-tuned model is saved."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to the test set JSONL file (e.g., test_set_prompts_completions.jsonl)."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum generation length for the model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generating predictions to speed up the process."
    )
    args = parser.parse_args()
    
    # Note: This script uses the 'rich' library for better terminal output.
    # If you don't have it, please install it via: pip install rich
    
    run_evaluation(args.model_dir, args.test_file, args.max_length, args.batch_size) 