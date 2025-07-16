import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import argparse

def generate_text(model_dir, prompt_text, max_length=100, temperature=0.3, top_k=50, top_p=0.95, repetition_penalty=1.2):
    """
    Generates text using a fine-tuned GPT-2 model.
    """
    print(f"Loading model and tokenizer from {model_dir}...")
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print(f"Please ensure that '{model_dir}' contains a valid fine-tuned GPT-2 model and tokenizer.")
        return

    # Ensure using CPU
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    print(f"Model and tokenizer loaded successfully. Using device: {device}")

    # Tokenize prompt
    try:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    except Exception as e:
        print(f"Error tokenizing prompt: {e}")
        return

    # Generate text
    print("Generating text...")
    with torch.no_grad():  # No need to track gradients during inference
        try:
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 50256 # 50256 is GPT2's default eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nPrompt: {prompt_text}")
            print(f"Generated: {generated_text}")
            return generated_text
        except Exception as e:
            print(f"Error during text generation: {e}")
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt a fine-tuned GPT-2 model.")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory where the fine-tuned model and tokenizer are saved.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=150,
        help="Maximum length of the generated text.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Controls randomness. Lower is more deterministic.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Filters the K most likely next words.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Filters using nucleus sampling (cumulative probability).",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Penalty for repeating tokens.",
    )

    args = parser.parse_args()

    print("--- Interactive GPT-2 Model Prompting ---")
    print(f"Using model from: {args.model_dir}")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        prompt = input("\nEnter your prompt: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting.")
            break
        if not prompt.strip():
            print("Prompt cannot be empty. Try again.")
            continue

        generate_text(
            model_dir=args.model_dir,
            prompt_text=prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ) 