# finetune_gpt2.py

import json
import os
import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast, # Using Fast tokenizer for efficiency
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

def run_finetuning(
    jsonl_file_path: str,
    model_name: str = "gpt2", # "gpt2", "gpt2-medium", etc.
    output_dir: str = "./gpt2-finetuned-density",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4, # Adjust based on GPU memory
    learning_rate: float = 5e-5,
    save_steps: int = 500, # Save a checkpoint every n steps
    logging_steps: int = 50, # Log training progress every n steps
    test_size: float = 0.1, # Proportion of data to use for evaluation (if desired)
):
    """
    Fine-tunes a GPT-2 model on a JSONL dataset with "prompt" and "completion" fields.
    """
    # --- 1. Load Tokenizer and Model ---
    print(f"Loading tokenizer for {model_name}...")
    # Use GPT2TokenizerFast for better performance
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

    # GPT-2 doesn't have a pad token by default. We'll use the EOS token as the pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}")

    print(f"Loading model {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    # Ensure model's pad_token_id is also set, especially if resizing token embeddings later
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- 2. Load and Prepare Dataset ---
    print(f"Loading dataset from {jsonl_file_path}...")
    # The load_dataset function can directly read jsonl files
    raw_datasets = load_dataset("json", data_files=jsonl_file_path)

    # For causal LM, we typically combine prompt and completion into a single text sequence.
    # The model learns to generate the completion following the prompt.
    def preprocess_function(examples):
        # Concatenate prompt and completion. Add EOS token at the end.
        # Using a clear separator might help the model, but often direct concatenation works.
        # Here, we just add a space.
        texts = [
            prompt + " " + completion + tokenizer.eos_token
            for prompt, completion in zip(examples["prompt"], examples["completion"])
        ]
        # Tokenize the texts
        # Setting truncation=True and padding="max_length" (or another strategy)
        # is important for ensuring all sequences have the same length.
        # `max_length` can be tuned.
        tokenized_inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length", # Pad to the longest sequence in the batch or a fixed max_length
            max_length=512,      # Or tokenizer.model_max_length if defined and suitable
        )
        # For language modeling, the `labels` are typically the `input_ids` themselves.
        # The DataCollatorForLanguageModeling will handle shifting labels for causal LM if needed.
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    print("Preprocessing dataset...")
    # `batched=True` processes multiple elements of the dataset at once for speed.
    # `remove_columns` removes the original text columns after tokenization.
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=os.cpu_count() // 2 if os.cpu_count() > 1 else 1, # Use multiple processors if available
        remove_columns=raw_datasets["train"].column_names,
    )

    # If you want a train/test split:
    if test_size > 0 and "train" in tokenized_datasets:
        print(f"Splitting dataset into train and test ({1-test_size}/{test_size})...")
        split_datasets = tokenized_datasets["train"].train_test_split(test_size=test_size, shuffle=True, seed=42)
        train_dataset = split_datasets["train"]
        eval_dataset = split_datasets["test"]
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Evaluation dataset size: {len(eval_dataset)}")
    elif "train" in tokenized_datasets:
        train_dataset = tokenized_datasets["train"]
        eval_dataset = None
        print(f"Train dataset size: {len(train_dataset)}")
        print("No evaluation dataset will be used during training.")
    else:
        raise ValueError("The dataset does not contain a 'train' split after loading.")


    # --- 3. Data Collator ---
    # Data collator for language modeling. MLM (Masked Language Modeling) is False for GPT-2.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- 4. Training Arguments ---
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size, # Reduce if OOM error
        # per_device_eval_batch_size=per_device_train_batch_size * 2, # Can be larger
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=save_steps if eval_dataset else None, # Evaluate at the same frequency as saving
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=2, # Only keep the last 2 checkpoints
        fp16=torch.cuda.is_available(), # Use mixed precision if a CUDA GPU is available
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None, # Or "perplexity" if you compute it
        report_to="tensorboard", # or "wandb", "none"
        # push_to_hub=False, # Set to True if you want to upload to Hugging Face Hub
    )
    # Simpler eval config if no eval dataset:
    # The logic below is now incorporated directly into TrainingArguments above.
    # if eval_dataset:
    #     training_args.evaluation_strategy = "steps"
    #     training_args.eval_steps = save_steps
    #     training_args.load_best_model_at_end = True
    #     training_args.metric_for_best_model = "loss" # Or "perplexity" if you compute it
    # else:
    #     training_args.evaluation_strategy = "no"


    # --- 5. Trainer ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Pass None if no eval dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 6. Train ---
    print("Starting training...")
    try:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too
        print("Training complete.")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if eval_dataset:
            print("Evaluating final model...")
            eval_metrics = trainer.evaluate()
            # Calculate perplexity if desired (requires a bit more setup or trainer.predict)
            # For simplicity, we'll just log the loss.
            print(f"Evaluation Loss: {eval_metrics['eval_loss']}")
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)


    except Exception as e:
        print(f"An error occurred during training: {e}")
        # You might want to save any partial progress here if applicable
        # For example, if checkpoints were saved, they remain.

    # --- 7. Save Final Model ---
    # The trainer.save_model() above handles this, but an explicit call can be here.
    # final_model_path = os.path.join(output_dir, "final_model")
    # model.save_pretrained(final_model_path)
    # tokenizer.save_pretrained(final_model_path)
    print(f"Fine-tuned model and tokenizer saved to {output_dir}")

    # Example of how to use the fine-tuned model (optional)
    # from transformers import pipeline
    # pipe = pipeline("text-generation", model=output_dir, tokenizer=output_dir, device=0 if torch.cuda.is_available() else -1)
    # sample_prompt = "Subset: SIE4x4, Functional: PBE, SMILES: CC(C)C1" # Example prompt start
    # generated_text = pipe(sample_prompt, max_length=100, num_return_sequences=1)
    # print("\n--- Sample Generation ---")
    # print(f"Prompt: {sample_prompt}")
    # print(f"Generated: {generated_text[0]['generated_text']}")


if __name__ == "__main__":
    # This assumes your `make_finetuning_dataset` script has been run
    # and the output file is available.
    # Let's use the pathname you defined (though it was unused in your script).
    # You'll need to dynamically get the latest timestamped file or specify it.
    # For this example, let's assume a fixed name for simplicity,
    # or you can modify this to find the latest file.

    # --- Create a dummy JSONL file for testing if you don't have one ---
    # dummy_data_path = "dummy_finetuning_data.jsonl"
    # if not os.path.exists(dummy_data_path):
    #     print(f"Creating dummy data file: {dummy_data_path}")
    #     dummy_entries = [
    #         {"prompt": "Subset: A, Functional: X, SMILES: C", "completion": "Density sensitive: True"},
    #         {"prompt": "Subset: B, Functional: Y, SMILES: CC", "completion": "Density sensitive: False"},
    #         {"prompt": "Subset: C, Functional: Z, SMILES: CO", "completion": "Density sensitive: True"},
    #         {"prompt": "Subset: D, Functional: P, SMILES: CN", "completion": "Density sensitive: N/A"}, # Example with N/A
    #     ] * 25 # Make it a bit larger for training to run
    #     with open(dummy_data_path, "w") as f:
    #         for entry in dummy_entries:
    #             json.dump(entry, f)
    #             f.write("\n")
    #     target_jsonl_file = dummy_data_path
    # else:
    #     # Replace this with the actual path to your generated JSONL file
    #     # e.g., target_jsonl_file = "finetuning_sets/finetuned_data_YYYY-MM-DD_HH-MM-SS.jsonl"
    #     target_jsonl_file = dummy_data_path # Default to dummy if you haven't replaced
    #     print(f"Using existing dummy data file: {target_jsonl_file}. Replace with your actual file.")

    target_jsonl_file = "finetuning_sets/finetuned_data_2025-05-30_14-40-00.jsonl"

    # Check if the specified JSONL file exists
    if not os.path.exists(target_jsonl_file):
        print(f"Error: JSONL file not found at {target_jsonl_file}")
        print("Please run your `make_finetuning_dataset` script first or specify the correct path.")
    else:
        run_finetuning(
            jsonl_file_path=target_jsonl_file,
            model_name="gpt2",  # Start with "gpt2" (the smallest version)
            output_dir="./results/gpt2_density_finetuned_test",
            num_train_epochs=1, # For a quick test, usually 3-5
            per_device_train_batch_size=2, # Adjust based on your GPU memory (2 or 4 is a safe start)
            save_steps=100, # Save more frequently for small datasets
            logging_steps=10
        )