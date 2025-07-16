# finetune_gpt2_cpu.py

import json
import os
import torch
from typing import Optional, List, Tuple, Dict
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# ---- Helper Functions ----

def _initialize_model_and_tokenizer(model_name: str) -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
    """Initializes the GPT-2 model and tokenizer."""
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set tokenizer.pad_token to tokenizer.eos_token: {tokenizer.eos_token}")

    print(f"Loading model {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    device = torch.device("cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    return model, tokenizer

def _load_and_prepare_data(
    jsonl_file_path: str,
    tokenizer: GPT2TokenizerFast,
    max_seq_length: int,
    num_proc: int,
    functionals: Optional[List[str]],
    subsets: Optional[List[str]],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    """Loads, filters, tokenizes, and splits the dataset."""
    print(f"Loading dataset from {jsonl_file_path}...")
    raw_datasets = load_dataset("json", data_files=jsonl_file_path)

    # --- Filtering ---
    if functionals or subsets:
        print(f"Filtering dataset for Functionals: {functionals} and Subsets: {subsets}...")
        def general_filter_logic(example):
            try:
                prompt_str = example.get('prompt', '')
                if not prompt_str: return False
                current_functional, current_subset = None, None
                parts = prompt_str.split(',')
                for part in parts:
                    if ':' in part:
                        key, val = [p.strip() for p in part.split(':', 1)]
                        if key == "Functional": current_functional = val
                        elif key == "Subset": current_subset = val
                
                functional_passes = not functionals or (current_functional and current_functional in functionals)
                subset_passes = not subsets or (current_subset and current_subset in subsets)
                return functional_passes and subset_passes
            except Exception as e:
                print(f"Error parsing prompt for filtering: '{example.get('prompt', 'PROMPT_NOT_FOUND')}', Error: {e}. Excluding.")
                return False

        if "train" in raw_datasets:
            original_count = len(raw_datasets["train"])
            filtered_dataset = raw_datasets["train"].filter(general_filter_logic, num_proc=num_proc)
            raw_datasets = DatasetDict({"train": filtered_dataset})
            print(f"Original 'train' size: {original_count}, after filtering: {len(raw_datasets['train'])}")
            if not raw_datasets['train']:
                print("Warning: No entries found after filtering. Training might be meaningless.")
        else:
            print("Warning: 'train' split not found. Skipping filtering.")

    # --- Tokenization ---
    def preprocess_function(examples):
        texts = [
            prompt + " " + completion + tokenizer.eos_token
            for prompt, completion in zip(examples["prompt"], examples["completion"])
        ]
        tokenized_inputs = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_seq_length
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    print("Preprocessing dataset...")
    if "train" not in raw_datasets or not raw_datasets["train"]:
        print("Error: 'train' split is missing or empty before tokenization. Cannot proceed.")
        return None, None, None
    
    # Determine columns to remove: all original columns EXCEPT 'prompt' and 'completion'
    original_columns = raw_datasets["train"].column_names
    columns_to_remove_for_tokenization = [col for col in original_columns if col not in ['prompt', 'completion']]
        
    tokenized_datasets = raw_datasets.map(
        preprocess_function, batched=True, num_proc=num_proc, 
        remove_columns=columns_to_remove_for_tokenization
    )
    # Now, tokenized_datasets["train"] contains 'prompt', 'completion', 'input_ids', 'attention_mask', 'labels'

    # --- Dataset Splitting ---
    train_ds, val_ds, test_ds = None, None, None
    if "train" not in tokenized_datasets or not tokenized_datasets["train"]:
        print("Error: 'train' split is missing or empty after tokenization.")
        return None, None, None

    dataset_to_split = tokenized_datasets["train"]
    full_len = len(dataset_to_split)
    print(f"Total data for splitting: {full_len} entries.")

    if full_len == 0:
        print("Warning: Dataset is empty after tokenization. No splits will be created.")
        return None, None, None

    # Validate ratios sum to 1
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if not (0.999 < ratio_sum < 1.001): # Allow for small floating point inaccuracies
        # Normalize ratios if they don't sum to 1, only if sum is not zero
        if ratio_sum > 0.001: 
            adjusted_train_ratio = train_ratio / ratio_sum
            adjusted_validation_ratio = validation_ratio / ratio_sum
            adjusted_test_ratio = test_ratio / ratio_sum
            print(f"Warning: Ratios sum to {ratio_sum}, not 1. Normalizing to: "
                  f"Train: {adjusted_train_ratio:.3f}, Val: {adjusted_validation_ratio:.3f}, Test: {adjusted_test_ratio:.3f}")
            train_ratio, validation_ratio, test_ratio = adjusted_train_ratio, adjusted_validation_ratio, adjusted_test_ratio
        else: # All ratios are zero or very close to zero
            print("Warning: All ratios are zero or sum to zero. No data will be split. Ensure at least one ratio is positive.")
            # If train_ratio was intended to be 1.0 implicitly, this logic won't catch it.
            # The calling function should validate that not all ratios are zero if data is expected.
            return None, None, None # No data to split based on zero ratios

    if test_ratio > 0:
        test_split_size_float = full_len * test_ratio
        if test_split_size_float >= 1:
            test_split_size = max(1, int(test_split_size_float))
            if test_split_size < full_len:
                split1 = dataset_to_split.train_test_split(test_size=test_split_size, shuffle=True, seed=42)
                dataset_to_split = split1["train"]
                test_ds = split1["test"]
            else: 
                test_ds = dataset_to_split
                dataset_to_split = None 
        else:
            print(f"Warning: Not enough data ({full_len}) for a test set with ratio {test_ratio:.3f}. Skipping test set.")

    if dataset_to_split and validation_ratio > 0: 
        current_len = len(dataset_to_split)
        # val_proportion within the remaining train+val part
        train_plus_val_ratio = train_ratio + validation_ratio
        val_within_remainder_ratio = validation_ratio / train_plus_val_ratio if train_plus_val_ratio > 0.001 else 0
        
        val_split_size_float = current_len * val_within_remainder_ratio
        if val_split_size_float >= 1 and val_within_remainder_ratio < 1.0:
            val_split_size = max(1, int(val_split_size_float))
            if val_split_size < current_len:
                split2 = dataset_to_split.train_test_split(test_size=val_split_size, shuffle=True, seed=42)
                train_ds = split2["train"]
                val_ds = split2["test"]
            else:
                val_ds = dataset_to_split
                train_ds = None
        elif val_within_remainder_ratio >= 1.0 and current_len > 0 : 
             val_ds = dataset_to_split
             train_ds = None
        else: 
            train_ds = dataset_to_split 
            if validation_ratio > 0: 
                 print(f"Warning: Not enough data in remainder ({current_len}) for validation set with proportion {val_within_remainder_ratio:.3f}. Remainder assigned to train.")
    elif dataset_to_split: 
        train_ds = dataset_to_split
    
    if train_ds: print(f"Train dataset size: {len(train_ds)}")
    else: print("Train dataset: None or empty.")
    if val_ds: print(f"Validation dataset size: {len(val_ds)}")
    else: print("Validation dataset: None or empty.")
    if test_ds: print(f"Test dataset size: {len(test_ds)}")
    else: print("Test dataset: None or empty.")
    
    return train_ds, val_ds, test_ds

def _get_training_args(
    output_dir: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    save_steps: int,
    logging_steps: int,
    save_total_limit: int,
    num_proc: int,
    eval_dataset_exists: bool
) -> TrainingArguments:
    """Configures and returns TrainingArguments."""
    print("Setting up training arguments...")
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        fp16=False,
        dataloader_num_workers=num_proc,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        # remove_unused_columns=True by default, handles extra 'prompt', 'completion' during training
    )
    if eval_dataset_exists:
        args.evaluation_strategy = "steps"
        args.eval_steps = save_steps
        args.load_best_model_at_end = True
        args.metric_for_best_model = "loss" 
        print("Evaluation on validation set will be performed during training.")
    else:
        args.evaluation_strategy = "no"
        print("No evaluation will be performed during training (no validation set).")
    return args

def _evaluate_and_log(
    trainer: Trainer, 
    eval_dataset: Dataset, 
    dataset_name: str
) -> Dict:
    """Evaluates the model and logs metrics."""
    print(f"Evaluating final model on the {dataset_name} set...")
    # The eval_dataset here should contain 'input_ids', 'attention_mask', 'labels'
    # It might also contain 'prompt', 'completion' if not removed, Trainer handles this.
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    eval_metrics[f"{dataset_name}_samples"] = len(eval_dataset)
    loss_key = 'eval_loss' 
    
    print(f"{dataset_name.capitalize()} Loss: {eval_metrics[loss_key]:.4f}")
    try:
        perplexity = float(torch.exp(torch.tensor(eval_metrics[loss_key])))
        print(f"{dataset_name.capitalize()} Perplexity: {perplexity:.2f}")
        eval_metrics[f'{dataset_name}_perplexity'] = perplexity
    except OverflowError:
        print(f"Could not compute perplexity from {dataset_name} loss: {eval_metrics[loss_key]}")
        eval_metrics[f'{dataset_name}_perplexity'] = float('inf')
    
    specific_loss_key = f'{dataset_name}_loss'
    if loss_key in eval_metrics and loss_key != specific_loss_key:
        eval_metrics[specific_loss_key] = eval_metrics.pop(loss_key)

    trainer.log_metrics(dataset_name, eval_metrics)
    trainer.save_metrics(dataset_name, eval_metrics)
    return eval_metrics

# ---- Main Finetuning Function ----
def run_finetuning(
    jsonl_file_path: str,
    model_name: str = "gpt2",
    output_dir: str = "./gpt2-finetuned-density-cpu",
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    save_steps: int = 100,
    logging_steps: int = 20,
    save_total_limit: int = 3,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_seq_length: int = 128,
    functionals: Optional[List[str]] = None,
    subsets: Optional[List[str]] = None
):
    """
    Fine-tunes a GPT-2 model on a JSONL dataset, optimized for CPU, 
    with train/validation/test splitting and modularized components.
    Saves the test set prompts/completions if a test set is created.
    """
    # Validate Ratios
    if not (train_ratio >= 0 and validation_ratio >= 0 and test_ratio >= 0):
        raise ValueError("Dataset split ratios cannot be negative.")
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if not (0.999 < ratio_sum < 1.001) and ratio_sum > 0.001:
        print(f"Warning: Ratios sum to {ratio_sum}, not 1. They will be normalized in data preparation.")
    elif ratio_sum < 0.001 and (train_ratio > 0 or validation_ratio > 0 or test_ratio > 0):
        pass
    elif ratio_sum < 0.001:
        raise ValueError(
            f"All dataset split ratios (Train: {train_ratio}, Val: {validation_ratio}, Test: {test_ratio}) "
            f"are zero or sum to zero. At least one must be positive to proceed."
        )

    num_proc = max(1, (os.cpu_count() or 4) // 4)
    os.makedirs(output_dir, exist_ok=True) # Ensure output_dir exists for saving test set

    model, tokenizer = _initialize_model_and_tokenizer(model_name)

    train_dataset, val_dataset, test_dataset = _load_and_prepare_data(
        jsonl_file_path, tokenizer, max_seq_length, num_proc,
        functionals, subsets, train_ratio, validation_ratio, test_ratio
    )

    # Save test set prompts and completions if test_dataset exists
    if test_dataset and len(test_dataset) > 0:
        test_data_save_path = os.path.join(output_dir, "test_set_prompts_completions.jsonl")
        print(f"Saving original prompts and completions from the test set to {test_data_save_path}...")
        try:
            with open(test_data_save_path, "w") as f:
                for example in test_dataset:
                    if 'prompt' in example and 'completion' in example:
                        json.dump({"prompt": example["prompt"], "completion": example["completion"]}, f)
                        f.write("\n")
                    else:
                        # This shouldn't happen if _load_and_prepare_data correctly keeps prompt/completion
                        print(f"Warning: 'prompt' or 'completion' field missing in a test_dataset example: {example}. Skipping saving this entry.")
            print(f"Test set prompts and completions saved to {test_data_save_path}.")
        except Exception as e:
            print(f"Error saving test set prompts/completions: {e}")
    elif test_ratio > 0: # Test data was expected but not created (e.g. dataset too small)
        print("Test data was expected (test_ratio > 0) but test_dataset is empty or None. Nothing to save.")
    else:
        print("No test dataset to save (test_ratio is 0 or test_dataset is empty).")

    if not train_dataset and train_ratio > 0:
        print("Critical: Training data is required (train_ratio > 0) but not available after preparation. Aborting training part.")
        # Depending on desired behavior, could return output_dir or raise error
        # For now, will proceed and Trainer will handle no train_dataset if applicable

    training_args = _get_training_args(
        output_dir, num_train_epochs, per_device_train_batch_size,
        gradient_accumulation_steps, learning_rate, weight_decay,
        save_steps, logging_steps, save_total_limit, num_proc,
        eval_dataset_exists=(val_dataset is not None and len(val_dataset) > 0)
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if train_dataset and len(train_dataset)>0 else None, 
        eval_dataset=val_dataset if val_dataset and len(val_dataset)>0 else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    try:
        if train_dataset and len(train_dataset) > 0:
            print("Starting training...")
            train_result = trainer.train()
            trainer.save_model() 
            print("Training complete.")
            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset)
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
        elif train_ratio > 0:
            print("Skipping training as train_dataset is empty or None, though train_ratio > 0.")
        else:
            print("Skipping training as train_ratio is 0.")

        if val_dataset and len(val_dataset) > 0:
            _evaluate_and_log(trainer, val_dataset, "validation")
        else:
            print("No validation set to evaluate.")

        if test_dataset and len(test_dataset) > 0:
            _evaluate_and_log(trainer, test_dataset, "test")
        else:
            print("No test set to evaluate.")

    except Exception as e:
        print(f"An error occurred during the training/evaluation process: {e}")
        import traceback
        traceback.print_exc()

    print(f"Process complete. Model and logs saved to {output_dir if os.path.exists(output_dir) else 'specified output directory (may not exist if error occurred)'}")
    return output_dir

# ---- Test Generation Function (largely unchanged) ----
def test_generation(model_dir, prompt_text, max_length=100):
    """Test the fine-tuned model with a sample prompt"""
    print("\n--- Testing Model Generation ---")
    
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)
    except Exception as e:
        print(f"Error loading model/tokenizer from {model_dir}: {e}")
        return
    
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        try:
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=0.7, top_k=50, top_p=0.95,
                repetition_penalty=1.2, do_sample=True,
                num_return_sequences=1, pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Prompt: {prompt_text}")
            print(f"Generated: {generated_text}")
            return generated_text
        except Exception as e:
            print(f"Error during generation: {e}")
            return

# ---- Main Execution ----
if __name__ == "__main__":
    # os.makedirs("results", exist_ok=True) # Ensure results dir exists if not done by script

    # IMPORTANT: Update this path to point to the new dataset you generated with preprocessing_data.py
    # It will be located in the 'finetuning_sets' directory and will have a timestamp in its name.
    target_jsonl_file = "finetuning_sets/PBE+M06+BLYP_large_finetuned_dataset.jsonl"

    if not os.path.exists(target_jsonl_file):
        print(f"Error: JSONL file not found at {target_jsonl_file}")
        print("Please ensure the file exists or adjust the path in the script.")
    else:
        print(f"Using dataset: {target_jsonl_file}")
        output_model_dir = "./results/gpt2_density_finetuned_cpu_v5"
        
        output_model_dir = run_finetuning(
            jsonl_file_path=target_jsonl_file,
            model_name="gpt2",
            output_dir=output_model_dir,
            num_train_epochs=5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            weight_decay=0.01,
            save_steps=100,
            logging_steps=20,
            max_seq_length=128,
            functionals=None,
            subsets=None,
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15
        )
        
        # if os.path.exists(output_model_dir) and os.listdir(output_model_dir):
        #     print(f"Attempting to test generation from model in: {output_model_dir}")
        #     test_generation(
        #         model_dir=output_model_dir,
        #         prompt_text="Functional: ACONF, SMILES: C" # Updated to a relevant prompt
        #     )
        #     # You can now find the test set data (if created) in:
        #     # os.path.join(output_model_dir, "test_set_prompts_completions.jsonl")
        # else:
        #     print(f"Model output directory {output_model_dir} not found or is empty. Skipping test generation.")
