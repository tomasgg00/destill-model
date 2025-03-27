#!/usr/bin/env python
# train_multitask_model.py
"""
Script to train a multi-task model for refugee/migrant misinformation detection
with both label prediction and rationale generation.
"""

import os
import json
import argparse
import logging
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed
)
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import evaluate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")]
)
logger = logging.getLogger(__name__)

class MultitaskTrainer(Seq2SeqTrainer):
    def __init__(self, alpha=0.5, *args, **kwargs):
        """
        Custom trainer for multi-task learning with weighted loss.
        
        Args:
            alpha (float): Weight for label prediction task (1-alpha for rationale task)
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        logger.info(f"Initialized MultitaskTrainer with alpha={alpha}")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation based on input task type.
        Label task gets weight alpha, rationale task gets weight (1-alpha).
        """
        # Get batch texts to identify task type
        inputs_text = self.tokenizer.batch_decode(
            inputs["input_ids"], skip_special_tokens=True
        )
        
        # Determine if this is a label or rationale task
        is_label_task = all(text.startswith("[label]") for text in inputs_text)
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Apply task-specific weighting
        if is_label_task:
            weighted_loss = self.alpha * loss
        else:
            weighted_loss = (1 - self.alpha) * loss
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss

def set_seed_all(seed=42):
    """Set all seeds for reproducibility."""
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Split into label and rationale predictions
    label_preds = [pred for pred, label in zip(decoded_preds, decoded_labels) 
                  if label in ["TRUE", "FALSE"]]
    label_refs = [label for label in decoded_labels if label in ["TRUE", "FALSE"]]
    
    rationale_preds = [pred for pred, label in zip(decoded_preds, decoded_labels) 
                      if label not in ["TRUE", "FALSE"]]
    rationale_refs = [label for label in decoded_labels if label not in ["TRUE", "FALSE"]]
    
    # Calculate metrics
    results = {}
    
    # Label task metrics
    if label_refs:
        label_accuracy = accuracy_score(label_refs, label_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            label_refs, label_preds, average='weighted', zero_division=0
        )
        
        results.update({
            "label_accuracy": label_accuracy,
            "label_precision": precision,
            "label_recall": recall,
            "label_f1": f1,
        })
        
        # Add per-class metrics
        true_indices = [i for i, label in enumerate(label_refs) if label == "TRUE"]
        false_indices = [i for i, label in enumerate(label_refs) if label == "FALSE"]
        
        if true_indices:
            true_preds = [label_preds[i] for i in true_indices]
            true_refs = [label_refs[i] for i in true_indices]
            true_accuracy = accuracy_score(true_refs, true_preds)
            results["true_accuracy"] = true_accuracy
        
        if false_indices:
            false_preds = [label_preds[i] for i in false_indices]
            false_refs = [label_refs[i] for i in false_indices]
            false_accuracy = accuracy_score(false_refs, false_preds)
            results["false_accuracy"] = false_accuracy
    
    # Rationale task metrics
    if rationale_refs:
        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(
            predictions=rationale_preds, 
            references=rationale_refs,
            use_stemmer=True
        )
        
        results.update({
            "rationale_rouge1": rouge_scores["rouge1"],
            "rationale_rouge2": rouge_scores["rouge2"],
            "rationale_rougeL": rouge_scores["rougeL"],
            "rationale_rougeLsum": rouge_scores["rougeLsum"],
        })
    
    return results

def load_and_preprocess_data(train_file, val_file, tokenizer, max_length=512):
    """
    Load and preprocess data for training.
    
    Args:
        train_file (str): Path to training data
        val_file (str): Path to validation data
        tokenizer: Tokenizer for preprocessing
        max_length (int): Maximum sequence length
    """
    logger.info(f"Loading data from {train_file} and {val_file}")
    
    # Load datasets
    data_files = {
        "train": train_file,
        "validation": val_file
    }
    
    # Determine file format
    if train_file.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_files)
    else:
        # Try auto-detection
        dataset = load_dataset("json", data_files=data_files)
    
    logger.info(f"Loaded {len(dataset['train'])} training examples and {len(dataset['validation'])} validation examples")
    
    # Preprocess function
    def preprocess_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples["input"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        
        # Tokenize outputs
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["output"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Apply preprocessing
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing datasets",
    )
    
    return tokenized_datasets

def main():
    parser = argparse.ArgumentParser(description="Train multi-task model for misinformation detection")
    parser.add_argument("--model_name", type=str, default="t5-base", help="Base model name")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for label task loss")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--report_to", type=str, default="none", help="Reporting integration (none, wandb, etc.)")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load best model at end")
    parser.add_argument("--metric_for_best_model", type=str, default="label_accuracy", 
                       help="Metric for best model")
    parser.add_argument("--greater_is_better", action="store_true", help="Greater is better for metric")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args for reproducibility
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Load tokenizer
    global tokenizer
    logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and preprocess data
    tokenized_datasets = load_and_preprocess_data(
        args.train_file,
        args.val_file,
        tokenizer,
        max_length=args.max_length
    )
    
    # Load model
    logger.info(f"Loading model from {args.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # Apply LoRA if requested
    if args.use_lora:
        logger.info(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q", "v"],
        )
        
        # Prepare model for k-bit training if using quantization
        if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
            model = prepare_model_for_kbit_training(model)
        
        # Get PEFT model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        predict_with_generate=True,
        fp16=args.fp16,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        report_to=args.report_to,
        remove_unused_columns=False,  # Keep all columns
    )
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if args.fp16 else None,
    )
    
    # Create callbacks
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )
    
    # Initialize trainer
    trainer = MultitaskTrainer(
        alpha=args.alpha,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Log final evaluation
    logger.info("Final evaluation")
    eval_results = trainer.evaluate()
    
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)
    
    logger.info(f"Training completed. Model saved to {args.output_dir}")
    
    # Log metrics summary
    logger.info("Evaluation metrics summary:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()