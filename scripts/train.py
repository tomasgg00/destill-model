#!/usr/bin/env python
# scripts/train.py
import os
import argparse
import logging
import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def preprocess_for_multitask(examples, tokenizer, max_length=512):
    """Preprocess examples for multi-task training - both label and rationale tasks."""
    # Create prompts for label task
    label_prompts = [f"[label] {text}" for text in examples["content"]]
    
    # Process label format for outputs
    label_outputs = []
    for label in examples["label"]:
        if isinstance(label, str):
            # Keep the label as is if it's already in string format
            label_outputs.append(label)
        else:
            # If it's a boolean or numeric value, convert to string format
            label_outputs.append("TRUE (Factual)" if label else "FALSE (Misinformation)")
    
    # Create prompts for rationale task
    rationale_prompts = [f"[rationale] {text}" for text in examples["content"]]
    rationale_outputs = examples["rationale"]
    
    # Combine inputs and outputs
    all_inputs = label_prompts + rationale_prompts
    all_outputs = label_outputs + rationale_outputs
    
    # Tokenize inputs
    input_encodings = tokenizer(
        all_inputs,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None
    )
    
    # Tokenize outputs
    with tokenizer.as_target_tokenizer():
        output_encodings = tokenizer(
            all_outputs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
    
    model_inputs = {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": output_encodings["input_ids"]
    }
    
    return model_inputs

def main():
    parser = argparse.ArgumentParser(description="Train a model with step-by-step distillation")
    parser.add_argument("--model_path", type=str, default="t5-base", help="Path to the base model")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data")
    parser.add_argument("--val_file", type=str, default=None, help="Path to the validation data (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--label_weight", type=float, default=1.0, help="Weight for label prediction task (not used in this script)")
    parser.add_argument("--rationale_weight", type=float, default=1.0, help="Weight for rationale generation task (not used in this script)")
    parser.add_argument("--true_class_weight", type=float, default=1.0, 
                        help="Extra weight for TRUE class examples (for imbalanced datasets)")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logger.info(f"Loading training data from {args.train_file}")
    train_df = pd.read_json(args.train_file, lines="jsonl" in args.train_file) if args.train_file.endswith((".json", ".jsonl")) else pd.read_csv(args.train_file)
    
    val_df = None
    if args.val_file:
        logger.info(f"Loading validation data from {args.val_file}")
        val_df = pd.read_json(args.val_file, lines="jsonl" in args.val_file) if args.val_file.endswith((".json", ".jsonl")) else pd.read_csv(args.val_file)
    
    # Apply class weighting if requested
    if args.true_class_weight > 1.0:
        logger.info(f"Applying class weighting: TRUE class weight = {args.true_class_weight}")
        
        # Function to standardize labels for consistent processing
        def standardize_label(label):
            if isinstance(label, str):
                if "TRUE" in label.upper() or "FACTUAL" in label.upper():
                    return "TRUE"
                else:
                    return "FALSE"
            else:
                # If it's a boolean or numeric value
                return "TRUE" if label else "FALSE"
        
        # Count classes to report distribution
        train_df["std_label"] = train_df["label"].apply(standardize_label)
        true_count = (train_df["std_label"] == "TRUE").sum()
        false_count = (train_df["std_label"] == "FALSE").sum()
        
        logger.info(f"Label distribution: TRUE={true_count} ({true_count/(true_count+false_count)*100:.1f}%), "
                   f"FALSE={false_count} ({false_count/(true_count+false_count)*100:.1f}%)")
        
        # Duplicate TRUE examples to increase their weight
        if args.true_class_weight > 1.0 and true_count < false_count:
            # Find TRUE examples
            true_samples = train_df[train_df["std_label"] == "TRUE"]
            
            # Calculate how many additional copies we need (ceiling of weight - 1)
            additional_copies = int(args.true_class_weight - 1)
            if additional_copies > 0:
                additional_samples = pd.concat([true_samples] * additional_copies)
                train_df = pd.concat([train_df, additional_samples])
                train_df = train_df.sample(frac=1, random_state=args.random_seed)  # Shuffle
                
                logger.info(f"Added {len(additional_samples)} additional TRUE examples")
                logger.info(f"New training set size: {len(train_df)} examples")
        
        # Remove temporary column
        train_df = train_df.drop(columns=["std_label"])
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    
    # Apply LoRA if requested
    if args.use_lora:
        logger.info(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Prepare datasets
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(
        lambda examples: preprocess_for_multitask(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = None
    if val_df is not None:
        val_dataset = Dataset.from_pandas(val_df)
        val_dataset = val_dataset.map(
            lambda examples: preprocess_for_multitask(examples, tokenizer, args.max_length),
            batched=True,
            remove_columns=val_dataset.column_names
        )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="max_length",
        max_length=args.max_length
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True if val_dataset else False,
        predict_with_generate=True,
        fp16=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        report_to="none"  # Disable tensorboard to avoid dependency issues
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train model
    logger.info("Starting multi-task training for both label and rationale prediction...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()