#!/usr/bin/env python
# scripts/train_rationale.py
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def preprocess_for_rationale(examples, tokenizer, max_length=512):
    """Preprocess examples for rationale generation."""
    # Create prompts for rationale generation
    texts = examples["content"]
    rationale_outputs = examples["rationale"]
    
    # Tokenize inputs
    input_encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None
    )
    
    # Tokenize outputs
    with tokenizer.as_target_tokenizer():
        output_encodings = tokenizer(
            rationale_outputs,
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
    parser = argparse.ArgumentParser(description="Train T5 for rationale generation")
    parser.add_argument("--model_path", type=str, default="t5-base", help="Path to base model")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_file", type=str, default=None, help="Path to validation data (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
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
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    
    # Prepare datasets
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(
        lambda examples: preprocess_for_rationale(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = None
    if val_df is not None:
        val_dataset = Dataset.from_pandas(val_df)
        val_dataset = val_dataset.map(
            lambda examples: preprocess_for_rationale(examples, tokenizer, args.max_length),
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
        report_to="none"  # Disable tensorboard
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
    logger.info("Starting rationale generation model training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()