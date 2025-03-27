#!/usr/bin/env python
# scripts/train_classification.py
import os
import argparse
import logging
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def preprocess_for_classification(examples, tokenizer, max_length=512):
    """Preprocess examples for classification."""
    texts = examples["content"]
    
    # Tokenize texts
    tokenized = tokenizer(
        texts, 
        padding="max_length", 
        truncation=True, 
        max_length=max_length,
        return_tensors=None
    )
    
    # Process labels
    labels = []
    for label in examples["label"]:
        if isinstance(label, str):
            if "TRUE" in label.upper() or "FACTUAL" in label.upper():
                labels.append(1)  # TRUE class
            else:
                labels.append(0)  # FALSE class
        else:
            # Handle boolean or numeric values
            labels.append(1 if label else 0)
    
    tokenized["labels"] = labels
    return tokenized

def main():
    parser = argparse.ArgumentParser(description="Train ModernBERT for classification")
    parser.add_argument("--model_path", type=str, default="answerdotai/ModernBERT-base", help="Path to base model")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_file", type=str, default=None, help="Path to validation data (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
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
        
        # Standardize labels
        def standardize_label(label):
            if isinstance(label, str):
                if "TRUE" in label.upper() or "FACTUAL" in label.upper():
                    return "TRUE"
                else:
                    return "FALSE"
            else:
                return "TRUE" if label else "FALSE"
        
        # Count classes 
        train_df["std_label"] = train_df["label"].apply(standardize_label)
        true_count = (train_df["std_label"] == "TRUE").sum()
        false_count = (train_df["std_label"] == "FALSE").sum()
        
        logger.info(f"Label distribution: TRUE={true_count} ({true_count/(true_count+false_count)*100:.1f}%), "
                   f"FALSE={false_count} ({false_count/(true_count+false_count)*100:.1f}%)")
        
        # Duplicate TRUE examples 
        if args.true_class_weight > 1.0 and true_count < false_count:
            # Find TRUE examples
            true_samples = train_df[train_df["std_label"] == "TRUE"]
            
            # Calculate additional copies
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
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=2
    )
    
    # Prepare datasets
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(
        lambda examples: preprocess_for_classification(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = None
    if val_df is not None:
        val_dataset = Dataset.from_pandas(val_df)
        val_dataset = val_dataset.map(
            lambda examples: preprocess_for_classification(examples, tokenizer, args.max_length),
            batched=True,
            remove_columns=val_dataset.column_names
        )
    
    # Set up class weights for loss function
    class_weights = None
    if args.true_class_weight > 1.0:
        class_weights = torch.tensor([1.0, args.true_class_weight]).to(model.device)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="f1",
        report_to="none"  # Disable tensorboard
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # Train model
    logger.info("Starting classification model training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()