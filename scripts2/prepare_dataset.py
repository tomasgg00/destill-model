#!/usr/bin/env python
# prepare_dataset.py
"""
Script to prepare datasets for multi-task learning with label prediction and rationale generation.
This script processes datasets with generated rationales into a format suitable for training.
"""

import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("dataset_preparation.log")]
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load data from various file formats."""
    logger.info(f"Loading data from {file_path}")
    
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return pd.DataFrame(json.load(f))
    elif file_path.endswith(".jsonl"):
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def format_for_multitask(df, text_column="text", label_column="label", 
                        rationale_column="rationale"):
    """Format dataset for multi-task learning with label and rationale tasks."""
    logger.info("Formatting for multi-task learning")
    
    # Ensure columns exist
    required_columns = [text_column, label_column, rationale_column]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset")
    
    # Create label task examples
    label_examples = []
    for _, row in df.iterrows():
        label_examples.append({
            "input": f"[label] {row[text_column]}",
            "output": str(row[label_column])
        })
    
    # Create rationale task examples
    rationale_examples = []
    for _, row in df.iterrows():
        rationale_examples.append({
            "input": f"[rationale] {row[text_column]}",
            "output": str(row[rationale_column])
        })
    
    # Combine both tasks
    combined_examples = label_examples + rationale_examples
    
    # Convert to DataFrame
    df_multitask = pd.DataFrame(combined_examples)
    
    logger.info(f"Created {len(df_multitask)} examples for multi-task learning")
    return df_multitask

def balance_labels(df, label_column="label", random_seed=42):
    """Balance the dataset by upsampling or downsampling to ensure equal class distribution."""
    logger.info("Balancing class distribution")
    
    # Count labels
    label_counts = df[label_column].value_counts()
    logger.info(f"Original distribution: {label_counts.to_dict()}")
    
    # Standardize TRUE/FALSE labels
    df['std_label'] = df[label_column].apply(
        lambda x: "TRUE" if str(x).upper() in ["TRUE", "1", "T"] else "FALSE"
    )
    
    # Split by class
    true_samples = df[df['std_label'] == "TRUE"]
    false_samples = df[df['std_label'] == "FALSE"]
    
    # Find sizes
    true_count = len(true_samples)
    false_count = len(false_samples)
    min_count = min(true_count, false_count)
    max_count = max(true_count, false_count)
    
    logger.info(f"Class counts - TRUE: {true_count}, FALSE: {false_count}")
    
    # Balance dataset
    if true_count < false_count:
        # Upsample TRUE
        if min_count < 10:  # Very small dataset, upsample minority
            logger.info("Upsampling TRUE class (minority)")
            true_samples = true_samples.sample(max_count, replace=True, random_state=random_seed)
        else:  # Downsample majority
            logger.info("Downsampling FALSE class (majority)")
            false_samples = false_samples.sample(min_count, random_state=random_seed)
    else:
        # Upsample FALSE
        if min_count < 10:  # Very small dataset, upsample minority
            logger.info("Upsampling FALSE class (minority)")
            false_samples = false_samples.sample(max_count, replace=True, random_state=random_seed)
        else:  # Downsample majority
            logger.info("Downsampling TRUE class (majority)")
            true_samples = true_samples.sample(min_count, random_state=random_seed)
    
    # Combine and shuffle
    balanced_df = pd.concat([true_samples, false_samples]).sample(
        frac=1, random_state=random_seed
    )
    
    # Remove temporary column
    balanced_df = balanced_df.drop(columns=['std_label'])
    
    # Log new distribution
    new_counts = balanced_df[label_column].value_counts()
    logger.info(f"Balanced distribution: {new_counts.to_dict()}")
    
    return balanced_df

def fix_labels(df, label_column="label"):
    """Standardize labels to TRUE/FALSE format."""
    logger.info("Standardizing labels")
    
    def standardize_label(label):
        if isinstance(label, str):
            if "TRUE" in label.upper() or "FACTUAL" in label.upper() or label.upper() in ["T", "1", "YES"]:
                return "TRUE"
            else:
                return "FALSE"
        elif isinstance(label, bool):
            return "TRUE" if label else "FALSE"
        elif isinstance(label, (int, float)):
            return "TRUE" if label > 0 else "FALSE"
        else:
            return "FALSE"  # Default case
    
    df[label_column] = df[label_column].apply(standardize_label)
    
    # Log class distribution
    label_counts = df[label_column].value_counts()
    logger.info(f"Label distribution after standardization: {label_counts.to_dict()}")
    
    return df

def clean_rationales(df, rationale_column="rationale", max_length=1024):
    """Clean and truncate rationales."""
    logger.info("Cleaning rationales")
    
    # Function to clean individual rationale
    def clean_rationale(rationale):
        if not isinstance(rationale, str):
            return ""
        
        # Remove special tokens and formatting
        clean = rationale.replace("[/INST]", "").replace("<end_of_turn>", "").strip()
        
        # Truncate if too long
        if len(clean) > max_length:
            clean = clean[:max_length] + "..."
        
        return clean
    
    df[rationale_column] = df[rationale_column].apply(clean_rationale)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for multi-task training")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input file with rationales")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--text_column", type=str, default="text", help="Column name for text data")
    parser.add_argument("--label_column", type=str, default="label", help="Column name for label data")
    parser.add_argument("--rationale_column", type=str, default="rationale", help="Column name for rationales")
    parser.add_argument("--val_size", type=float, default=0.1, help="Size of validation split")
    parser.add_argument("--test_size", type=float, default=0.1, help="Size of test split")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--balance_labels", action="store_true", help="Balance class distribution")
    parser.add_argument("--validation_file", type=str, default=None, help="Optional external validation file")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df = load_data(args.input_file)
    logger.info(f"Loaded {len(df)} examples from {args.input_file}")
    
    # Clean and standardize data
    df = fix_labels(df, label_column=args.label_column)
    df = clean_rationales(df, rationale_column=args.rationale_column)
    
    # Balance labels if requested
    if args.balance_labels:
        df = balance_labels(df, label_column=args.label_column, random_seed=args.random_seed)
    
    # Split into train/val/test
    if args.validation_file:
        # Use external validation file
        val_df = load_data(args.validation_file)
        logger.info(f"Using external validation set with {len(val_df)} examples")
        
        # Standardize validation data
        val_df = fix_labels(val_df, label_column=args.label_column)
        if args.rationale_column in val_df.columns:
            val_df = clean_rationales(val_df, rationale_column=args.rationale_column)
        else:
            # If validation data doesn't have rationales, use empty strings
            logger.warning(f"Validation data doesn't have rationales, using empty strings")
            val_df[args.rationale_column] = ""
        
        # Split remaining data into train/test
        train_df, test_df = train_test_split(
            df, 
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=df[args.label_column]
        )
    else:
        # Split into train/val/test
        train_df, temp_df = train_test_split(
            df, 
            test_size=args.val_size + args.test_size,
            random_state=args.random_seed,
            stratify=df[args.label_column]
        )
        
        # Further split temp into val/test
        val_size_adjusted = args.val_size / (args.val_size + args.test_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size_adjusted),
            random_state=args.random_seed,
            stratify=temp_df[args.label_column]
        )
    
    logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Format for multi-task learning
    train_multitask = format_for_multitask(
        train_df, 
        text_column=args.text_column,
        label_column=args.label_column,
        rationale_column=args.rationale_column
    )
    
    val_multitask = format_for_multitask(
        val_df,
        text_column=args.text_column,
        label_column=args.label_column,
        rationale_column=args.rationale_column
    )
    
    test_multitask = format_for_multitask(
        test_df,
        text_column=args.text_column,
        label_column=args.label_column,
        rationale_column=args.rationale_column
    )
    
    # Save to files
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")
    
    # Convert to jsonl format
    train_multitask.to_json(train_path, orient="records", lines=True)
    val_multitask.to_json(val_path, orient="records", lines=True)
    test_multitask.to_json(test_path, orient="records", lines=True)
    
    # Also save original splits for reference
    train_df.to_json(os.path.join(args.output_dir, "train_original.jsonl"), orient="records", lines=True)
    val_df.to_json(os.path.join(args.output_dir, "val_original.jsonl"), orient="records", lines=True)
    test_df.to_json(os.path.join(args.output_dir, "test_original.jsonl"), orient="records", lines=True)
    
    logger.info(f"Successfully saved processed data to {args.output_dir}")
    logger.info(f"  Train: {len(train_multitask)} examples ({train_path})")
    logger.info(f"  Validation: {len(val_multitask)} examples ({val_path})")
    logger.info(f"  Test: {len(test_multitask)} examples ({test_path})")

if __name__ == "__main__":
    main()