# scripts/prepare_dataset.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for multi-task learning")
    parser.add_argument("--input_file", type=str, required=True, help="Path to dataset with rationales")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save splits")
    parser.add_argument("--text_column", type=str, default="text", help="Column name for text data")
    parser.add_argument("--rationale_column", type=str, default="rationale", help="Column name for rationales")
    parser.add_argument("--label_column", type=str, default="label", help="Column name for labels")
    parser.add_argument("--train_val_only", action="store_true", help="Only create train and validation splits (no test)")
    parser.add_argument("--val_size", type=float, default=0.2, help="Size of validation split")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--balance_classes", action="store_true", help="Balance classes in the training set")
    parser.add_argument("--test_size", type=float, default=0.1, help="Size of test split (when not using train_val_only)")
    args = parser.parse_args()
    
    # Load the dataset
    print(f"Loading dataset from {args.input_file}")
    
    # Determine file format based on extension
    file_ext = os.path.splitext(args.input_file)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(args.input_file)
    elif file_ext == '.json':
        df = pd.read_json(args.input_file)
    elif file_ext == '.jsonl':
        # Specifically handle jsonl files (lines=True)
        df = pd.read_json(args.input_file, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .json, .jsonl, or .csv")
    
    print(f"Loaded dataset with {len(df)} examples")
    
    # Rename columns to standard names for consistency
    column_mapping = {}
    if args.text_column != "content" and args.text_column in df.columns:
        column_mapping[args.text_column] = "content"
    elif "text" in df.columns and "content" not in df.columns:
        column_mapping["text"] = "content"
    
    if args.rationale_column != "rationale" and args.rationale_column in df.columns:
        column_mapping[args.rationale_column] = "rationale"
    
    if args.label_column != "label" and args.label_column in df.columns:
        column_mapping[args.label_column] = "label"
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # Verify essential columns exist
    essential_columns = ["content", "rationale", "label"]
    missing_columns = [col for col in essential_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing essential columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use train/val split only (80/20) if specified
    if args.train_val_only:
        # Split directly into train and validation sets
        train_df, val_df = train_test_split(
            df, 
            test_size=args.val_size, 
            random_state=args.random_seed,
            stratify=df["label"] if "label" in df.columns else None  # Stratify by label if available
        )
        
        # Create an empty test set to maintain compatibility
        test_df = pd.DataFrame(columns=df.columns)
        
        print(f"Split sizes (train/val only): Train={len(train_df)}, Val={len(val_df)}")
    else:
        # Use the original three-way split
        train_df, test_df = train_test_split(
            df, 
            test_size=args.test_size, 
            random_state=args.random_seed,
            stratify=df["label"] if "label" in df.columns else None
        )
        
        remaining_val_ratio = args.val_size / (1 - args.test_size)
        train_df, val_df = train_test_split(
            train_df, 
            test_size=remaining_val_ratio, 
            random_state=args.random_seed,
            stratify=train_df["label"] if "label" in train_df.columns else None
        )
        
        print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Balance classes in training set if requested
    if args.balance_classes:
        print("Balancing classes in training set...")
        
        # Check label distribution before balancing
        label_counts = train_df["label"].value_counts()
        print(f"Label distribution before balancing: {label_counts.to_dict()}")
        
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
        
        # Add standardized label column for easier processing
        train_df["std_label"] = train_df["label"].apply(standardize_label)
        
        # Separate by standardized class
        true_samples = train_df[train_df["std_label"] == "TRUE"]
        false_samples = train_df[train_df["std_label"] == "FALSE"]
        
        print(f"Classes found: TRUE={len(true_samples)}, FALSE={len(false_samples)}")
        
        # Find size of classes
        min_class_size = min(len(true_samples), len(false_samples))
        max_class_size = max(len(true_samples), len(false_samples))
        
        # Balance by upsampling or downsampling
        if len(true_samples) < len(false_samples):
            # TRUE is minority class
            if min_class_size < 10:  # Very small dataset, upsample minority
                true_samples = true_samples.sample(max_class_size, replace=True, random_state=args.random_seed)
            else:  # Downsample majority
                false_samples = false_samples.sample(min_class_size, random_state=args.random_seed)
        else:
            # FALSE is minority class
            if min_class_size < 10:  # Very small dataset, upsample minority
                false_samples = false_samples.sample(max_class_size, replace=True, random_state=args.random_seed)
            else:  # Downsample majority
                true_samples = true_samples.sample(min_class_size, random_state=args.random_seed)
        
        # Combine and shuffle
        balanced_train_df = pd.concat([true_samples, false_samples]).sample(frac=1, random_state=args.random_seed)
        balanced_train_df = balanced_train_df.drop(columns=["std_label"])  # Remove the temporary column
        
        # Check label distribution after balancing
        train_label_counts_after = balanced_train_df["label"].apply(standardize_label).value_counts()
        print(f"Label distribution after balancing: TRUE={len(true_samples)}, FALSE={len(false_samples)}")
        
        # Replace the training set with the balanced version
        train_df = balanced_train_df
    
    # Save splits
    train_df.to_json(f"{args.output_dir}/train.jsonl", orient="records", lines=True)
    val_df.to_json(f"{args.output_dir}/val.jsonl", orient="records", lines=True)
    test_df.to_json(f"{args.output_dir}/test.jsonl", orient="records", lines=True)
    
    print(f"Saved dataset splits to {args.output_dir}")

if __name__ == "__main__":
    main()