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
    parser.add_argument("--test_size", type=float, default=0.1, help="Size of test split")
    parser.add_argument("--val_size", type=float, default=0.1, help="Size of validation split")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for splitting")
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
    
    # Create train/validation/test splits
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.random_seed)
    
    remaining_val_ratio = args.val_size / (1 - args.test_size)
    train_df, val_df = train_test_split(train_df, test_size=remaining_val_ratio, random_state=args.random_seed)
    
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save splits
    train_df.to_json(f"{args.output_dir}/train.jsonl", orient="records", lines=True)
    val_df.to_json(f"{args.output_dir}/val.jsonl", orient="records", lines=True)
    test_df.to_json(f"{args.output_dir}/test.jsonl", orient="records", lines=True)
    
    print(f"Saved dataset splits to {args.output_dir}")

if __name__ == "__main__":
    main()