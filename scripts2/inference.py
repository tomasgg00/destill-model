#!/usr/bin/env python
# inference.py
"""
Script to run inference on unlabeled data using a trained model.
This is designed to be used with the UNHCR dataset.
"""

import os
import json
import argparse
import logging
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("inference.log")]
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path, device=None):
    """Load model and tokenizer from path, handling LoRA models if needed."""
    logger.info(f"Loading model from {model_path}")
    
    # Check if this is a LoRA model
    is_lora = False
    try:
        peft_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(peft_config_path):
            is_lora = True
            logger.info("Detected LoRA model")
    except:
        pass
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if is_lora:
        # Load LoRA model
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_path = peft_config.base_model_name_or_path
        
        # Load base model
        logger.info(f"Loading base model from {base_model_path}")
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, model_path)
    else:
        # Load regular model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    model.eval()
    
    return model, tokenizer

def load_data(file_path):
    """Load data from file."""
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
        # Try loading as a HuggingFace dataset
        try:
            dataset = load_dataset(file_path)["train"]  # Assume "train" split
            return pd.DataFrame(dataset)
        except:
            raise ValueError(f"Unsupported file format: {file_path}")

def predict_labels(model, tokenizer, texts, batch_size=8, max_length=50, device=None):
    """Predict labels for texts using the model."""
    if device is None:
        device = model.device
    
    predictions = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting labels"):
        batch_texts = texts[i:i+batch_size]
        batch_inputs = [f"[label] {text}" for text in batch_texts]
        
        # Tokenize
        inputs = tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_preds)
    
    # Standardize predictions to TRUE/FALSE
    standardized_predictions = []
    for pred in predictions:
        if "TRUE" in pred.upper() or "FACTUAL" in pred.upper() or pred.upper() in ["T", "1", "YES"]:
            standardized_predictions.append("TRUE")
        else:
            standardized_predictions.append("FALSE")
    
    return predictions, standardized_predictions

def generate_rationales(model, tokenizer, texts, batch_size=8, max_length=256, device=None):
    """Generate rationales for texts using the model."""
    if device is None:
        device = model.device
    
    rationales = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating rationales"):
        batch_texts = texts[i:i+batch_size]
        batch_inputs = [f"[rationale] {text}" for text in batch_texts]
        
        # Tokenize
        inputs = tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode
        batch_rats = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        rationales.extend(batch_rats)
    
    return rationales

def main():
    parser = argparse.ArgumentParser(description="Run inference on unlabeled data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to unlabeled data")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--text_column", type=str, default="text", help="Column name for text data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--generate_rationales", action="store_true", help="Generate rationales")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu)")
    parser.add_argument("--id_column", type=str, default=None, help="Column to use as ID")
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
    
    # Load data
    data = load_data(args.input_file)
    
    # Ensure text column exists
    if args.text_column not in data.columns:
        raise ValueError(f"Text column '{args.text_column}' not found in data")
    
    # Get texts
    texts = data[args.text_column].tolist()
    
    # Generate predictions
    logger.info("Generating predictions")
    raw_predictions, predictions = predict_labels(
        model, tokenizer, texts, batch_size=args.batch_size
    )
    
    # Generate rationales if requested
    if args.generate_rationales:
        logger.info("Generating rationales")
        rationales = generate_rationales(
            model, tokenizer, texts, batch_size=args.batch_size
        )
    else:
        rationales = [""] * len(texts)
    
    # Create results
    results = []
    for i, (text, pred, raw_pred, rationale) in enumerate(zip(texts, predictions, raw_predictions, rationales)):
        result = {
            "text": text,
            "prediction": pred,
            "raw_prediction": raw_pred,
        }
        
        # Add rationale if generated
        if args.generate_rationales:
            result["rationale"] = rationale
        
        # Add ID if specified
        if args.id_column and args.id_column in data.columns:
            result["id"] = data[args.id_column].iloc[i]
        else:
            result["id"] = i
        
        # Add all original columns
        for col in data.columns:
            if col != args.text_column and col != "id":
                result[col] = data[col].iloc[i]
        
        results.append(result)
    
    # Save results
    logger.info(f"Saving predictions to {args.output_file}")
    
    # Determine output format
    if args.output_file.endswith(".csv"):
        pd.DataFrame(results).to_csv(args.output_file, index=False)
    elif args.output_file.endswith(".json"):
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=4)
    elif args.output_file.endswith(".jsonl"):
        with open(args.output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
    else:
        # Default to jsonl
        with open(args.output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
    
    # Count predictions
    true_count = predictions.count("TRUE")
    false_count = predictions.count("FALSE")
    total = len(predictions)
    
    logger.info("Inference statistics:")
    logger.info(f"  Total predictions: {total}")
    logger.info(f"  TRUE predictions: {true_count} ({true_count/total*100:.1f}%)")
    logger.info(f"  FALSE predictions: {false_count} ({false_count/total*100:.1f}%)")
    logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()