#!/usr/bin/env python
# evaluate_model.py
"""
Script to evaluate a trained multi-task model for refugee/migrant misinformation detection,
with special focus on the validation set performance.
"""

import os
import json
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from peft import PeftModel, PeftConfig
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("evaluation.log")]
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
    predictions = [standardize_label(pred) for pred in predictions]
    
    return predictions

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

def standardize_label(label):
    """Standardize label to TRUE/FALSE format."""
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

def calculate_metrics(true_labels, predicted_labels):
    """Calculate metrics for evaluation."""
    # Ensure labels are standardized
    true_labels = [standardize_label(label) for label in true_labels]
    predicted_labels = [standardize_label(label) for label in predicted_labels]
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, 
        predicted_labels,
        labels=["TRUE", "FALSE"],
        average='weighted',
        zero_division=0
    )
    
    # Calculate per-class metrics
    class_metrics = precision_recall_fscore_support(
        true_labels, 
        predicted_labels,
        labels=["TRUE", "FALSE"],
        average=None,
        zero_division=0
    )
    
    # Create confusion matrix
    cm = confusion_matrix(
        true_labels, 
        predicted_labels,
        labels=["TRUE", "FALSE"]
    )
    
    # Count predictions by class
    true_count = predicted_labels.count("TRUE")
    false_count = predicted_labels.count("FALSE")
    total_preds = len(predicted_labels)
    
    # Count actual labels by class
    actual_true = true_labels.count("TRUE")
    actual_false = true_labels.count("FALSE")
    
    # Compile results
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "class_distribution": {
            "predictions": {
                "TRUE": true_count,
                "FALSE": false_count,
                "percent_true": true_count / total_preds * 100 if total_preds > 0 else 0,
                "percent_false": false_count / total_preds * 100 if total_preds > 0 else 0
            },
            "actual": {
                "TRUE": actual_true,
                "FALSE": actual_false,
                "percent_true": actual_true / len(true_labels) * 100,
                "percent_false": actual_false / len(true_labels) * 100
            }
        },
        "per_class": {
            "precision": {
                "TRUE": class_metrics[0][0],
                "FALSE": class_metrics[0][1]
            },
            "recall": {
                "TRUE": class_metrics[1][0],
                "FALSE": class_metrics[1][1]
            },
            "f1": {
                "TRUE": class_metrics[2][0],
                "FALSE": class_metrics[2][1]
            }
        }
    }
    
    return results

def plot_confusion_matrix(cm, labels=["TRUE", "FALSE"], output_path="confusion_matrix.png"):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_model(model, tokenizer, data, output_dir, generate_rationales_flag=False, batch_size=8):
    """Evaluate model on data and save results."""
    # Get texts and true labels
    texts = data["text"].tolist()
    true_labels = data["label"].tolist()
    
    # Predict labels
    predicted_labels = predict_labels(model, tokenizer, texts, batch_size=batch_size)
    
    # Generate rationales if requested
    if generate_rationales_flag:
        rationales = generate_rationales(model, tokenizer, texts, batch_size=batch_size)
    else:
        rationales = [""] * len(texts)
    
    # Calculate metrics
    metrics = calculate_metrics(true_labels, predicted_labels)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        "text": texts,
        "true_label": true_labels,
        "predicted_label": predicted_labels,
        "correct": [t == p for t, p in zip(true_labels, predicted_labels)],
        "rationale": rationales
    })
    
    # Save detailed results
    results_df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Create confusion matrix visualization
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        labels=["TRUE", "FALSE"],
        output_path=os.path.join(output_dir, "confusion_matrix.png")
    )
    
    # Analyze error cases
    errors_df = results_df[results_df["correct"] == False]
    errors_df.to_csv(os.path.join(output_dir, "error_cases.csv"), index=False)
    
    return metrics, results_df

def main():
    parser = argparse.ArgumentParser(description="Evaluate model for misinformation detection")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test data")
    parser.add_argument("--validation_file", type=str, default=None, help="Path to validation data (optional)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--text_column", type=str, default="text", help="Column name for text data")
    parser.add_argument("--label_column", type=str, default="label", help="Column name for label data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--generate_rationales", action="store_true", help="Generate rationales during evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args for reference
    with open(os.path.join(args.output_dir, "eval_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
    
    # Load test data
    test_data = load_data(args.test_file)
    
    # Ensure required columns exist
    if args.text_column not in test_data.columns:
        raise ValueError(f"Text column '{args.text_column}' not found in test data")
    if args.label_column not in test_data.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in test data")
    
    # Rename columns for consistency
    test_data = test_data.rename(columns={
        args.text_column: "text",
        args.label_column: "label"
    })
    
    # Standardize labels
    test_data["label"] = test_data["label"].apply(standardize_label)
    
    # Evaluate on test data
    logger.info("Evaluating model on test data")
    test_metrics, test_results = evaluate_model(
        model, 
        tokenizer, 
        test_data, 
        os.path.join(args.output_dir, "test"),
        generate_rationales_flag=args.generate_rationales,
        batch_size=args.batch_size
    )
    
    # Log test results
    logger.info("Test metrics:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1: {test_metrics['f1']:.4f}")
    
    # Evaluate on validation data if provided
    if args.validation_file:
        logger.info(f"Evaluating model on validation data from {args.validation_file}")
        
        # Load validation data
        val_data = load_data(args.validation_file)
        
        # Rename columns for consistency
        if args.text_column in val_data.columns:
            val_data = val_data.rename(columns={
                args.text_column: "text",
                args.label_column: "label"
            })
        
        # Standardize labels
        val_data["label"] = val_data["label"].apply(standardize_label)
        
        # Evaluate
        val_metrics, val_results = evaluate_model(
            model, 
            tokenizer, 
            val_data, 
            os.path.join(args.output_dir, "validation"),
            generate_rationales_flag=args.generate_rationales,
            batch_size=args.batch_size
        )
        
        # Log validation results
        logger.info("Validation metrics:")
        logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {val_metrics['precision']:.4f}")
        logger.info(f"  Recall: {val_metrics['recall']:.4f}")
        logger.info(f"  F1: {val_metrics['f1']:.4f}")
        
        # Compare test and validation performance
        logger.info("Performance comparison:")
        logger.info(f"  Test accuracy: {test_metrics['accuracy']:.4f} | Validation accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Test F1: {test_metrics['f1']:.4f} | Validation F1: {val_metrics['f1']:.4f}")
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()