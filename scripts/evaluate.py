#!/usr/bin/env python
# scripts/evaluate.py 
import os
import argparse
import logging
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def evaluate_model(model, tokenizer, test_df, output_dir, batch_size=8):
    """Evaluate the model on test data with detailed analysis."""
    model.eval()
    
    # Prepare data
    texts = test_df["content"].tolist()
    true_labels = test_df["label"].tolist()
    
    # Process label format for true labels
    processed_true_labels = []
    for label in true_labels:
        if isinstance(label, str):
            if "TRUE" in label.upper() or "FACTUAL" in label.upper():
                processed_true_labels.append("TRUE (Factual)")
            else:
                processed_true_labels.append("FALSE (Misinformation)")
        else:
            # If it's a boolean or numeric value
            processed_true_labels.append("TRUE (Factual)" if label else "FALSE (Misinformation)")
    
    predictions = []
    rationales = []
    raw_outputs = []  # Store raw model outputs for debugging
    
    # Process in batches for label prediction
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Prepare inputs with task prefix for label prediction
        prompted_texts = [f"[label] {text}" for text in batch_texts]
        
        # Tokenize
        inputs = tokenizer(
            prompted_texts, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=20,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode predictions
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_predictions)
        raw_outputs.extend(batch_predictions)  # Save raw outputs
    
    # Process predictions to standard format
    processed_predictions = []
    for pred in predictions:
        if "TRUE" in pred.upper() or "FACTUAL" in pred.upper():
            processed_predictions.append("TRUE (Factual)")
        else:
            processed_predictions.append("FALSE (Misinformation)")
    
    # Generate rationales for deeper understanding
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Prepare inputs with task prefix for rationale task
        prompted_texts = [f"[rationale] {text}" for text in batch_texts]
        
        # Tokenize
        inputs = tokenizer(
            prompted_texts, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate rationales
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode rationales
        batch_rationales = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        rationales.extend(batch_rationales)
    
    # Calculate metrics
    accuracy = accuracy_score(processed_true_labels, processed_predictions)
    
    # Add a small epsilon to avoid division by zero
    precision, recall, f1, _ = precision_recall_fscore_support(
        processed_true_labels, 
        processed_predictions, 
        average='binary',
        pos_label="TRUE (Factual)",
        zero_division=0
    )
    
    # Calculate class-wise metrics for debugging
    class_metrics = precision_recall_fscore_support(
        processed_true_labels, 
        processed_predictions, 
        average=None,
        labels=["TRUE (Factual)", "FALSE (Misinformation)"],
        zero_division=0
    )
    
    # Create confusion matrix
    cm = confusion_matrix(
        processed_true_labels, 
        processed_predictions, 
        labels=["TRUE (Factual)", "FALSE (Misinformation)"]
    )
    
    # Count predictions by class
    true_count = processed_predictions.count("TRUE (Factual)")
    false_count = processed_predictions.count("FALSE (Misinformation)")
    total_preds = len(processed_predictions)
    
    # Count actual labels by class
    actual_true = processed_true_labels.count("TRUE (Factual)")
    actual_false = processed_true_labels.count("FALSE (Misinformation)")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=["TRUE (Factual)", "FALSE (Misinformation)"],
        yticklabels=["TRUE (Factual)", "FALSE (Misinformation)"]
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    
    # Save results
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "class_distribution": {
            "predictions": {
                "TRUE (Factual)": true_count,
                "FALSE (Misinformation)": false_count,
                "percent_true": true_count / total_preds * 100 if total_preds > 0 else 0,
                "percent_false": false_count / total_preds * 100 if total_preds > 0 else 0
            },
            "actual": {
                "TRUE (Factual)": actual_true,
                "FALSE (Misinformation)": actual_false,
                "percent_true": actual_true / len(processed_true_labels) * 100,
                "percent_false": actual_false / len(processed_true_labels) * 100
            }
        }
    }
    
    # Create detailed results DataFrame
    results_df = pd.DataFrame({
        "text": texts,
        "true_label": processed_true_labels,
        "predicted_label": processed_predictions,
        "raw_prediction": raw_outputs,
        "rationale": rationales,
        "correct": [t == p for t, p in zip(processed_true_labels, processed_predictions)]
    })
    
    # Save to CSV
    results_df.to_csv(os.path.join(output_dir, "detailed_evaluation_results.csv"), index=False)
    
    # Log results
    logger.info(f"Evaluation Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Prediction distribution:")
    logger.info(f"    TRUE: {true_count}/{total_preds} ({results['class_distribution']['predictions']['percent_true']:.1f}%)")
    logger.info(f"    FALSE: {false_count}/{total_preds} ({results['class_distribution']['predictions']['percent_false']:.1f}%)")
    logger.info(f"  Actual distribution:")
    logger.info(f"    TRUE: {actual_true}/{len(processed_true_labels)} ({results['class_distribution']['actual']['percent_true']:.1f}%)")
    logger.info(f"    FALSE: {actual_false}/{len(processed_true_labels)} ({results['class_distribution']['actual']['percent_false']:.1f}%)")
    logger.info(f"  Confusion Matrix:")
    logger.info(f"    {cm[0][0]} | {cm[0][1]}")
    logger.info(f"    {cm[1][0]} | {cm[1][1]}")
    
    # Save raw outputs for further analysis
    with open(os.path.join(output_dir, "raw_outputs.txt"), "w") as f:
        for i, (text, true, pred, raw) in enumerate(zip(texts, processed_true_labels, processed_predictions, raw_outputs)):
            f.write(f"Example {i+1}:\n")
            f.write(f"Text: {text[:100]}...\n")
            f.write(f"True label: {true}\n")
            f.write(f"Predicted label: {pred}\n")
            f.write(f"Raw output: {raw}\n\n")
    
    return results, results_df

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained multi-task model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test data")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    logger.info(f"Loading test data from {args.test_file}")
    test_df = pd.read_json(args.test_file, lines="jsonl" in args.test_file) if args.test_file.endswith((".json", ".jsonl")) else pd.read_csv(args.test_file)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Evaluate model
    logger.info("Evaluating model...")
    results, results_df = evaluate_model(model, tokenizer, test_df, args.output_dir, args.batch_size)
    
    # Save results to file
    results_file = os.path.join(args.output_dir, "metrics.txt")
    with open(results_file, "w") as f:
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n\n")
        f.write(f"Prediction distribution:\n")
        f.write(f"  TRUE: {results['class_distribution']['predictions']['TRUE (Factual)']} ")
        f.write(f"({results['class_distribution']['predictions']['percent_true']:.1f}%)\n")
        f.write(f"  FALSE: {results['class_distribution']['predictions']['FALSE (Misinformation)']} ")
        f.write(f"({results['class_distribution']['predictions']['percent_false']:.1f}%)\n\n")
        f.write(f"Actual distribution:\n")
        f.write(f"  TRUE: {results['class_distribution']['actual']['TRUE (Factual)']} ")
        f.write(f"({results['class_distribution']['actual']['percent_true']:.1f}%)\n")
        f.write(f"  FALSE: {results['class_distribution']['actual']['FALSE (Misinformation)']} ")
        f.write(f"({results['class_distribution']['actual']['percent_false']:.1f}%)\n")
    
    # Display most common error cases
    errors_df = results_df[results_df["correct"] == False]
    if len(errors_df) > 0:
        error_file = os.path.join(args.output_dir, "error_analysis.txt")
        with open(error_file, "w") as f:
            f.write(f"Error Analysis (showing up to 10 examples):\n\n")
            for i, (idx, row) in enumerate(errors_df.head(10).iterrows()):
                f.write(f"Error Example {i+1}:\n")
                f.write(f"Text: {row['text'][:200]}...\n")
                f.write(f"True label: {row['true_label']}\n")
                f.write(f"Predicted label: {row['predicted_label']}\n")
                f.write(f"Raw prediction: {row['raw_prediction']}\n")
                f.write(f"Rationale: {row['rationale'][:300]}...\n\n")
    
    logger.info(f"Detailed evaluation results saved to {args.output_dir}")

if __name__ == "__main__":
    main()