# scripts/evaluate_model.py
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
    """Evaluate the model on test data."""
    model.eval()
    
    # Prepare data
    texts = test_df["content"].tolist()
    true_labels = test_df["label"].tolist()
    
    # Process label format
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
    confidences = []
    
    # Process in batches
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
    
    # Process predictions
    processed_predictions = []
    for pred in predictions:
        if "TRUE" in pred.upper() or "FACTUAL" in pred.upper():
            processed_predictions.append("TRUE (Factual)")
        else:
            processed_predictions.append("FALSE (Misinformation)")
    
    # Calculate metrics
    accuracy = accuracy_score(processed_true_labels, processed_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        processed_true_labels, 
        processed_predictions, 
        average='binary',
        pos_label="TRUE (Factual)"
    )
    
    # Create confusion matrix
    cm = confusion_matrix(processed_true_labels, processed_predictions, labels=["TRUE (Factual)", "FALSE (Misinformation)"])
    
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
        "confusion_matrix": cm.tolist()
    }
    
    # Create detailed results DataFrame
    results_df = pd.DataFrame({
        "text": texts,
        "true_label": processed_true_labels,
        "predicted_label": processed_predictions,
        "correct": [t == p for t, p in zip(processed_true_labels, processed_predictions)]
    })
    
    # Save to CSV
    results_df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    
    # Log results
    logger.info(f"Evaluation Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
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
    results = evaluate_model(model, tokenizer, test_df, args.output_dir, args.batch_size)
    
    # Save results to file
    results_file = os.path.join(args.output_dir, "metrics.txt")
    with open(results_file, "w") as f:
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n")
    
    logger.info(f"Evaluation results saved to {args.output_dir}")

if __name__ == "__main__":
    main()