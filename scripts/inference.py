# scripts/inference.py
import argparse
import logging
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# scripts/inference.py (continued)
def run_inference(model, tokenizer, texts, batch_size=8):
    """Run inference on texts using the trained model."""
    model.eval()
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Running inference"):
        batch_texts = texts[i:i+batch_size]
        
        # Prepare inputs with task prefix for label task
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
        
        # Process batch results
        for text, prediction in zip(batch_texts, batch_predictions):
            # Determine the label
            if "TRUE" in prediction.upper() or "FACTUAL" in prediction.upper():
                label = "TRUE (Factual)"
            else:
                label = "FALSE (Misinformation)"
            
            results.append({
                "content": text,
                "prediction": prediction,
                "label": label
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--input_file", type=str, help="Path to input file with texts")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save results")
    parser.add_argument("--text_column", type=str, default="content", help="Column name for text data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    args = parser.parse_args()
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load input data
    if args.input_file:
        logger.info(f"Loading data from {args.input_file}")
        if args.input_file.endswith((".json", ".jsonl")):
            df = pd.read_json(args.input_file, lines="jsonl" in args.input_file)
        elif args.input_file.endswith(".csv"):
            df = pd.read_csv(args.input_file)
        else:
            raise ValueError("Unsupported file format. Use .json, .jsonl, or .csv")
        
        texts = df[args.text_column].tolist()
    else:
        # Interactive mode
        logger.info("No input file provided. Enter text directly (type 'exit' to quit):")
        texts = []
        while True:
            text = input("> ")
            if text.lower() == 'exit':
                break
            texts.append(text)
    
    # Run inference
    logger.info(f"Running inference on {len(texts)} texts")
    results = run_inference(model, tokenizer, texts, args.batch_size)
    
    # Save results
    results_df = pd.DataFrame(results)
    
    logger.info(f"Saving results to {args.output_file}")
    if args.output_file.endswith(".json"):
        results_df.to_json(args.output_file, orient="records", indent=2)
    elif args.output_file.endswith(".jsonl"):
        results_df.to_json(args.output_file, orient="records", lines=True)
    elif args.output_file.endswith(".csv"):
        results_df.to_csv(args.output_file, index=False)
    else:
        results_df.to_json(args.output_file, orient="records", indent=2)
    
    logger.info("Inference completed!")

if __name__ == "__main__":
    main()