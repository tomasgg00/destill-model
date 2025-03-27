#!/usr/bin/env python
# scripts/inference_multitask.py
import argparse
import logging
import pandas as pd
import torch
import os
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def run_inference(model, tokenizer, texts, generate_rationales=False, batch_size=8):
    """Run inference on texts using the trained multi-task model."""
    model.eval()
    results = []
    
    # Decide which task to perform
    task_prefix = "[rationale]" if generate_rationales else "[label]"
    task_desc = "rationale generation" if generate_rationales else "classification"
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Running {task_desc}"):
        batch_texts = texts[i:i+batch_size]
        
        # Prepare inputs with task prefix
        prompted_texts = [f"{task_prefix} {text}" for text in batch_texts]
        
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
            if generate_rationales:
                # Generate longer outputs for rationales
                outputs = model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True
                )
            else:
                # Generate shorter outputs for labels
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
            if generate_rationales:
                results.append({
                    "content": text,
                    "rationale": prediction,
                })
            else:
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
    parser = argparse.ArgumentParser(description="Run inference with trained multi-task model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--input_file", type=str, help="Path to input file with texts")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save results")
    parser.add_argument("--text_column", type=str, default="content", help="Column name for text data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--generate_rationales", action="store_true", help="Generate rationales instead of labels")
    parser.add_argument("--generate_both", action="store_true", help="Generate both labels and rationales")
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
    
    # Run inference for labels
    final_results = []
    
    if args.generate_both or not args.generate_rationales:
        logger.info(f"Running label classification on {len(texts)} texts")
        label_results = run_inference(model, tokenizer, texts, generate_rationales=False, batch_size=args.batch_size)
        
        if args.generate_both:
            # Store for later merging
            label_dict = {item["content"]: item for item in label_results}
        else:
            final_results = label_results
    
    # Run inference for rationales if requested
    if args.generate_both or args.generate_rationales:
        logger.info(f"Generating rationales for {len(texts)} texts")
        rationale_results = run_inference(model, tokenizer, texts, generate_rationales=True, batch_size=args.batch_size)
        
        if args.generate_both:
            # Merge results
            for item in rationale_results:
                content = item["content"]
                if content in label_dict:
                    combined_item = label_dict[content].copy()
                    combined_item["rationale"] = item["rationale"]
                    final_results.append(combined_item)
        else:
            final_results = rationale_results
    
    # Save results
    logger.info(f"Saving results to {args.output_file}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    if args.output_file.endswith(".json"):
        with open(args.output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
    elif args.output_file.endswith(".jsonl"):
        with open(args.output_file, 'w') as f:
            for result in final_results:
                f.write(json.dumps(result) + '\n')
    elif args.output_file.endswith(".csv"):
        pd.DataFrame(final_results).to_csv(args.output_file, index=False)
    else:
        with open(args.output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
    
    logger.info("Inference completed!")

if __name__ == "__main__":
    main()