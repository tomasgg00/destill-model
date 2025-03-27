#!/usr/bin/env python
# generate_rationales.py
"""
Flexible script to generate rationales from fine-tuned models
Supports both sequence-to-sequence and causal language models
Includes detailed progress tracking
"""

import os
import json
import torch
import argparse
import logging
from tqdm import tqdm
import gc
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM, 
    AutoTokenizer
)
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("rationale_generation.log")]
)
logger = logging.getLogger(__name__)

# Default Chain of Thought examples
DEFAULT_COT_EXAMPLES = [
    {
        "text": "Official statistics reveal that 80% of refugees abandon their host country within 5 years of arrival to return home once conditions improve.",
        "reasoning": "Step 1: The text makes a specific statistical claim about refugee behavior.\nStep 2: This statistic contradicts official UNHCR data about refugee displacement.\nStep 3: The claim lacks citation of any specific official source.\nStep 4: The framing implies refugees are temporary visitors.\nStep 5: This appears to be false information misrepresenting refugee patterns.",
        "label": "FALSE"
    },
    {
        "text": "Research shows migrants contribute significantly to host country economies through taxes and entrepreneurship.",
        "reasoning": "Step 1: The text claims migrants have a positive economic impact.\nStep 2: This aligns with studies from organizations like IMF and OECD.\nStep 3: The statement provides a broad economic perspective.\nStep 4: The language is neutral and fact-based.\nStep 5: This appears to be a factual statement supported by economic research.",
        "label": "TRUE"
    }
]

def set_env_for_memory_efficiency():
    """Optimize environment for memory-efficient model loading."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_fine_tuned_model(model_path):
    """
    Flexibly load fine-tuned model with memory optimizations
    
    Args:
        model_path (str): Path to fine-tuned model
    
    Returns:
        Tuple of (model, tokenizer, model_type)
    """
    set_env_for_memory_efficiency()
    
    try:
        logger.info(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info(f"Loading model from {model_path} with memory optimizations...")
        try:
            # First try Seq2Seq model
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                device_map='auto',
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            model_type = "seq2seq"
        except Exception:
            # Fallback to Causal Language Model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map='auto',
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            model_type = "causal"
        
        # Handle potential tokenizer padding issues
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        logger.info(f"Model loaded successfully (Type: {model_type})")
        return model, tokenizer, model_type
    
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise

def create_cot_prompt(examples, input_text, model_type="causal"):
    """
    Create chain-of-thought prompt template
    
    Args:
        examples (list): Chain of thought examples
        input_text (str): Text to generate rationale for
        model_type (str): Type of model (causal or seq2seq)
    
    Returns:
        str: Formatted prompt
    """
    # Prompt template with flexibility for different model types
    if model_type == "causal":
        prompt = """Analyze this text about potential misinformation:

Examples:
"""
        for example in examples:
            prompt += f"""
Text: "{example['text']}"
Reasoning: {example['reasoning']}
Label: {example['label']}
"""
        
        prompt += f"""\nNow analyze this text:
Text: "{input_text}"
Reasoning:"""
    
    else:  # seq2seq
        prompt = f"""Provide a step-by-step reasoning for the following text:
{input_text}

Steps of Analysis:
1. Identify main claims
2. Assess supporting evidence
3. Note potential context or biases
4. Evaluate information accuracy
5. Determine if information is TRUE or FALSE
"""
    
    return prompt

def generate_efficiently(model, tokenizer, prompt, model_type="causal", max_new_tokens=512):
    """
    Generate text with efficiency and control
    
    Args:
        model: Fine-tuned model
        tokenizer: Corresponding tokenizer
        prompt (str): Input prompt
        model_type (str): Type of model (causal or seq2seq)
        max_new_tokens (int): Maximum tokens to generate
    
    Returns:
        str: Generated text
    """
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
        
        # Generate with controlled parameters
        with torch.no_grad():
            if model_type == "causal":
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    do_sample=False,
                    temperature=0.1,
                    repetition_penalty=1.1
                )
            else:  # seq2seq
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_new_tokens,
                    num_return_sequences=1,
                    do_sample=False
                )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return f"Error generating rationale: {str(e)}"

def process_response(response, input_text, model_type="causal"):
    """
    Extract rationale and label from model response
    
    Args:
        response (str): Full model response
        input_text (str): Original input text
        model_type (str): Type of model (causal or seq2seq)
    
    Returns:
        Tuple of (rationale, label)
    """
    try:
        # Remove input text if present in response
        if input_text in response:
            response = response.split(input_text, 1)[1].strip()
        
        # Processing for different model types
        if model_type == "causal":
            # Look for label indicators
            if "Label:" in response:
                parts = response.split("Label:", 1)
                rationale = parts[0].strip()
                label = parts[1].strip()
            elif "TRUE" in response.upper() and "FALSE" not in response.upper().split("TRUE")[-1]:
                idx = response.upper().find("TRUE")
                rationale = response[:idx].strip()
                label = "TRUE"
            elif "FALSE" in response.upper() and "TRUE" not in response.upper().split("FALSE")[-1]:
                idx = response.upper().find("FALSE")
                rationale = response[:idx].strip()
                label = "FALSE"
            else:
                # Use the last few sentences to determine the label
                sentences = response.split('.')
                last_sentences = '.'.join(sentences[-3:])
                
                rationale = response
                if "false" in last_sentences.lower() or "misinformation" in last_sentences.lower():
                    label = "FALSE"
                else:
                    label = "TRUE"
        else:  # seq2seq
            # For seq2seq, assume the entire response is the rationale
            rationale = response
            
            # Try to determine label from content
            if "false" in response.lower() or "misinformation" in response.lower():
                label = "FALSE"
            else:
                label = "TRUE"
        
        # Clean up the label
        label = label.strip().upper()
        if "TRUE" in label:
            label = "TRUE"
        elif "FALSE" in label:
            label = "FALSE"
        
        return rationale, label
    
    except Exception as e:
        logger.error(f"Response processing error: {e}")
        return "Error processing response", "ERROR"

def batch_generate_rationales(model, tokenizer, texts, cot_examples, model_type="causal", batch_size=1, save_path=None):
    """
    Generate rationales in batches with detailed progress tracking
    
    Args:
        model: Fine-tuned model
        tokenizer: Corresponding tokenizer
        texts (list): Input texts
        cot_examples (list): Chain of thought examples
        model_type (str): Type of model (causal or seq2seq)
        batch_size (int): Number of texts to process simultaneously
        save_path (str, optional): Path to save intermediate results
    
    Returns:
        list: Generated rationales with metadata
    """
    logger.info(f"Generating rationales for {len(texts)} texts")
    all_results = []
    
    # Create a progress bar for the entire process
    progress_bar = tqdm(
        total=len(texts), 
        desc="Generating Rationales", 
        unit="text",
        dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_results = []
        
        for text in batch_texts:
            try:
                # Create prompt
                prompt = create_cot_prompt(cot_examples, text, model_type)
                
                # Generate response
                response = generate_efficiently(model, tokenizer, prompt, model_type)
                
                # Process response
                rationale, label = process_response(response, text, model_type)
                
                batch_results.append({
                    "text": text,
                    "rationale": rationale,
                    "label": label,
                    "full_response": response
                })
                
                # Update progress bar
                progress_bar.update(1)
            
            except Exception as e:
                logger.error(f"Error processing text: {text[:50]}... Error: {e}")
                batch_results.append({
                    "text": text,
                    "rationale": "Error generating rationale",
                    "label": "ERROR",
                    "full_response": str(e)
                })
                
                # Still update progress bar for failed items
                progress_bar.update(1)
        
        all_results.extend(batch_results)
        
        # Optional intermediate saving
        if save_path and i % 10 == 0:
            with open(f"{save_path}_checkpoint_{i}.json", "w") as f:
                json.dump(all_results, f, indent=2)
    
    # Close progress bar
    progress_bar.close()
    
    # Save final results
    if save_path:
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Generate step-by-step rationales")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to fine-tuned model")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to input dataset")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="Path to save generated rationales")
    parser.add_argument("--text_column", type=str, default="text", 
                        help="Column name for text data")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for generation")
    parser.add_argument("--examples_file", type=str, default=None, 
                        help="JSON file with custom CoT examples")
    
    args = parser.parse_args()
    
    # Load examples
    cot_examples = DEFAULT_COT_EXAMPLES
    if args.examples_file:
        try:
            with open(args.examples_file, 'r') as f:
                cot_examples = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load custom examples: {e}")
    
    # Load dataset
    try:
        if args.input_file.endswith(".csv"):
            df = pd.read_csv(args.input_file)
            dataset = Dataset.from_pandas(df)
        elif args.input_file.endswith(".json"):
            dataset = load_dataset("json", data_files=args.input_file)["train"]
        else:
            dataset = load_dataset(args.input_file)["train"]
        
        texts = dataset[args.text_column]
    except Exception as e:
        logger.error(f"Dataset loading error: {e}")
        return
    
    # Log dataset information
    print(f"Total texts to process: {len(texts)}")
    print(f"Batch size: {args.batch_size}")
    
    # Load fine-tuned model
    model, tokenizer, model_type = load_fine_tuned_model(args.model_path)
    
    # Generate rationales
    results = batch_generate_rationales(
        model, 
        tokenizer, 
        texts, 
        cot_examples,
        model_type=model_type,
        batch_size=args.batch_size,
        save_path=args.output_file
    )
    
    # Log summary
    labels = [result["label"] for result in results]
    print("\nRationale Generation Summary:")
    print(f"Total texts processed: {len(results)}")
    print(f"TRUE labels: {labels.count('TRUE')} ({labels.count('TRUE')/len(results)*100:.2f}%)")
    print(f"FALSE labels: {labels.count('FALSE')} ({labels.count('FALSE')/len(results)*100:.2f}%)")
    print(f"ERROR labels: {labels.count('ERROR')} ({labels.count('ERROR')/len(results)*100:.2f}%)")

if __name__ == "__main__":
    main()