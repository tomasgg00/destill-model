# scripts/generate_rationales.py
import os
import json
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def create_cot_prompt(examples):
    """Create chain-of-thought prompt template with examples."""
    prompt = """[INST] Analyze the following text about refugees or migrants and determine if it contains misinformation.
    For your analysis, provide step-by-step reasoning by:
    1. Identifying the main claims being made
    2. Assessing evidence that supports or contradicts these claims
    3. Noting relevant context that might be missing
    4. Identifying potential biases or misleading framing
    
    Then conclude with either TRUE (Factual) or FALSE.
    
    """
    
    # Add demonstration examples with rationales
    for i, example in enumerate(examples):
        prompt += f"""Example {i+1}:
        Text: "{example['text']}"
        Reasoning: {example['reasoning']}
        Label: {example['label']}
        
        """
    
    prompt += """Now analyze this text:
    Text: {input_text}
    Reasoning: [/INST]"""
    
    return prompt

def setup_teacher_model(model_path):
    """Set up the fine-tuned teacher model with robust error handling."""
    import torch  # Add this import here to ensure torch is available
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading teacher model from {model_path}")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Check if it's a LoRA model
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if is_lora:
        try:
            # First try the normal approach
            from peft import PeftModel, PeftConfig
            
            # Get base model name from config
            config = PeftConfig.from_pretrained(model_path)
            base_model_name = config.base_model_name_or_path
            
            print(f"Loading base model: {base_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            try:
                # Try to load with safetensors
                model = PeftModel.from_pretrained(base_model, model_path)
            except Exception as e:
                print(f"Error loading adapter with safetensors: {e}")
                print("Trying alternative loading method...")
                
                # Alternative: Use base model without adapter
                model = base_model
                print("Using base model without adapter due to loading issues")
                
        except Exception as outer_e:
            print(f"Error in LoRA setup: {outer_e}")
            print("Falling back to base model")
            
            # Get base model name from config file directly
            with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
                import json
                config_data = json.load(f)
                base_model_name = config_data.get("base_model_name_or_path")
            
            # Just use the base model as fallback
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
    else:
        # For full fine-tuned models
        print("Loading full fine-tuned model")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer

def process_response(response, input_text):
    """Extract rationale and label from model response."""
    # Skip the input text part if it's repeated in the output
    if input_text in response:
        response = response.split(input_text, 1)[1].strip()
    
    # Look for conclusion or label indicators
    if "FALSE" in response.upper() and "TRUE" not in response.upper().split("FALSE")[-1]:
        label = "FALSE"
    elif "TRUE" in response.upper() and "FALSE" not in response.upper().split("TRUE")[-1]:
        label = "TRUE"
    elif "MISINFORMATION" in response.upper():
        label = "FALSE"
    elif "FACTUAL" in response.upper():
        label = "TRUE"
    else:
        # Default label based on final few sentences
        last_sentences = " ".join(response.split(".")[-3:])
        if "misleading" in last_sentences.lower() or "false" in last_sentences.lower() or "incorrect" in last_sentences.lower():
            label = "FALSE"
        else:
            label = "TRUE"
    
    # Extract rationale - everything before the label
    if "FALSE" in response:
        rationale = response.split("FALSE")[0].strip()
    elif "TRUE" in response:
        rationale = response.split("TRUE")[0].strip()
    elif "conclusion:" in response.lower():
        rationale = response.lower().split("conclusion:")[0].strip()
    else:
        # Take everything except the last sentence as rationale
        sentences = response.split(".")
        if len(sentences) > 1:
            rationale = ".".join(sentences[:-1]).strip()
        else:
            rationale = response
    
    return rationale, label

def generate_rationales(model, tokenizer, texts, prompt_template, batch_size=1, max_length=512):
    """Generate rationales for each text using the teacher model."""
    from tqdm import tqdm
    all_results = []
    
    # Calculate total steps for progress display
    total_steps = len(texts)
    
    # Create progress bar
    progress_bar = tqdm(
        total=total_steps,
        desc="Generating rationales",
        bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',  # Customize format for more visibility
        ncols=100
    )
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        batch_results = []
        for text in batch_texts:
            # Format the prompt
            prompt = prompt_template.format(input_text=text)
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_length,
                    temperature=0.3,  # Lower temperature for more consistent outputs
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=1
                )
            
            # Decode the response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Process to extract rationale and label
            rationale, label = process_response(full_response, text)
            
            batch_results.append({
                "text": text,
                "rationale": rationale,
                "label": label,
                "full_response": full_response
            })
            
            # Update progress bar for each example
            progress_bar.update(1)
            
            # Also print percentage completion
            percent_complete = (i + len(batch_results)) / total_steps * 100
            progress_bar.set_description(f"Generating rationales [{percent_complete:.1f}%]")
        
        all_results.extend(batch_results)
    
    progress_bar.close()
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Generate rationales using a teacher model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to teacher model")
    parser.add_argument("--input_data", type=str, required=True, help="Path to input dataset (JSON or CSV)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output file")
    parser.add_argument("--text_column", type=str, default="content", help="Column name for text data")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of examples to sample (optional)")
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.input_data}")
    
    # Determine file format based on extension
    if args.input_data.endswith(".json"):
        df = pd.read_json(args.input_data, lines="jsonl" in args.input_data)
    elif args.input_data.endswith(".jsonl"):
        df = pd.read_json(args.input_data, lines=True)
    elif args.input_data.endswith(".csv"):
        df = pd.read_csv(args.input_data)
    else:
        raise ValueError("Unsupported file format. Use .json, .jsonl, or .csv")
    
    # Sample dataset if specified
    if args.sample_size and args.sample_size < len(df):
        print(f"Sampling {args.sample_size} examples from dataset")
        df = df.sample(args.sample_size, random_state=42)
    
    # Get texts
    texts = df[args.text_column].tolist()
    print(f"Loaded {len(texts)} texts for processing")
    
    # Load the teacher model
    model, tokenizer = setup_teacher_model(args.model_path)
    
    # Define examples with reasoning that cover misinformation, disinformation, and malinformation
    examples = [
        # Misinformation example (false information shared without intent to deceive)
        {
            "text": "A recent study showed that 80% of refugees abandon their host country within 5 years of arrival to return home once conditions improve.",
            "reasoning": "This claim presents a specific statistic (80% return rate) that is not supported by data from major refugee agencies like UNHCR. Return rates vary significantly by conflict and situation, but global averages show that most refugees remain displaced for many years, often decades. Long-term displacement is the norm rather than the exception. The statistic appears to be fabricated or severely misrepresented. This seems to be unintentional misinformation - a false claim that may have originated from misunderstanding refugee return statistics or conflating temporary displacement with refugee status. The information is incorrect but not presented with apparent malicious framing that would suggest deliberate deception.",
            "label": "FALSE"
        },
        
        # Disinformation example (deliberately false information intended to mislead)
        {
            "text": "LEAKED DOCUMENTS: UN's secret plan revealed to replace European populations with 10 million migrants annually. Government officials are deliberately hiding these settlement quotas from citizens.",
            "reasoning": "This claim contains hallmarks of a deliberate disinformation campaign. First, it references non-existent 'leaked documents' to create false credibility. Second, it invokes conspiracy theories about 'population replacement' that have been repeatedly debunked. Third, it fabricates specific numbers (10 million annually) that dramatically exceed actual migration figures. Fourth, it falsely claims governments are concealing information, when in reality migration policies are publicly debated and documented. This appears to be deliberately fabricated disinformation designed to provoke fear and distrust in institutions. The sensationalist language, false attribution to officials, and complete lack of verifiable sources indicate calculated deception rather than mere misunderstanding.",
            "label": "FALSE"
        },
        
        # Malinformation example (genuine information used out of context to mislead)
        {
            "text": "Breaking: Refugee arrested for assault in Berlin yesterday. This is what happens when we open our borders to these people. How many more innocent citizens must suffer?",
            "reasoning": "This text contains a potentially factual core statement (a single refugee being arrested) but deliberately frames it to mislead through harmful generalization. The isolated incident is presented to imply a pattern of criminal behavior among all refugees through the phrase 'this is what happens' and 'these people.' This is a classic example of malinformation - where a possibly real event is taken out of context and used to stigmatize an entire group. The text selectively highlights a single negative incident while ignoring statistics showing no increased crime rates among refugee populations. The inflammatory questions at the end ('How many more innocent citizens must suffer?') reveal the intent to provoke fear rather than inform, using an isolated case to create a false narrative about refugees as a whole.",
            "label": "FALSE"
        },
        
        # Factual example for contrast
        {
            "text": "UNHCR reports that refugee children often face disruptions to their education, with only 68% enrolled in primary school compared to 90% of children globally, based on their 2023 education report.",
            "reasoning": "This statement makes specific claims about refugee children's education access, citing statistics and a source (UNHCR 2023 education report). Checking UNHCR's official publications confirms these figures are accurate representations of their educational access data. The text presents the information neutrally without attempting to manipulate or mislead. It provides relevant context about the educational challenges faced by refugee populations compared to global averages. There is no evidence of false information, manipulation of facts, or harmful framing in this statement.",
            "label": "TRUE"
        }
    ]
    
    # Create prompt template
    prompt_template = create_cot_prompt(examples)
    
    # Generate rationales
    print("Generating rationales...")
    results = generate_rationales(model, tokenizer, texts, prompt_template, batch_size=args.batch_size)
    
    # Create output dataframe
    output_df = pd.DataFrame(results)
    
    # Add original columns from input dataframe
    for col in df.columns:
        if col != args.text_column and col not in output_df.columns:
            if len(df) == len(output_df):  # Only add if lengths match
                output_df[col] = df[col].values
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # Save results
    print(f"Saving results to {args.output_file}")
    if args.output_file.endswith(".json"):
        output_df.to_json(args.output_file, orient="records", indent=2)
    elif args.output_file.endswith(".jsonl"):
        output_df.to_json(args.output_file, orient="records", lines=True)
    elif args.output_file.endswith(".csv"):
        output_df.to_csv(args.output_file, index=False)
    else:
        output_df.to_json(args.output_file, orient="records", lines=True)
    
    print("Done!")

if __name__ == "__main__":
    main()