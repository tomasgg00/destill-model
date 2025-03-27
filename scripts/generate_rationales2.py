#!/usr/bin/env python
# scripts/generate_rationales.py
import os
import json
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Flexible imports
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer
)

# Add conditional import for Deepseek if available
try:
    from transformers import DeepseekForCausalLM, DeepseekTokenizer
except ImportError:
    # Fallback to using AutoClasses if specific Deepseek classes aren't available
    DeepseekForCausalLM = AutoModelForCausalLM
    DeepseekTokenizer = AutoTokenizer

# Similar approach for Gemma
try:
    from transformers import GemmaForCausalLM, GemmaTokenizer
except ImportError:
    GemmaForCausalLM = AutoModelForCausalLM
    GemmaTokenizer = AutoTokenizer

# PEFT support
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class RationaleGenerator:
    def __init__(self, 
                 model_path: str, 
                 model_type: str = 'auto', 
                 use_peft: bool = True):
        """
        Initialize model and tokenizer with flexible configuration
        
        Args:
            model_path (str): Path to model or HuggingFace model name
            model_type (str): Specific model type ('llama', 'deepseek', 'gemma', 'auto')
            use_peft (bool): Whether to use PEFT adapter if available
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.use_peft = use_peft and PEFT_AVAILABLE
        
        # Model and tokenizer mapping with more flexible fallbacks
        self.model_classes = {
            'llama': (LlamaForCausalLM, LlamaTokenizer),
            'deepseek': (DeepseekForCausalLM, DeepseekTokenizer),
            'gemma': (GemmaForCausalLM, GemmaTokenizer),
            'auto': (AutoModelForCausalLM, AutoTokenizer)
        }
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
    
    def _load_model(self):
        """
        Flexible model loading with PEFT support
        
        Returns:
            Tuple of (model, tokenizer)
        """
        # Determine model and tokenizer classes
        ModelClass, TokenizerClass = self.model_classes.get(
            self.model_type, 
            (AutoModelForCausalLM, AutoTokenizer)
        )
        
        print(f"Loading model from {self.model_path}")
        print(f"Using Model Class: {ModelClass.__name__}")
        print(f"Using Tokenizer Class: {TokenizerClass.__name__}")
        
        try:
            # Load tokenizer
            tokenizer = TokenizerClass.from_pretrained(self.model_path)
            
            # Detect base model from adapter config if using PEFT
            base_model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Default fallback
            adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                try:
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                        base_model_name = adapter_config.get('base_model_name_or_path', base_model_name)
                except Exception as config_error:
                    print(f"Error reading adapter config: {config_error}")
            
            # Load base model
            base_model = ModelClass.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # PEFT adapter loading
            if self.use_peft:
                try:
                    model = PeftModel.from_pretrained(
                        base_model, 
                        self.model_path,
                        is_trainable=False
                    )
                    print("PEFT model loaded successfully.")
                except Exception as peft_error:
                    print(f"PEFT loading failed: {peft_error}")
                    model = base_model
            else:
                model = base_model
            
            # Set to evaluation mode
            model.eval()
            
            return model, tokenizer
        
        except Exception as e:
            print(f"Model loading error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_cot_prompt(self, examples: List[Dict[str, str]]) -> str:
        """
        Create chain-of-thought prompt template with examples
        
        Args:
            examples (List[Dict]): List of example dictionaries with text, reasoning, label
        
        Returns:
            str: Formatted prompt template
        """
        prompt = """[INST] Analyze the following text for misinformation.
Provide a step-by-step reasoning process:
1. Identify the main claims
2. Assess supporting or contradicting evidence
3. Note missing context
4. Detect potential biases

Example Analyses:
"""
        
        for i, example in enumerate(examples, 1):
            prompt += f"""
Example {i}:
Text: "{example['text']}"
Reasoning: {example['reasoning']}
Conclusion: {example['label']}
"""
        
        prompt += """\nNow analyze this text:
Text: {input_text}
Reasoning: [/INST]"""
        
        return prompt
    
    def generate_rationale(
        self, 
        text: str, 
        examples: List[Dict[str, str]], 
        max_length: int = 512
    ) -> Dict[str, str]:
        """
        Generate rationale for a single text
        
        Args:
            text (str): Input text to analyze
            examples (List[Dict]): Demonstration examples
            max_length (int): Maximum generation length
        
        Returns:
            Dict with text, rationale, and label
        """
        # Create prompt
        prompt_template = self.create_cot_prompt(examples)
        full_prompt = prompt_template.format(input_text=text)
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length
        ).to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_length,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=1
                )
            except Exception as e:
                print(f"Generation error for text: {text}")
                print(f"Error: {e}")
                return {
                    "text": text,
                    "rationale": "Unable to generate rationale",
                    "label": "UNKNOWN"
                }
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract rationale and label
        return self._process_response(full_response, text)
    
    def _process_response(self, response: str, input_text: str) -> Dict[str, str]:
        """
        Process model response to extract rationale and label
        
        Args:
            response (str): Full model response
            input_text (str): Original input text
        
        Returns:
            Dict with processed rationale and label
        """
        # Remove input text if present
        if input_text in response:
            response = response.split(input_text, 1)[-1].strip()
        
        # Label detection
        response_upper = response.upper()
        if "FALSE" in response_upper and "TRUE" not in response_upper.split("FALSE")[-1]:
            label = "FALSE"
        elif "TRUE" in response_upper and "FALSE" not in response_upper.split("TRUE")[-1]:
            label = "TRUE"
        elif "MISINFORMATION" in response_upper:
            label = "FALSE"
        elif "FACTUAL" in response_upper:
            label = "TRUE"
        else:
            # Default label determination
            last_sentences = " ".join(response.split(".")[-3:])
            if any(word in last_sentences.lower() for word in ["misleading", "false", "incorrect"]):
                label = "FALSE"
            else:
                label = "TRUE"
        
        # Rationale extraction
        try:
            rationale_parts = response.split(label)[0].strip()
            if not rationale_parts:
                rationale_parts = response
        except Exception:
            rationale_parts = response
        
        return {
            "text": input_text,
            "rationale": rationale_parts,
            "label": label
        }

def load_dataset(input_data: str, text_column: str = "content", sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load dataset from various file formats
    
    Args:
        input_data (str): Path to input file
        text_column (str): Column containing text to analyze
        sample_size (int, optional): Number of samples to take
    
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    # File format detection
    if input_data.endswith(".json"):
        df = pd.read_json(input_data, lines="jsonl" in input_data)
    elif input_data.endswith(".jsonl"):
        df = pd.read_json(input_data, lines=True)
    elif input_data.endswith(".csv"):
        df = pd.read_csv(input_data)
    else:
        raise ValueError("Unsupported file format. Use .json, .jsonl, or .csv")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate rationales using flexible model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--model_type", type=str, default="auto", 
                        choices=['auto', 'llama', 'deepseek', 'gemma'], 
                        help="Specific model type")
    parser.add_argument("--input_data", type=str, required=True, help="Path to input dataset")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output file")
    parser.add_argument("--text_column", type=str, default="content", help="Column name for text data")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of examples to sample")
    parser.add_argument("--use_peft", action="store_true", help="Use PEFT adapter if available")
    
    args = parser.parse_args()
    
    # Demonstration examples for chain-of-thought reasoning
    examples = [
        {
            "text": "A recent study showed that 80% of refugees abandon their host country within 5 years of arrival to return home once conditions improve.",
            "reasoning": "This claim presents a specific statistic (80% return rate) that is not supported by data from major refugee agencies like UNHCR. Return rates vary significantly by conflict and situation, but global averages show that most refugees remain displaced for many years, often decades. Long-term displacement is the norm rather than the exception. The statistic appears to be fabricated or severely misrepresented. This seems to be unintentional misinformation - a false claim that may have originated from misunderstanding refugee return statistics or conflating temporary displacement with refugee status. The information is incorrect but not presented with apparent malicious framing that would suggest deliberate deception.",
            "label": "FALSE"
        },
        {
            "text": "LEAKED DOCUMENTS: UN's secret plan revealed to replace European populations with 10 million migrants annually. Government officials are deliberately hiding these settlement quotas from citizens.",
            "reasoning": "This claim contains hallmarks of a deliberate disinformation campaign. First, it references non-existent 'leaked documents' to create false credibility. Second, it invokes conspiracy theories about 'population replacement' that have been repeatedly debunked. Third, it fabricates specific numbers (10 million annually) that dramatically exceed actual migration figures. Fourth, it falsely claims governments are concealing information, when in reality migration policies are publicly debated and documented. This appears to be deliberately fabricated disinformation designed to provoke fear and distrust in institutions. The sensationalist language, false attribution to officials, and complete lack of verifiable sources indicate calculated deception rather than mere misunderstanding.",
            "label": "FALSE"
        },
        {
            "text": "Breaking: Refugee arrested for assault in Berlin yesterday. This is what happens when we open our borders to these people. How many more innocent citizens must suffer?",
            "reasoning": "This text contains a potentially factual core statement (a single refugee being arrested) but deliberately frames it to mislead through harmful generalization. The isolated incident is presented to imply a pattern of criminal behavior among all refugees through the phrase 'this is what happens' and 'these people.' This is a classic example of malinformation - where a possibly real event is taken out of context and used to stigmatize an entire group. The text selectively highlights a single negative incident while ignoring statistics showing no increased crime rates among refugee populations. The inflammatory questions at the end ('How many more innocent citizens must suffer?') reveal the intent to provoke fear rather than inform, using an isolated case to create a false narrative about refugees as a whole.",
            "label": "FALSE"
        },
        {
            "text": "UNHCR reports that refugee children often face disruptions to their education, with only 68% enrolled in primary school compared to 90% of children globally, based on their 2023 education report.",
            "reasoning": "This statement makes specific claims about refugee children's education access, citing statistics and a source (UNHCR 2023 education report). Checking UNHCR's official publications confirms these figures are accurate representations of their educational access data. The text presents the information neutrally without attempting to manipulate or mislead. It provides relevant context about the educational challenges faced by refugee populations compared to global averages. There is no evidence of false information, manipulation of facts, or harmful framing in this statement.",
            "label": "TRUE"
        },
        {
            "text": "According to the EU Border Agency Frontex, the number of irregular border crossings into the European Union decreased by 13% in 2020 compared to 2019, reaching its lowest level since 2013.",
            "reasoning": "This statement provides specific data from a credible source (EU Border Agency Frontex). The information about the decrease in irregular border crossings can be verified through official Frontex reports. The claim includes precise details (13% decrease, lowest since 2013) and presents the information in a neutral, factual manner without attempting to mislead. The statement does not omit crucial context or present the information in a biased way. This is an accurate representation of migration trends based on official data.",
            "label": "TRUE"
        },
        {
            "text": "Many refugees who resettled in the United States have started small businesses that create jobs and contribute to local economies, with studies showing refugee entrepreneurship rates exceed the national average.",
            "reasoning": "This statement is supported by research from organizations like the Fiscal Policy Institute and New American Economy, which have documented higher rates of entrepreneurship among refugees compared to native-born Americans. Multiple studies confirm that refugee-owned businesses create jobs and contribute to local economic growth. The claim does not exaggerate the positive impact and presents a factual assessment of refugee economic contributions. The information is presented without misleading framing and aligns with established economic research findings.",
            "label": "TRUE"
        }
    ]
    
    # Load dataset
    df = load_dataset(args.input_data, args.text_column, args.sample_size)
    texts = df[args.text_column].tolist()
    
    # Initialize model
    generator = RationaleGenerator(
        model_path=args.model_path, 
        model_type=args.model_type,
        use_peft=args.use_peft
    )
    
    # Generate rationales
    results = []
    for text in tqdm(texts, desc="Generating Rationales"):
        result = generator.generate_rationale(text, examples)
        results.append(result)
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Add original columns
    for col in df.columns:
        if col != args.text_column and col not in output_df.columns:
            if len(df) == len(output_df):
                output_df[col] = df[col].values
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # Save results
    if args.output_file.endswith(".json"):
        output_df.to_json(args.output_file, orient="records", indent=2)
    elif args.output_file.endswith(".jsonl"):
        output_df.to_json(args.output_file, orient="records", lines=True)
    elif args.output_file.endswith(".csv"):
        output_df.to_csv(args.output_file, index=False)
    else:
        output_df.to_json(args.output_file, orient="records", lines=True)
    
    print(f"Rationales saved to {args.output_file}")

if __name__ == "__main__":
    main()