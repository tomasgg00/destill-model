#!/usr/bin/env python
# pipeline.py
"""
Main pipeline script that orchestrates the entire distillation process.
This runs all the steps in sequence:
1. Generate rationales from teacher model
2. Prepare dataset for multi-task learning
3. Train multi-task model
4. Evaluate model
5. Run inference on unlabeled UNHCR data
"""

import os
import argparse
import subprocess
import logging
import sys
import time
import json
from pathlib import Path
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and log its output."""
    logger.info(f"Starting: {description}")
    
    try:
        # Run the command with real-time output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Real-time output handling
        while True:
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()
            
            if stdout_line == '' and stderr_line == '' and process.poll() is not None:
                break
                
            if stdout_line:
                logger.info(stdout_line.strip())
            if stderr_line:
                logger.warning(stderr_line.strip())
                
        # Get return code
        return_code = process.poll()
        
        if return_code == 0:
            logger.info(f"Successfully completed: {description}")
            return True
        else:
            logger.error(f"Command failed with exit code {return_code}: {description}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in {description}: {e}")
        logger.error(f"Command output: {e.output}")
        logger.error(f"Command stderr: {e.stderr}")
        return False

def create_directories(dirs):
    """Create necessary directories if they don't exist."""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Run the distillation pipeline")
    
    # Basic pipeline configuration
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--skip_steps", type=str, default="", help="Comma-separated list of steps to skip (e.g., '1,2')")
    parser.add_argument("--start_from_step", type=int, default=1, help="Start execution from this step (1-5)")
    parser.add_argument("--run_only_step", type=int, default=None, help="Run only this specific step (1-5)")
    
    # Step 1: Generate rationales
    parser.add_argument("--teacher_model", type=str, help="Path/name of teacher model")
    parser.add_argument("--model_type", type=str, choices=["llama", "gemma", "deepseek"], help="Type of teacher model")
    parser.add_argument("--input_data", type=str, help="Path to input dataset")
    parser.add_argument("--validation_data", type=str, help="Path to validation dataset")
    parser.add_argument("--sample_size", type=int, help="Number of examples to process")
    
    # Step 2: Prepare dataset
    parser.add_argument("--balance_classes", action="store_true", help="Balance classes in training data")
    
    # Step 3: Train model
    parser.add_argument("--student_model", type=str, help="Base model for student")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for label task (1-alpha for rationale task)")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    # Step 4: Evaluate model
    parser.add_argument("--generate_rationales_eval", action="store_true", help="Generate rationales during evaluation")
    
    # Step 5: Run inference
    parser.add_argument("--unhcr_data", type=str, help="Path to unlabeled UNHCR data")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Override args with config values if not provided on command line
        for key, value in config.items():
            if not getattr(args, key, None):
                setattr(args, key, value)
    
    # Create output directory structure
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_output_dir = os.path.join(args.output_dir, timestamp)
    
    dirs = [
        base_output_dir,
        os.path.join(base_output_dir, "rationales"),
        os.path.join(base_output_dir, "processed_data"),
        os.path.join(base_output_dir, "model"),
        os.path.join(base_output_dir, "evaluation"),
        os.path.join(base_output_dir, "predictions")
    ]
    create_directories(dirs)
    
    # Save configuration
    config_output = vars(args)
    with open(os.path.join(base_output_dir, "config.json"), 'w') as f:
        json.dump(config_output, f, indent=4)
    
    # Parse steps to skip
    skip_steps = []
    if args.skip_steps:
        skip_steps = [int(s) for s in args.skip_steps.split(",") if s.strip().isdigit()]
    
    # Determine starting step
    start_step = args.start_from_step
    
    # Determine which steps to run
    if args.run_only_step:
        steps_to_run = [args.run_only_step]
    else:
        steps_to_run = list(range(start_step, 6))  # Steps 1-5
        # Remove skipped steps
        steps_to_run = [step for step in steps_to_run if step not in skip_steps]
    
    logger.info(f"Will run steps: {steps_to_run}")
    
    # Step paths
    rationales_path = os.path.join(base_output_dir, "rationales", "data_with_rationales.jsonl")
    processed_dir = os.path.join(base_output_dir, "processed_data")
    train_path = os.path.join(processed_dir, "train.jsonl")
    val_path = os.path.join(processed_dir, "val.jsonl")
    test_path = os.path.join(processed_dir, "test.jsonl")
    model_dir = os.path.join(base_output_dir, "model")
    eval_dir = os.path.join(base_output_dir, "evaluation")
    predictions_dir = os.path.join(base_output_dir, "predictions")
    
    # ======== Step 1: Generate rationales ========
    if 1 in steps_to_run:
        logger.info("=== STEP 1: Generating rationales ===")
        
        if not args.teacher_model:
            logger.error("Teacher model not specified. Use --teacher_model or provide it in config.")
            return False
        
        if not args.input_data:
            logger.error("Input data not specified. Use --input_data or provide it in config.")
            return False
        
        # Build command
        command = [
            "python", "generate_rationales.py",
            "--model_path", args.teacher_model,
            "--input_file", args.input_data,
            "--output_file", rationales_path,
            "--batch_size", "1"  # Cautious with memory
        ]
        
        # Add optional arguments
        if args.model_type:
            command.extend(["--model_type", args.model_type])
        
        if args.validation_data:
            command.extend(["--validation_file", args.validation_data])
        
        if args.sample_size:
            command.extend(["--sample_size", str(args.sample_size)])
        
        # Run command
        success = run_command(
            command,
            "Generating rationales"
        )
        
        if not success:
            logger.error("Pipeline failed at step 1: Generate rationales")
            return False
    
    # ======== Step 2: Prepare dataset ========
    if 2 in steps_to_run:
        logger.info("=== STEP 2: Preparing dataset ===")
        
        # Check if previous step output exists if we didn't run it
        if 1 not in steps_to_run and not os.path.exists(rationales_path):
            logger.error(f"Input file for step 2 does not exist: {rationales_path}")
            return False
        
        # Build command
        command = [
            "python", "prepare_dataset.py",
            "--input_file", rationales_path,
            "--output_dir", processed_dir,
            "--val_size", "0.1"
        ]
        
        # Add optional arguments
        if args.balance_classes:
            command.append("--balance_labels")
        
        if args.validation_data:
            command.extend(["--validation_file", args.validation_data])
        
        # Run command
        success = run_command(
            command,
            "Preparing dataset"
        )
        
        if not success:
            logger.error("Pipeline failed at step 2: Prepare dataset")
            return False
    
    # ======== Step 3: Train multi-task model ========
    if 3 in steps_to_run:
        logger.info("=== STEP 3: Training multi-task model ===")
        
        # Check if previous step outputs exist if we didn't run it
        if 2 not in steps_to_run and (not os.path.exists(train_path) or not os.path.exists(val_path)):
            logger.error(f"Input files for step 3 do not exist: {train_path} or {val_path}")
            return False
        
        # Build command
        command = [
            "python", "train_multitask_model.py",
            "--model_name", args.student_model or "t5-base",
            "--train_file", train_path,
            "--val_file", val_path,
            "--output_dir", model_dir,
            "--alpha", str(args.alpha),
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs)
        ]
        
        # Add optional arguments
        if args.use_lora:
            command.append("--use_lora")
        
        # Run command
        success = run_command(
            command,
            "Training multi-task model"
        )
        
        if not success:
            logger.error("Pipeline failed at step 3: Train multi-task model")
            return False
    
    # ======== Step 4: Evaluate model ========
    if 4 in steps_to_run:
        logger.info("=== STEP 4: Evaluating model ===")
        
        # Check if previous step output exists if we didn't run it
        if 3 not in steps_to_run and not os.path.exists(model_dir):
            logger.error(f"Model directory for step 4 does not exist: {model_dir}")
            return False
        
        # Check if test file exists
        if not os.path.exists(test_path):
            logger.warning(f"Test file {test_path} not found, using validation set for evaluation")
            test_path = val_path
        
        # Build command
        command = [
            "python", "evaluate_model.py",
            "--model_path", model_dir,
            "--test_file", test_path,
            "--output_dir", eval_dir
        ]
        
        # Add validation data if available
        if args.validation_data:
            command.extend(["--validation_file", args.validation_data])
        
        # Add optional arguments
        if args.generate_rationales_eval:
            command.append("--generate_rationales")
        
        # Run command
        success = run_command(
            command,
            "Evaluating model"
        )
        
        if not success:
            logger.error("Pipeline failed at step 4: Evaluate model")
            return False
    
    # ======== Step 5: Run inference on UNHCR data ========
    if 5 in steps_to_run and args.unhcr_data:
        logger.info("=== STEP 5: Running inference on UNHCR data ===")
        
        # Check if previous step output exists if we didn't run it
        if 3 not in steps_to_run and not os.path.exists(model_dir):
            logger.error(f"Model directory for step 5 does not exist: {model_dir}")
            return False
        
        # Build command
        command = [
            "python", "inference.py",
            "--model_path", model_dir,
            "--input_file", args.unhcr_data,
            "--output_file", os.path.join(predictions_dir, "unhcr_predictions.jsonl")
        ]
        
        # Add optional arguments
        if args.generate_rationales_eval:
            command.append("--generate_rationales")
        
        # Run command
        success = run_command(
            command,
            "Running inference on UNHCR data"
        )
        
        if not success:
            logger.error("Pipeline failed at step 5: Run inference")
            return False
    elif 5 in steps_to_run and not args.unhcr_data:
        logger.warning("UNHCR data path not provided, skipping inference step")
    
    # Create a copy of the scripts in the output directory for reproducibility
    script_files = [
        "generate_rationales.py",
        "prepare_dataset.py",
        "train_multitask_model.py",
        "evaluate_model.py",
        "inference.py",
        "pipeline.py"
    ]
    
    scripts_dir = os.path.join(base_output_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    
    for script in script_files:
        if os.path.exists(script):
            shutil.copy(script, os.path.join(scripts_dir, script))
    
    logger.info(f"Pipeline completed successfully! Results saved in {base_output_dir}")
    
    # Print summary
    logger.info("=== Pipeline Summary ===")
    logger.info(f"Teacher model: {args.teacher_model}")
    logger.info(f"Student model: {args.student_model or 't5-base'}")
    logger.info(f"Alpha (label weight): {args.alpha}")
    logger.info(f"Steps executed: {steps_to_run}")
    
    # Print the paths to key outputs
    logger.info("=== Output Locations ===")
    logger.info(f"Rationales: {rationales_path}")
    logger.info(f"Processed data: {processed_dir}")
    logger.info(f"Trained model: {model_dir}")
    logger.info(f"Evaluation results: {eval_dir}")
    logger.info(f"UNHCR predictions: {os.path.join(predictions_dir, 'unhcr_predictions.jsonl')}")
    
    return True

if __name__ == "__main__":
    start_time = time.time()
    success = main()
    elapsed_time = time.time() - start_time
    
    if success:
        logger.info(f"Total pipeline execution time: {elapsed_time/60:.2f} minutes")
    else:
        logger.error(f"Pipeline failed after {elapsed_time/60:.2f} minutes")
        sys.exit(1)
        