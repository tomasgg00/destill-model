# scripts/run_pipeline.py
import os
import argparse
import subprocess
import logging
import sys
import time
from pathlib import Path

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

def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = ["data", "data/processed", "models", "evaluation_results", "predictions"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the Distilling Step-by-Step pipeline")
    parser.add_argument("--teacher_model", required=True, help="Path to your fine-tuned teacher model")
    parser.add_argument("--data_path", required=True, help="Path to your dataset")
    parser.add_argument("--output_dir", default="models/refugee_misinfo_distilled", help="Output directory for trained model")
    parser.add_argument("--sample_size", type=int, default=5000, help="Number of examples to process (0 for all)")
    parser.add_argument("--skip_steps", type=str, default="", help="Comma-separated list of steps to skip (e.g., '1,2')")
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Parse steps to skip
    skip_steps = [int(s) for s in args.skip_steps.split(",") if s.strip().isdigit()]
    
    # Step 1: Generate rationales
    if 1 not in skip_steps:
        success = run_command(
            [
                "python", "scripts/generate_rationales.py",
                "--model_path", args.teacher_model,
                "--input_data", args.data_path,
                "--output_file", "data/with_rationales.jsonl",
                "--text_column", "content",
                "--batch_size", "1",
                "--sample_size", str(args.sample_size)
            ],
            "Generating rationales using teacher model"
        )
        if not success:
            logger.error("Pipeline failed at step 1")
            return False
    
    # Step 2: Prepare dataset
    if 2 not in skip_steps:
        success = run_command(
            [
                "python", "scripts/prepare_dataset.py",
                "--input_file", "data/with_rationales.jsonl",
                "--output_dir", "data/processed",
                "--text_column", "text",
                "--test_size", "0.1",
                "--val_size", "0.1"
            ],
            "Preparing dataset for training"
        )
        if not success:
            logger.error("Pipeline failed at step 2")
            return False
    
    # Step 3: Train model
    if 3 not in skip_steps:
        success = run_command(
            [
                "python", "scripts/train_model.py",
                "--model_path", "google/modernbert-base",
                "--train_file", "data/processed/train.jsonl",
                "--val_file", "data/processed/val.jsonl",
                "--output_dir", args.output_dir,
                "--num_epochs", "3",
                "--batch_size", "8",
                "--learning_rate", "2e-5",
                "--use_lora"
            ],
            "Training model with multi-task learning"
        )
        if not success:
            logger.error("Pipeline failed at step 3")
            return False
    
    # Step 4: Evaluate model
    if 4 not in skip_steps:
        success = run_command(
            [
                "python", "scripts/evaluate_model.py",
                "--model_path", args.output_dir,
                "--test_file", "data/processed/test.jsonl",
                "--output_dir", "evaluation_results",
                "--batch_size", "8"
            ],
            "Evaluating model"
        )
        if not success:
            logger.error("Pipeline failed at step 4")
            return False
    
    # Step 5: Run inference on UNHCR validation set
    if 5 not in skip_steps:
        # Check if the file exists before running this step
        if os.path.exists("data/unhcr_validation.json"):
            success = run_command(
                [
                    "python", "scripts/inference.py",
                    "--model_path", args.output_dir,
                    "--input_file", "data/unhcr_validation.json",
                    "--output_file", "predictions/unhcr_predictions.json",
                    "--text_column", "content",
                    "--batch_size", "8"
                ],
                "Running inference on UNHCR validation set"
            )
            if not success:
                logger.error("Pipeline failed at step 5")
                return False
        else:
            logger.warning("UNHCR validation file not found, skipping inference step")
    
    logger.info("Pipeline completed successfully!")
    return True

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Total pipeline execution time: {elapsed_time/60:.2f} minutes")