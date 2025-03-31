#!/usr/bin/env python3
import os
import subprocess
import json
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Path to the zip mapping file
ZIP_MAPPING_PATH = "/home/ubuntu/ai4gcc-virginia/climate-cooperation-competition/zips/zip_mapping.txt"
# Path to the evaluation script
EVAL_SCRIPT_PATH = "/home/ubuntu/ai4gcc-virginia/climate-cooperation-competition/scripts/evaluate_submission.py"
# Output JSONL file
OUTPUT_JSONL_PATH = "/home/ubuntu/ai4gcc-virginia/climate-cooperation-competition/evaluation_results.jsonl"

def read_zip_mapping(file_path):
    """Read the zip mapping file and return a dictionary"""
    mapping = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, value = line.split(':', 1)
                mapping[key] = value
        logging.info(f"Successfully read mapping file with {len(mapping)} entries")
        return mapping
    except Exception as e:
        logging.error(f"Error reading zip mapping file: {e}")
        return {}

def evaluate_zip_file(zip_path, key, results):
    """Evaluate a single zip file and capture the output"""
    try:
        logging.info(f"Evaluating {key}: {zip_path}")
        
        # Run the evaluation script and capture output
        cmd = ["python", EVAL_SCRIPT_PATH, "-r", zip_path]
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        # Combine stdout and stderr for the full output
        output = stdout
        if stderr:
            output += "\n" + stderr
            
        # Store the result
        results[key] = output
        
        # Write result to JSONL file immediately to avoid losing progress
        with open(OUTPUT_JSONL_PATH, 'a') as f:
            f.write(json.dumps({key: output}) + '\n')
            
        logging.info(f"Completed evaluation for {key}")
        return True
    except Exception as e:
        error_msg = f"Error evaluating {key}: {e}"
        logging.error(error_msg)
        results[key] = f"ERROR: {error_msg}"
        with open(OUTPUT_JSONL_PATH, 'a') as f:
            f.write(json.dumps({key: f"ERROR: {error_msg}"}) + '\n')
        return False

def main():
    # Ensure output file directory exists
    os.makedirs(os.path.dirname(OUTPUT_JSONL_PATH), exist_ok=True)
    
    # Clear the output file if it exists
    if os.path.exists(OUTPUT_JSONL_PATH):
        os.remove(OUTPUT_JSONL_PATH)
    
    # Read the zip mapping
    zip_mapping = read_zip_mapping(ZIP_MAPPING_PATH)
    if not zip_mapping:
        logging.error("Failed to read zip mapping. Exiting.")
        return
    
    # Dictionary to store results
    results = {}
    
    # Process each zip file
    total = len(zip_mapping)
    success_count = 0
    
    for i, (key, zip_path) in enumerate(zip_mapping.items(), 1):
        logging.info(f"Processing {i}/{total}: {key}")
        if os.path.exists(zip_path):
            success = evaluate_zip_file(zip_path, key, results)
            if success:
                success_count += 1
        else:
            error_msg = f"Zip file not found: {zip_path}"
            logging.error(error_msg)
            results[key] = f"ERROR: {error_msg}"
            with open(OUTPUT_JSONL_PATH, 'a') as f:
                f.write(json.dumps({key: f"ERROR: {error_msg}"}) + '\n')
    
    # Log summary
    logging.info(f"Completed evaluation of {success_count}/{total} experiments")
    logging.info(f"Results saved to {OUTPUT_JSONL_PATH}")

if __name__ == "__main__":
    main()