#!/usr/bin/env python3
import os
import zipfile
import shutil
from pathlib import Path

# Base paths
base_tmp_path = "/home/ubuntu/ai4gcc-virginia/climate-cooperation-competition/tmp"
zip_output_path = "/home/ubuntu/ai4gcc-virginia/climate-cooperation-competition/zips"

# Ensure the output directory exists
os.makedirs(zip_output_path, exist_ok=True)

# Experiment types to process
experiment_types = [
    "discrete_basic_club",
    "discrete_bilateral",
    "discrete_max_mitigation",
    "discrete_min_mitigation",
    "discrete_no_nego"
]

# Perturbation values
perturbation_values = ["0.01", "0.02", "0.03", "0.04", "-0.01", "-0.02", "-0.03", "-0.04"]

# Dictionary to store the mapping
zip_mapping = {}

# Process each combination
for exp_type in experiment_types:
    for perturbation in perturbation_values:
        perturbation_path = os.path.join(base_tmp_path, exp_type, "experiments", "is_perturbation", perturbation)
        
        # Skip if this combination doesn't exist
        if not os.path.exists(perturbation_path):
            print(f"Skipping {perturbation_path} - path does not exist")
            continue
        
        # Get all experiment IDs in this perturbation
        try:
            exp_ids = [d for d in os.listdir(perturbation_path) if os.path.isdir(os.path.join(perturbation_path, d))]
        except FileNotFoundError:
            print(f"Could not find any experiment IDs in {perturbation_path}")
            continue
        
        for exp_id in exp_ids:
            # Construct the key
            key = f"{exp_type}_{perturbation}_{exp_id}"
            
            # Source directory to zip
            source_dir = os.path.join(perturbation_path, exp_id)
            
            # Output zip file path
            zip_file_path = os.path.join(zip_output_path, f"{key}.zip")
            
            # Skip if source directory doesn't exist
            if not os.path.exists(source_dir):
                print(f"Skipping {source_dir} - path does not exist")
                continue
            
            print(f"Processing {key}...")
            
            # Create a zip file of the source directory
            try:
                with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(source_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, source_dir)
                            zipf.write(file_path, arcname)
                
                # Add to mapping
                zip_mapping[key] = zip_file_path
                print(f"Successfully created {zip_file_path}")
            except Exception as e:
                print(f"Error zipping {source_dir}: {e}")
                continue

# Print summary
print("\nZip file mapping created:")
print(f"Total entries: {len(zip_mapping)}")
print("Sample entries:")
for i, (key, path) in enumerate(list(zip_mapping.items())[:5]):
    print(f"  {key} -> {path}")

# Optionally save the mapping to a file
mapping_file = os.path.join(zip_output_path, "zip_mapping.txt")
with open(mapping_file, 'w') as f:
    for key, path in zip_mapping.items():
        f.write(f"{key}:{path}\n")

print(f"\nMapping saved to {mapping_file}")