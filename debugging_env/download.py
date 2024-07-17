import wandb
import pandas as pd
from tqdm import tqdm
import csv

# Initialize wandb API
api = wandb.Api()

# Specify your project
project_name = "tianyuzhang/ricen-fair-abatement-function-debugging"

# Get all runs in the project
runs = api.runs(project_name)

# Define the CSV file name
csv_filename = "all_project_data.csv"

# Initialize the CSV file with headers
first_run = next(iter(runs))
first_history = first_run.history()
headers = list(first_history.columns) + ["run_id", "run_name"]

with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()

# Iterate through all runs
for run in tqdm(runs):
    # Get the run's history
    history = run.history()

    # Add run information to each row
    history["run_id"] = run.id
    history["run_name"] = run.name

    # Append to the CSV file
    with open(csv_filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        for _, row in history.iterrows():
            writer.writerow(row.to_dict())

print(f"Data saved to {csv_filename}")
