import os
from dotenv import load_dotenv
import pymrio
import pandas as pd

# Load environment variables from .env file
load_dotenv()

MRIO_EMAIL = os.getenv('mrio_email')
MRIO_PASSWORD = os.getenv('mrio_password')

# Paths
csv_asset_dir = 'csv_asset'
country_class_files = {
    3: 'CountryClass_3.csv',
    7: 'CountryClass_7.csv',
    20: 'CountryClass_20.csv'
}

# Example: Download EORA MRIO (adjust as needed for pymrio)
# eora = pymrio.download('EORA26', email=MRIO_EMAIL, password=MRIO_PASSWORD)
# For local file, use: eora = pymrio.read_mrio('path_to_eora_file')

# Aggregation function
def aggregate_mrio(mrio, group_csv):
    group_df = pd.read_csv(group_csv)
    # Expect columns: country, group
    mapping = dict(zip(group_df['country'], group_df['group']))
    # Aggregate MRIO using pymrio's aggregation
    agg_mrio = pymrio.aggregation.aggregate(mrio, mapping)
    return agg_mrio

if __name__ == '__main__':
    # Download EORA data locally using pymrio's download_eora
    # This will download the EORA26 MRIO database to a local directory
    eora_dir = os.path.join(csv_asset_dir, 'mrio')
    if not os.path.exists(eora_dir):
        os.makedirs(eora_dir)

    # Download EORA database (will prompt for credentials if needed)
    pymrio.download_eora26(eora_dir, MRIO_EMAIL, MRIO_PASSWORD)

    # Read the downloaded MRIO data
    eora_file = os.path.join(eora_dir, 'EORA26_2015.zip')  # Adjust year if needed
    if os.path.exists(eora_file):
        eora = pymrio.read_eora(eora_file)
    else:
        print(f'EORA file not found at {eora_file}. Please check download.')
        eora = None

    for n_agents, csv_file in country_class_files.items():
        group_csv_path = os.path.join(csv_asset_dir, csv_file)
        if eora is not None:
            agg_mrio = aggregate_mrio(eora, group_csv_path)
            # Save aggregated MRIO
            agg_mrio.save(os.path.join(csv_asset_dir, f'eora_agg_{n_agents}.mrio'))
        else:
            print(f'MRIO data not loaded. Please set up eora loading.')
