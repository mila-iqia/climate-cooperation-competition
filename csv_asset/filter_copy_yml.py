import os
import shutil

# List of country codes to check against file names
country_codes = [
    "ZAF",
    "NGA",
    "ETH",
    "SYC",
    "MDV",
    "IND",
    "USA",
    "IRQ",
    "DZA",
    "SAU",
    "BRA",
    "BOL",
    "CHL",
    "TUR",
    "UZB",
    "DEU",
    "CHN",
    "IDN",
    "JPN",
]

# Paths to the source and output directories
source_directory_path = "H:\\vscode_git\\mila-sfdc-climate-econ\\region_yamls_190"
output_directory_path = (
    "H:\\vscode_git\\mila-sfdc-climate-econ\\pop_gdp_rank1_countries"
)

# Ensure output directory exists
os.makedirs(output_directory_path, exist_ok=True)

# Loop through all files in the source directory
for filename in os.listdir(source_directory_path):
    # Check if the file is a YAML file and its name matches the country codes
    if (
        filename.split(".")[-1].lower() == "yml"
        and filename.split(".")[0].upper() in country_codes
    ):
        # Construct the full paths for the source and destination
        source_path = os.path.join(source_directory_path, filename)
        destination_path = os.path.join(output_directory_path, filename)

        # Copy the file to the output directory
        shutil.copy2(source_path, destination_path)
        print(f"Copied: {filename}")

print("Copying complete.")
