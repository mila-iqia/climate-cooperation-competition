import os
import zipfile

# Directory containing MRIO zips
eora_dir = os.path.join('csv_asset', 'mrio')
unpack_dir = os.path.join(eora_dir, 'unzipped')

if not os.path.exists(unpack_dir):
    os.makedirs(unpack_dir)

# Find all zip files in eora_dir
for fname in os.listdir(eora_dir):
    if fname.endswith('.zip'):
        zip_path = os.path.join(eora_dir, fname)
        target_folder = os.path.join(unpack_dir, fname.replace('.zip', ''))
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_folder)
        print(f"Unpacked {fname} to {target_folder}")
