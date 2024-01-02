import os
import shutil
import pandas as pd

# Define paths
base_folder = "/home/santosh.sanjeev/model-soups/my_soups/rsna_18/train_full/"
csv_file = "your_csv_file.csv"  # Replace with your actual CSV file name

# Create folders if they don't exist
os.makedirs(os.path.join(base_folder, "train", "0"), exist_ok=True)
os.makedirs(os.path.join(base_folder, "train", "1"), exist_ok=True)
os.makedirs(os.path.join(base_folder, "val", "0"), exist_ok=True)
os.makedirs(os.path.join(base_folder, "val", "1"), exist_ok=True)
os.makedirs(os.path.join(base_folder, "test", "0"), exist_ok=True)
os.makedirs(os.path.join(base_folder, "test", "1"), exist_ok=True)

# Load CSV file
df = pd.read_csv(os.path.join(base_folder, csv_file))

# Iterate through the folder
for filename in os.listdir(os.path.join(base_folder, "train")):
    if filename.endswith(".jpg"):  # Assuming image files, modify if needed
        file_path = os.path.join(base_folder, "train", filename)
        target = df.loc[df['filename'] == filename, 'target'].values[0]
        shutil.move(file_path, os.path.join(base_folder, "train", str(target), filename))

for filename in os.listdir(os.path.join(base_folder, "val")):
    if filename.endswith(".jpg"):  # Assuming image files, modify if needed
        file_path = os.path.join(base_folder, "val", filename)
        target = df.loc[df['filename'] == filename, 'target'].values[0]
        shutil.move(file_path, os.path.join(base_folder, "val", str(target), filename))

for filename in os.listdir(os.path.join(base_folder, "test")):
    if filename.endswith(".jpg"):  # Assuming image files, modify if needed
        file_path = os.path.join(base_folder, "test", filename)
        target = df.loc[df['filename'] == filename, 'target'].values[0]
        shutil.move(file_path, os.path.join(base_folder, "test", str(target), filename))
