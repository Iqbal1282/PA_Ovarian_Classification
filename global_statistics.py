import pandas as pd
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

# Load your CSV file
csv_path = "PAT features/roi_thb_image_metadata.csv"
df = pd.read_csv(csv_path)

# List of image paths
image_paths = df['Filename'].tolist()

# Initialize accumulators
sum_pixels = 0.0
sum_squared_pixels = 0.0
num_pixels = 0
global_max = -np.inf

# Iterate through each image to accumulate stats
for path in tqdm(image_paths, desc="Computing stats"):
    path = f'PAT features/ROI_MAT/{path}'  # Adjust path if necessary
    mat = loadmat(path)
    image = mat['img']  # Adjust based on your .mat structure
    image = np.squeeze(image).astype(np.float32)  # Remove singleton dimensions if any.
    #image = np.load(path).astype(np.float32)  # ensure float for safe math

    sum_pixels += image.sum()
    sum_squared_pixels += (image ** 2).sum()
    num_pixels += image.size
    global_max = max(global_max, image.max())

# Calculate global mean and std
global_mean = sum_pixels / num_pixels
global_std = np.sqrt((sum_squared_pixels / num_pixels) - global_mean ** 2)

# Show results
print(f"Global Mean: {global_mean:.4f}")
print(f"Global Std: {global_std:.4f}")
print(f"Global Max: {global_max:.4f}")
