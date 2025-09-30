import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths and labels
# -----------------------------
csv_path = r"C:\Users\HP\OneDrive\Desktop\deep_learning\face\fer2013.csv"  # change if your CSV is elsewhere
output_dir = "dataset"

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Create directories
for split in ['train', 'test']:
    for emotion in emotions:
        os.makedirs(os.path.join(output_dir, split, emotion), exist_ok=True)

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(csv_path)

# -----------------------------
# Convert pixel strings to images
# -----------------------------
def pixels_to_image(pixels):
    arr = np.array(pixels.split(), dtype=np.uint8).reshape(48,48)
    return Image.fromarray(arr)

# -----------------------------
# Split dataset into train/test
# -----------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])

# -----------------------------
# Function to save images
# -----------------------------
def save_images(dataframe, split):
    counters = {e:0 for e in emotions}
    
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        emotion_idx = row['emotion']
        emotion_name = emotions[emotion_idx]
        img = pixels_to_image(row['pixels'])
        filename = f"im{counters[emotion_name]}.png"
        img.save(os.path.join(output_dir, split, emotion_name, filename))
        counters[emotion_name] += 1

# Save train images
print("Saving train images...")
save_images(train_df, 'train')

# Save test images
print("Saving test images...")
save_images(test_df, 'test')

print("✅ All images saved in 'data/train/' and 'data/test/' folders.")
