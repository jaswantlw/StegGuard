import os
import cv2
import numpy as np
import joblib
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

BASE_DIR = os.path.join(PROJECT_ROOT, "archive")
EXTRACTED_DATA_DIR = os.path.join(PROJECT_ROOT, "extracted_data")
SPLITS = ['train', 'val', 'test']

def extract_histogram(image_path):
    """Extract normalized histogram features from an image"""
    if not os.path.exists(image_path):
        return None
        
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    features = []
    for chan in cv2.split(image):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        features.extend(hist.flatten())
    return features

def process_split(split):
    """Process a single dataset split"""
    print(f"[INFO] Processing {split} split...")
    
    nested_dir = os.path.join(BASE_DIR, split, split)
    data, labels = [], []
    
    for subfolder in os.listdir(nested_dir):
        subfolder_path = os.path.join(nested_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
            
        label = 0 if subfolder.lower() == "clean" else 1
        
        images = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for filename in tqdm(images, desc=f"{split}/{subfolder}"):
            img_path = os.path.join(subfolder_path, filename)
            if (features := extract_histogram(img_path)) is not None:
                data.append(features)
                labels.append(label)
    
    if data:
        joblib.dump(
            (np.array(data), np.array(labels)),
            os.path.join(EXTRACTED_DATA_DIR, f"{split}_data.pkl")
        )
        print(f"[SUCCESS] Saved {split} set with {len(data)} samples")
    else:
        print(f"[WARNING] No data found for {split} split")

if __name__ == "__main__":
    for split in SPLITS:
        process_split(split)