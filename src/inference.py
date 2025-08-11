
import sys
import os
import numpy as np
import cv2
import glob
from mesonet import Meso4

# Check arguments
if len(sys.argv) < 2:
    print("Usage: python inference.py <base_frames_folder>")
    sys.exit(1)

base_folder = sys.argv[1]  # e.g., data/frames/test_video

# Load model
model = Meso4()
model.load('models/Meso4_DF.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    return img

def predict_folder(folder_path):
    frame_paths = glob.glob(f"{folder_path}/*.jpg")
    if not frame_paths:
        print(f"No images found in {folder_path}")
        return None, None

    predictions = []
    for path in frame_paths:
        img = preprocess_image(path)
        pred = model.predict(np.expand_dims(img, axis=0))[0][0]
        print(f"{path}: {'Real' if pred > 0.5 else 'Fake'} ({pred:.4f})")
        predictions.append(pred)

    avg_score = np.mean(predictions)
    label = "Real" if avg_score > 0.5 else "Fake"
    return label, avg_score

if __name__ == "__main__":
    for subfolder in ['real', 'fake']:
        folder_path = os.path.join(base_folder, subfolder)
        if os.path.exists(folder_path):
            print(f"\n--- Predicting for folder: {subfolder.upper()} ---")
            label, score = predict_folder(folder_path)
            if label is not None:
                print(f"FINAL RESULT for {subfolder}: {label} (Score: {score:.4f})")
