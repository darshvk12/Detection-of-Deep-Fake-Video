import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import streamlit as st
import cv2
import numpy as np
import tempfile
import glob
from mesonet import Meso4

# Load model once
@st.cache_resource
def load_model():
    model = Meso4()
    model.load("models/Meso4_DF.h5")
    return model

model = load_model()

# Preprocess image
def preprocess_image(img):
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    return img

# Predict single image
def predict_image(img):
    processed = preprocess_image(img)
    pred = model.predict(np.expand_dims(processed, axis=0))[0][0]
    label = "Real" if pred > 0.5 else "Fake"
    return label, pred

# Extract frames from video
def extract_frames(video_path, every_n_frames=10):
    temp_dir = tempfile.mkdtemp()
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n_frames == 0:
            cv2.imwrite(os.path.join(temp_dir, f"frame_{saved}.jpg"), frame)
            saved += 1
        count += 1
    cap.release()
    return temp_dir

# Predict video
def predict_video(video_path):
    frames_folder = extract_frames(video_path)
    frame_paths = glob.glob(f"{frames_folder}/*.jpg")
    predictions = []
    for path in frame_paths:
        img = cv2.imread(path)
        _, score = predict_image(img)
        predictions.append(score)
    avg_score = np.mean(predictions)
    label = "Real" if avg_score > 0.5 else "Fake"
    return label, avg_score

# Streamlit UI
st.title("🔍 Deepfake Video & Image Detector")
st.write("Upload a video or image to check if it's real or fake.")

uploaded_file = st.file_uploader("Upload File", type=["mp4", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    # Handle image upload
    if "image" in file_type:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
        label, score = predict_image(img)
        
        if label == "Real":
            st.markdown(f"<h3 style='color:green;'>Prediction: {label} ({score:.4f})</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:red;'>Prediction: {label} ({score:.4f})</h3>", unsafe_allow_html=True)

    # Handle video upload
    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.video(tfile.name)
        label, score = predict_video(tfile.name)

        if label == "Real":
            st.markdown(f"<h3 style='color:green;'>Prediction: {label} ({score:.4f})</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:red;'>Prediction: {label} ({score:.4f})</h3>", unsafe_allow_html=True)

