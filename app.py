import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# Load the model
model = load_model('model/paddy_model.keras')

# Define class labels
class_labels = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 
                'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']

# SIFT feature visualization
def show_sift_features(img_pil):
    img_cv = np.array(img_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)
    img_sift = cv2.drawKeypoints(img_cv, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_sift_rgb = cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB)
    return img_sift_rgb

# LoG feature visualization
def show_log_features(img_pil, sigma=1.0):
    img_cv = np.array(img_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    log_img = cv2.GaussianBlur(gray, (0, 0), sigma)
    log_img = cv2.Laplacian(log_img, cv2.CV_64F)
    log_img = cv2.convertScaleAbs(log_img)
    return log_img

# Streamlit UI
st.set_page_config(page_title="Rice Leaf Disease Classifier", layout="centered")

st.title("🌾 Rice Leaf Disease Classifier")
st.write("Upload an image of a rice leaf to detect disease type.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array_expanded = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array_expanded)
    predicted_idx = np.argmax(prediction)
    confidence = prediction[0][predicted_idx] * 100
    predicted_label = class_labels[predicted_idx]

    # Show result
    st.success(f"**Predicted Class:** {predicted_label}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Show SIFT
    st.subheader("🔍 SIFT Key Features")
    sift_img = show_sift_features(img)
    st.image(sift_img, use_column_width=True, caption="SIFT Keypoints")

    # Show LoG
    st.subheader("🌀 Laplacian of Gaussian (LoG) Features")
    log_img = show_log_features(img)
    st.image(log_img, clamp=True, use_column_width=True, caption="LoG Edge Map")
