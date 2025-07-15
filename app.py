import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Load model
model = tf.keras.models.load_model('model/best_model.h5')

# Define class names
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Title
st.title("🧠 Brain Tumor MRI Image Classifier")
st.markdown("Upload an MRI scan and let the AI detect the tumor type (if any).")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded MRI", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.markdown(f"### 🧾 Prediction: **{predicted_class}**")
    st.markdown(f"#### 🔍 Confidence: `{confidence*100:.2f}%`")

    # Show all probabilities
    st.markdown("#### 📊 Probability Scores:")
    for i, score in enumerate(predictions):
        st.write(f"- {class_names[i]}: `{score*100:.2f}%`")

sample_dir = "sample_images"
sample_imgs = os.listdir(sample_dir)
selected_img = st.selectbox("🎯 Or choose a sample image:", sample_imgs)

if selected_img:
    img_path = os.path.join(sample_dir, selected_img)
    img = Image.open(img_path).convert("RGB")
    st.image(img, caption="Sample MRI", use_column_width=True)
    
    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.markdown(f"### 🧾 Prediction: **{predicted_class}**")
    st.markdown(f"#### 🔍 Confidence: `{confidence*100:.2f}%`")

    # Show all probabilities
    st.markdown("#### 📊 Probability Scores:")
    for i, score in enumerate(predictions):
        st.write(f"- {class_names[i]}: `{score*100:.2f}%`")
