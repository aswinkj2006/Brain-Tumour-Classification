import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

# Class labels
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Load model
@st.cache_resource
def load_model():
    input_tensor = Input(shape=(224, 224, 3))
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(4, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.load_weights("model/mobilenet_weights.h5")
    return model

model = load_model()

# App title
st.title("🧠 Brain Tumor Classifier")
st.markdown("Upload a brain MRI or choose a sample image to detect tumor type.")

# Upload section
uploaded_file = st.file_uploader("📤 Upload your MRI image", type=['jpg', 'jpeg', 'png'])

# Sample image selector
sample_dir = "sample_images"
sample_options = os.listdir(sample_dir) if os.path.exists(sample_dir) else []
selected_sample = st.selectbox("📁 Or choose a sample image:", ["None"] + sample_options)

# Determine the image to use
image_to_predict = None
if uploaded_file:
    image_to_predict = Image.open(uploaded_file).convert("RGB")
elif selected_sample != "None":
    image_path = os.path.join(sample_dir, selected_sample)
    image_to_predict = Image.open(image_path).convert("RGB")

# Prediction section
if image_to_predict:
    st.image(image_to_predict, caption="Selected Image", use_column_width=True)

    img = image_to_predict.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### 🔍 Prediction: **{predicted_class}**")
    st.markdown(f"### 📈 Confidence: `{confidence * 100:.2f}%`")

    st.markdown("### 🧪 Class Probabilities:")
    for i in range(len(class_names)):
        st.write(f"- {class_names[i]}: `{prediction[i] * 100:.2f}%`")
