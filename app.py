import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/mobilenet_model.h5')
    return model

model = load_model()

# Class names (make sure this order matches your training generator)
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# App title
st.title("🧠 Brain Tumor Classifier")
st.markdown("Upload an MRI scan and the model will classify it into one of the tumor types.")

# Upload image
uploaded_file = st.file_uploader("Upload Brain MRI Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded MRI Image', use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### 🔍 Prediction: **{predicted_class}**")
    st.markdown(f"### 📈 Confidence: `{confidence*100:.2f}%`")

    st.markdown("### 🧪 Confidence for each class:")
    for i in range(len(class_names)):
        st.write(f"- {class_names[i]}: `{prediction[i]*100:.2f}%`")
