import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Waste Segregation",
    page_icon="♻️",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("♻️Waste Segregation System")
st.write("Upload a waste image and the AI model will classify it.")

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "waste_segregation_system.keras"
    )

model = load_model()

# -----------------------------
# Class names
# -----------------------------
class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📷 Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict Waste"):

        # Resize
        img = image.resize((224, 224))

        # Convert to array
        img_array = np.array(img)

        # MobileNetV2 preprocessing
        img_array = preprocess_input(img_array)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(prediction[0])
        predicted_class = class_names[predicted_index]
        confidence = prediction[0][predicted_index] * 100

        # -----------------------------
        # Display result
        # -----------------------------
        st.success(
            f"Predicted Waste: {predicted_class.capitalize()}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        # Show probabilities
        st.subheader("Prediction Probabilities")

        for i, class_name in enumerate(class_names):
            probability = prediction[0][i] * 100
            st.write(
                f"{class_name.capitalize()}: {probability:.2f}%"
            )