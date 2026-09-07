# ♻️ Waste Segregation System

## Project Overview

The AI Waste Segregation System is a deep learning-based image classification project that automatically identifies different types of waste from an uploaded image.

The system uses **MobileNetV2 Transfer Learning** to classify waste into six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

A **Streamlit web application** is used to provide a simple user interface for uploading an image and getting the predicted waste category.

---

## Objectives

- Automatically classify waste using an image.
- Reduce manual waste segregation.
- Use deep learning for image classification.
- Build an easy-to-use web interface.
- Deploy the trained model using Streamlit.

---

## Technology Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pillow
- Streamlit
- Jupyter Notebook / Google Colab

---

## Waste Categories

The model classifies images into:

| Class | Description |
|---|---|
| Cardboard | Cardboard waste |
| Glass | Glass waste |
| Metal | Metal waste |
| Paper | Paper waste |
| Plastic | Plastic waste |
| Trash | Other/general waste |

---

## Model

The project uses **MobileNetV2** with Transfer Learning.

### Model Architecture

- MobileNetV2 as the base model
- Image size: 224 × 224
- Global feature extraction using MobileNetV2
- Dense layer with 256 neurons
- Dropout layer
- Output layer with 6 classes
- Softmax activation for classification

The model was fine-tuned using a low learning rate to improve classification performance.

---

## Model Evaluation

The model was evaluated using:

- Accuracy
- Loss
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-score

The final model achieved approximately **82% validation accuracy**.

---

## Streamlit Application

The Streamlit application allows the user to:

1. Upload a waste image.
2. Display the uploaded image.
3. Predict the waste category.
4. Display prediction confidence.
5. Display probabilities for all six classes.

---

## Project Structure

```text
Waste_Segregation_ML/
│
├── Waste_Segregation.ipynb
├── app.py
├── waste_segregation_mobilenetv2.keras
├── requirements.txt
└── README.md