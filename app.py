# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model_path = r"C:\Users\aashutosh kumar\Teachnoo_Project\CIFAR.h5"
model = load_model(model_path)

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  
    return img_array

def make_prediction(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class, predictions

st.title("CIFAR-10 Image Classification")


uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    img_array = preprocess_image(uploaded_file)
    predicted_class, predictions = make_prediction(img_array)

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Class: {classes[predicted_class]}")
    
 
    st.write("Prediction Probabilities:")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{classes[i]}: {prob*100:.2f}%")



