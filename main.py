import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Load your trained models
model_path = r'D:\Venkatesh\Projects\CNN Projects\Gender classification project\Male vs Female trained model_50 tt_epochs VGG  part 2 till now best.h5'
# model_path = 'Male vs Female trained model_50 tt_epochs VGG  part 2 till now best.h5'
model = tf.keras.models.load_model(model_path)


# Define function to make predictions
def predict_gender(image):
    # Resize the image
    resized_image = image.resize((256, 256))

    # Convert to numpy array
    img_array = np.array(resized_image)

    # Reshape for model input
    test_input = np.expand_dims(img_array, axis=0)

    # Predict gender
    prediction = model.predict(test_input)

    # Assuming 'prediction' contains the probability value
    probability = prediction[0][0]

    # Apply a threshold (0.5 in this case) to classify as 0 or 1
    if probability >= 0.5:
        return "Male"
    else:
        return "Female"


# Define the Streamlit UI
st.title('Gender Classification App of Geetha')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","webp"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction when 'Predict' button is clicked
    if st.button('Predict'):
        prediction = predict_gender(image)
        st.title('Prediction: {}'.format(prediction))
