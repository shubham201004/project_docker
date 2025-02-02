import os
import tempfile
import tensorflow as tf
import numpy as np
import streamlit as st
from openai import OpenAI
import openai  # Import the OpenAI library

# Ensure the OpenAI API key is loaded from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("API key not found! Please check the Dockerfile and ensure OPENAI_API_KEY is set.")

# Set OpenAI API key
openai.api_key = openai_api_key

def model_prediction(test_image_path):
    """Loads model and predicts the class of the given image."""
    model_path = "/code/trained_model.keras"  # Model path inside Docker container
    model = tf.keras.models.load_model(model_path)
    
    # Load image
    image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Expand dims for model input

    # Predict
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar menu
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Disease Recognition Page
if app_mode == "Disease Recognition":
    st.header("Plant Disease Recognition System")
    
    test_image = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
    
    if test_image:
        st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.write("Processing Image...")

            # Save uploaded image to a temporary directory inside the container
            temp_dir = "/code/tmp"  # Define temp directory inside the container
            os.makedirs(temp_dir, exist_ok=True)  # Create temp directory if it doesn't exist
            temp_image_path = os.path.join(temp_dir, "temp_image.jpg")
            
            with open(temp_image_path, "wb") as f:
                f.write(test_image.getbuffer())

            # Get model prediction
            result_index = model_prediction(temp_image_path)

            # Class names (same as before)
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
                'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            predicted_output = class_name[result_index]

            # Format disease name
            name_of_disease = predicted_output.replace('_', ' ').replace('___', ' - ')

            st.success(f"Model Prediction: **{name_of_disease}**")

            # OpenAI prompt
            prompt = f"Given the following plant disease: {name_of_disease}, provide 5 ways to cure it and 5 reasons why it happens."
            
            if prompt:
                try:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}]
                    )

                    # Print raw response to debug structure
                    print("Raw OpenAI Response:", completion)

                    # Extract and display response properly
                    output = completion.choices[0].message.content.strip()
                    st.info(f"**AI Response:**\n\n{output}")

                except Exception as e:
                    st.error(f"An error occurred while fetching AI response: {e}")
            else:
                st.header("Enter a prompt ")
