import os
import tempfile
import tensorflow as tf
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import sys
from functools import lru_cache
from PIL import Image
import io
import time
import zipfile
import tempfile
from tensorflow.keras.models import model_from_json
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import warnings
warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))

os.environ["OPENAI_API_KEY"]=st.secrets["api_key"]["OPENAI_API_KEY"]

try:
    load_dotenv(
        dotenv_path = os.path.join(current_dir,'..','.env')
    )
except Exception as e:
    print('Cannot Load Env')
    sys.exit(1)

model_name_zip = os.getenv('KERAS_MODEL_NAME_ZIP')
model_name_json = os.getenv('KERAS_MODE_ARCH')

model_name_zip_path = os.path.join(current_dir,'..','models',model_name_zip)
model_name_json_path = os.path.join(current_dir,'..','models',model_name_json)

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

def load_model():

    global model_name_json_path
    global model_name_zip_path

    with open(model_name_json_path,'r+') as f:
        model = model_from_json(f.read())

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(model_name_zip_path, 'r') as zi:
            for file in zi.namelist():
                extracted_file_path = os.path.join(temp_dir, file)
                os.makedirs(os.path.dirname(extracted_file_path), exist_ok=True)
                with zi.open(file) as f_in:
                    with open(extracted_file_path, 'wb') as f_out:
                        f_out.write(f_in.read())  
                model.load_weights(extracted_file_path)
    
    return model

@lru_cache()
def model_prediction(test_image_path):
    global model_path
    model = load_model()
    
    image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)

    return result_index

@lru_cache()
def generate_ai_results(name_of_prediction):

    Prompt = PromptTemplate(
        input_variables = ["name_of_disease"],
        template = "Given the following plant Disease: {name_of_disease}, Give me 5 ways to cure them and 5 reasons why does it happen."
    )
    Large_language_model = ChatOpenAI(
        model_name="gpt-4o-mini",
    ) 
    chain = Prompt | Large_language_model | StrOutputParser()

    output = chain.invoke(
        {
            "name_of_disease" :  name_of_prediction
        }
    )
    
    st.markdown(output,unsafe_allow_html=True)


test_image = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

if test_image:
    with st.container(border = True):
        image = Image.open(test_image)
        resized_image = image.resize((460, 460))
        
        img_byte_arr = io.BytesIO()
        resized_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        st.image(img_byte_arr, width=150)

        if st.button("Predict"):
            with st.spinner("Predicting Image...."):
                time.sleep(2)
            
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
                    temp_image.write(test_image.getbuffer())
                    temp_image_path = temp_image.name

                result_index = model_prediction(temp_image_path)

                predicted_output = class_name[result_index]
                name_of_disease = ''.join(
                    [
                        item 
                            for item in predicted_output.split('__')
                    ]
                ).replace('_',' ').strip()
                
                st.success(f"Model Prediction: **{name_of_disease}**")
                
                os.remove(temp_image_path)

                with st.spinner(text="Generating AI Response"):
                    generate_ai_results(name_of_disease)