import pandas as pd
import numpy as np
import streamlit as st
import os
import zipfile
import re
import json
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Constants
MAX_SEQUENCE_LENGTH = 10
IMAGE_SIZE = (224, 224)

# Define a function to load models safely
def load_model_safely(path):
    try:
        model = load_model(path, compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        st.stop()

# Load models and tokenizer
LSTM_MODEL_PATH = 'best_lstm_model.keras'
VGG16_MODEL_PATH = 'best_vgg16_model.keras'
CONCATENATE_MODEL_PATH = 'concatenate.keras'
TOKENIZER_PATH = 'tokenizer_config.json'
WEIGHTS_PATH = 'best_weights.pkl'

lstm_model = load_model_safely(LSTM_MODEL_PATH)
vgg16_model = load_model_safely(VGG16_MODEL_PATH)
concatenate_model = load_model_safely(CONCATENATE_MODEL_PATH)

try:
    with open(TOKENIZER_PATH, 'r') as json_file:
        tokenizer_config = json_file.read()
    tokenizer = tokenizer_from_json(tokenizer_config)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

try:
    with open(WEIGHTS_PATH, "rb") as file:
        best_weights = pickle.load(file)
except Exception as e:
    st.error(f"Error loading weights: {e}")
    st.stop()

# Initialize text processing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load CSV files
def load_csv_files():
    X_train = pd.read_csv('X_train_update.csv')
    Y_train = pd.read_csv('Y_train_CVw08PX.csv')
    X_test = pd.read_csv('X_test_update.csv')
    return X_train, Y_train, X_test

X_train, Y_train, X_test = load_csv_files()

# Load images from zip file
def extract_images(zip_path, extract_to='images'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

extract_images('images.zip')

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = word_tokenize(text.lower())
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words[:10])

# Function to preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


st.title('Model Prediction App')

model_choice = st.sidebar.selectbox('Select Model', ['Text LSTM model', 'VGG16 model', 'Concatenate Model'])

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write(input_data.head())

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image_array = preprocess_image(uploaded_image)

if model_choice == 'Text LSTM model':
    if uploaded_file is not None:
        preprocessed_text = preprocess_text(input_data['description'].fillna('').values[0])
        text_sequence = tokenizer.texts_to_sequences([preprocessed_text])
        text_padded_sequence = pad_sequences(text_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        prediction = lstm_model.predict(text_padded_sequence)
        st.write(f'Prediction for Text LSTM model: {np.argmax(prediction)}')

elif model_choice == 'VGG16 model':
    if uploaded_image is not None:
        prediction = vgg16_model.predict(image_array)
        st.write(f'Prediction for VGG16 model: {np.argmax(prediction)}')

elif model_choice == 'Concatenate Model':
    if uploaded_file is not None and uploaded_image is not None:
        preprocessed_text = preprocess_text(input_data['description'].fillna('').values[0])
        text_sequence = tokenizer.texts_to_sequences([preprocessed_text])
        text_padded_sequence = pad_sequences(text_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        text_proba = lstm_model.predict(text_padded_sequence)
        
        image_array = preprocess_image(uploaded_image)
        image_proba = vgg16_model.predict(image_array)

        combined_proba = best_weights[0] * text_proba + best_weights[1] * image_proba
        prediction = np.argmax(combined_proba)
        st.write(f'Prediction for Concatenate model: {prediction}')
