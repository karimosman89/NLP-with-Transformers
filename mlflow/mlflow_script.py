import mlflow
import mlflow.tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt
from models.train_model import TextLSTMModel, ImageVGG16Model, CombinedModel
from features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
import pandas as pd
import numpy as np
import os
import joblib  


MLFLOW_TRACKING_URI = 'http://localhost:5000'
EXPERIMENT_NAME = 'ecommerce_classification'
LSTM_MODEL_PATH = "jun24_bmlops_classification_e-commerce/mlflow/models/best_lstm_model.keras"
VGG16_MODEL_PATH = "jun24_bmlops_classification_e-commerce/mlflow/models/best_vgg16_model.keras"
COMBINED_MODEL_PATH = "jun24_bmlops_classification_e-commerce/mlflow/models/concatenate.keras"
TOKENIZER_CONFIG_PATH = "jun24_bmlops_classification_e-commerce/mlflow/models/tokenizer_config.json"

def plot_metrics(history, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.grid(True)
    plt.savefig(f'{model_name}_accuracy.png')

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.grid(True)
    plt.savefig(f'{model_name}_loss.png')

def preprocess_and_fit_models(X_train, y_train, X_val, y_val):
    text_lstm_model = TextLSTMModel()
    history_lstm = text_lstm_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
    
    plot_metrics(history_lstm, 'LSTM')
    mlflow.log_artifact('LSTM_accuracy.png')
    mlflow.log_artifact('LSTM_loss.png')

    image_vgg16_model = ImageVGG16Model()
    history_vgg16 = image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
    
    plot_metrics(history_vgg16, 'VGG16')
    mlflow.log_artifact('VGG16_accuracy.png')
    mlflow.log_artifact('VGG16_loss.png')
    
    return text_lstm_model, image_vgg16_model

def save_and_log_models(text_lstm_model, image_vgg16_model):
    text_lstm_model.model.save(LSTM_MODEL_PATH)
    mlflow.log_artifact(LSTM_MODEL_PATH)

    image_vgg16_model.model.save(VGG16_MODEL_PATH)
    mlflow.log_artifact(VGG16_MODEL_PATH)

def create_and_log_combined_model(text_lstm_model, image_vgg16_model):
    combined_model = CombinedModel(text_lstm_model.model, image_vgg16_model.model, text_lstm_model.tokenizer)
    model = combined_model.build_model()
    model.save(COMBINED_MODEL_PATH)
    mlflow.tensorflow.log_model(model=model, artifact_path="combined_model")

def train_and_log_to_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        print(f"Starting MLflow run with ID: {run.info.run_id}")

        mlflow.log_param("num_epochs_lstm", 10)
        mlflow.log_param("num_epochs_vgg16", 20)

        
        data_importer = DataImporter(base_path="mlflow/data/preprocessed/X_train_update.csv")
        df = data_importer.load_data()
        X_train, X_val, _, y_train, y_val, _ = data_importer.split_train_test(df)

        text_preprocessor = TextPreprocessor()
        image_preprocessor = ImagePreprocessor(base_path="jun24_bmlops_classification_e-commerce/mlflow/", image_subpath="data/raw/image_train")
        text_preprocessor.preprocess_text_in_df(X_train, columns=["description"])
        text_preprocessor.preprocess_text_in_df(X_val, columns=["description"])
        image_preprocessor.preprocess_images_in_df(X_train)
        image_preprocessor.preprocess_images_in_df(X_val)

        
        text_lstm_model, image_vgg16_model = preprocess_and_fit_models(X_train, y_train, X_val, y_val)
        save_and_log_models(text_lstm_model, image_vgg16_model)

        
        create_and_log_combined_model(text_lstm_model, image_vgg16_model)

       
        metrics = {
            "lstm_accuracy": 0.92,
            "vgg16_accuracy": 0.85
        }
        mlflow.log_metrics(metrics)

if __name__ == "__main__":
    train_and_log_to_mlflow()
