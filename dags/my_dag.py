import json
import os
import pickle
from datetime import timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array, preprocess_input

from features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from models.train_model import TextLSTMModel, ImageVGG16Model, concatenate


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}



def ingest_data(**kwargs):
    base_path = "/opt/airflow/dags"
    raw_data_path = os.path.join(base_path, "data/raw/X_train_update.csv")
    processed_data_path = os.path.join(base_path, "data/preprocessed/X_train_update.csv")

    raw_data = pd.read_csv(raw_data_path)
    raw_data['description'] = raw_data['description'].str.lower().fillna('')
    raw_data.to_csv(processed_data_path, index=False)

    print("Data ingestion and preprocessing complete.")

def train_models(**kwargs):
    base_path = "/opt/airflow/dags"
    data_importer = DataImporter(base_path)
    df = data_importer.load_data()
    X_train, X_val, _, y_train, y_val, _ = data_importer.split_train_test(df)

    text_preprocessor = TextPreprocessor()
    image_preprocessor = ImagePreprocessor(base_path=base_path)

    text_preprocessor.preprocess_text_in_df(X_train, columns=["description"])
    text_preprocessor.preprocess_text_in_df(X_val, columns=["description"])
    image_preprocessor.preprocess_images_in_df(X_train)
    image_preprocessor.preprocess_images_in_df(X_val)

    print("Training LSTM Model")
    text_lstm_model = TextLSTMModel()
    text_lstm_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
    text_lstm_model.save(os.path.join(base_path, "models/best_lstm_model.keras"))

    print("Training VGG16 Model")
    image_vgg16_model = ImageVGG16Model()
    image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
    image_vgg16_model.save(os.path.join(base_path, "models/best_vgg16_model.keras"))

    with open(os.path.join(base_path, "models/tokenizer_config.json"), "r", encoding="utf-8") as file:
        tokenizer_config = file.read()
    tokenizer = tokenizer_from_json(tokenizer_config)
    lstm_model = tf.keras.models.load_model(os.path.join(base_path, "models/best_lstm_model.keras"))
    vgg16_model = tf.keras.models.load_model(os.path.join(base_path, "models/best_vgg16_model.keras"))

    print("Training concatenate model")
    model_concatenate = concatenate(tokenizer, lstm_model, vgg16_model)
    lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train)
    best_weights = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)

    with open(os.path.join(base_path, "models/best_weights.pkl"), "wb") as file:
        pickle.dump(best_weights, file)

    num_classes = 27
    proba_lstm = tf.keras.layers.Input(shape=(num_classes,))
    proba_vgg16 = tf.keras.layers.Input(shape=(num_classes,))

    weighted_proba = tf.keras.layers.Lambda(
        lambda x: best_weights[0] * x[0] + best_weights[1] * x[1]
    )([proba_lstm, proba_vgg16])

    concatenate_model = tf.keras.models.Model(
        inputs=[proba_lstm, proba_vgg16], outputs=weighted_proba
    )

    concatenate_model.save(os.path.join(base_path, "models/concatenate.h5"))

def predict_models(**kwargs):
    base_path = "/opt/airflow/dags"
    file_path = os.path.join(base_path, "data/preprocessed/X_test_update.csv")
    image_path = os.path.join(base_path, "data/raw/image_test")

    class Predictor:
        def __init__(self, tokenizer, lstm, vgg16, best_weights, mapper, filepath, imagepath):
            self.tokenizer = tokenizer
            self.lstm = lstm
            self.vgg16 = vgg16
            self.best_weights = best_weights
            self.mapper = mapper
            self.filepath = filepath
            self.imagepath = imagepath

        def preprocess_image(self, image_path, target_size):
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            return img_array

        def predict(self):
            X = pd.read_csv(self.filepath)[:10]
            text_preprocessor = TextPreprocessor()
            image_preprocessor = ImagePreprocessor(base_path="/opt/airflow/dags", image_subpath="data/raw/image_test")
            text_preprocessor.preprocess_text_in_df(X, columns=["description"])
            image_preprocessor.preprocess_images_in_df(X)

            sequences = self.tokenizer.texts_to_sequences(X["description"])
            padded_sequences = pad_sequences(sequences, maxlen=10, padding="post", truncating="post")

            target_size = (224, 224, 3)
            images = X["image_path"].apply(lambda x: self.preprocess_image(x, target_size))
            images = tf.convert_to_tensor(images.tolist(), dtype=tf.float32)

            lstm_proba = self.lstm.predict([padded_sequences])
            vgg16_proba = self.vgg16.predict([images])

            concatenate_proba = (
                self.best_weights[0] * lstm_proba + self.best_weights[1] * vgg16_proba
            )
            final_predictions = np.argmax(concatenate_proba, axis=1)

            return {
                i: self.mapper[str(final_predictions[i])]
                for i in range(len(final_predictions))
            }

    with open(os.path.join(base_path, "models/tokenizer_config.json"), "r", encoding="utf-8") as file:
        tokenizer_config = file.read()
    tokenizer = tokenizer_from_json(tokenizer_config)

    lstm_model = tf.keras.models.load_model(os.path.join(base_path, "models/best_lstm_model.keras"))
    vgg16_model = tf.keras.models.load_model(os.path.join(base_path, "models/best_vgg16_model.keras"))

    with open(os.path.join(base_path, "models/best_weights.pkl"), "rb") as file:
        best_weights = pickle.load(file)

    with open(os.path.join(base_path, "models/mapper.json"), "r") as file:
        mapper = json.load(file)

    predictor = Predictor(
        tokenizer=tokenizer,
        lstm=lstm_model,
        vgg16=vgg16_model,
        best_weights=best_weights,
        mapper=mapper,
        filepath=file_path,
        imagepath=image_path,
    )

    predictions = predictor.predict()

    with open(os.path.join(base_path, "data/preprocessed/predictions.json"), "w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=2)

def monitor_models(**kwargs):
    base_path = "/opt/airflow/dags/"
    evaluation_data_path = os.path.join(base_path, "data/preprocessed/X_test_update.csv")

    lstm_model = tf.keras.models.load_model(os.path.join(base_path, "models/best_lstm_model.keras"))
    vgg16_model = tf.keras.models.load_model(os.path.join(base_path, "models/best_vgg16_model.keras"))

    with open(os.path.join(base_path, "models/best_weights.pkl"), "rb") as file:
        best_weights = pickle.load(file)

    evaluation_data = pd.read_csv(evaluation_data_path)

    if 'description' not in evaluation_data.columns or 'imageid' not in evaluation_data.columns:
        raise ValueError("Required columns are missing in the evaluation data.")

    X_text = evaluation_data["description"]
    X_images = evaluation_data["imageid"]
    y_true = evaluation_data["productid"]

    text_preprocessor = TextPreprocessor()
    image_preprocessor = ImagePreprocessor(base_path=base_path, image_subpath="data/raw/image_test")

    X_text_df = pd.DataFrame({'description': X_text})
    text_preprocessor.preprocess_text_in_df(X_text_df, columns=["description"])

    image_preprocessor.preprocess_images_in_df(pd.DataFrame({"imageid": X_images}))

    sequences = text_preprocessor.tokenizer.texts_to_sequences(X_text)
    padded_sequences = pad_sequences(sequences, maxlen=10, padding="post", truncating="post")

    target_size = (224, 224, 3)
    images = X_images.apply(lambda x: image_preprocessor.preprocess_image(os.path.join(base_path, "data/raw/image_test", x), target_size))
    images = tf.convert_to_tensor(images.tolist(), dtype=tf.float32)

    lstm_proba = lstm_model.predict([padded_sequences])
    vgg16_proba = vgg16_model.predict([images])

    concatenate_proba = (
        best_weights[0] * lstm_proba + best_weights[1] * vgg16_proba
    )
    predictions = np.argmax(concatenate_proba, axis=1)

    accuracy = accuracy_score(y_true, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")


dag = DAG(
    'ml_pipeline_evently',
    default_args=default_args,
    description='A DAG to train and evaluate ML models with Evently AI',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
)


ingest_task = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag,
)

predict_task = PythonOperator(
    task_id='predict_models',
    python_callable=predict_models,
    dag=dag,
)

monitor_task = PythonOperator(
    task_id='monitor_models',
    python_callable=monitor_models,
    dag=dag,
)


ingest_task >> train_task >> predict_task >> monitor_task
