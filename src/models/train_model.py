import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.utils import resample
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score
from tensorflow import keras
import pickle
import json


class TextLSTMModel:
    def __init__(self, max_words=10000, max_sequence_length=10):
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.model = None

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val):
        self.tokenizer.fit_on_texts(X_train["description"])

        tokenizer_config = self.tokenizer.to_json()
        with open("models/tokenizer_config.json", "w", encoding="utf-8") as json_file:
            json_file.write(tokenizer_config)

        train_sequences = self.tokenizer.texts_to_sequences(X_train["description"])
        train_padded_sequences = pad_sequences(
            train_sequences,
            maxlen=self.max_sequence_length,
            padding="post",
            truncating="post",
        )

        val_sequences = self.tokenizer.texts_to_sequences(X_val["description"])
        val_padded_sequences = pad_sequences(
            val_sequences,
            maxlen=self.max_sequence_length,
            padding="post",
            truncating="post",
        )

        text_input = Input(shape=(self.max_sequence_length,))
        embedding_layer = Embedding(input_dim=self.max_words, output_dim=128)(
            text_input
        )
        lstm_layer = LSTM(128)(embedding_layer)
        output = Dense(27, activation="softmax")(lstm_layer)

        self.model = Model(inputs=[text_input], outputs=output)

        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        lstm_callbacks = [
            ModelCheckpoint(
                filepath="models/best_lstm_model.keras", save_best_only=True
            ),  # Enregistre le meilleur modèle
            EarlyStopping(
                patience=3, restore_best_weights=True
            ),  # Arrête l'entraînement si la performance ne s'améliore pas
            TensorBoard(log_dir="logs"),  # Enregistre les journaux pour TensorBoard
        ]

        self.model.fit(
            [train_padded_sequences],
            tf.keras.utils.to_categorical(y_train, num_classes=27),
            epochs=1,
            batch_size=32,
            validation_data=(
                [val_padded_sequences],
                tf.keras.utils.to_categorical(y_val, num_classes=27),
            ),
            callbacks=lstm_callbacks,
        )


class ImageVGG16Model:
    def __init__(self):
        self.model = None

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val):
        # Paramètres
        batch_size = 32
        num_classes = 27

        df_train = pd.concat([X_train, y_train.astype(str)], axis=1)
        df_val = pd.concat([X_val, y_val.astype(str)], axis=1)

        # Créer un générateur d'images pour le set d'entraînement
        train_datagen = ImageDataGenerator()  # Normalisation des valeurs de pixel
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=df_train,
            x_col="image_path",
            y_col="prdtypecode",
            target_size=(224, 224),  # Adapter à la taille d'entrée de VGG16
            batch_size=batch_size,
            class_mode="categorical",  # Utilisez 'categorical' pour les entiers encodés en one-hot
            shuffle=True,
        )

        # Créer un générateur d'images pour le set de validation
        val_datagen = ImageDataGenerator()  # Normalisation des valeurs de pixel
        val_generator = val_datagen.flow_from_dataframe(
            dataframe=df_val,
            x_col="image_path",
            y_col="prdtypecode",
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,  # Pas de mélange pour le set de validation
        )

        image_input = Input(
            shape=(224, 224, 3)
        )  # Adjust input shape according to your images

        vgg16_base = VGG16(
            include_top=False, weights="imagenet", input_tensor=image_input
        )

        x = vgg16_base.output
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)  # Add some additional layers if needed
        output = Dense(num_classes, activation="softmax")(x)

        self.model = Model(inputs=vgg16_base.input, outputs=output)

        for layer in vgg16_base.layers:
            layer.trainable = False

        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        vgg_callbacks = [
            ModelCheckpoint(
                filepath="models/best_vgg16_model.keras", save_best_only=True
            ),  # Enregistre le meilleur modèle
            EarlyStopping(
                patience=3, restore_best_weights=True
            ),  # Arrête l'entraînement si la performance ne s'améliore pas
            TensorBoard(log_dir="logs"),  # Enregistre les journaux pour TensorBoard
        ]

        self.model.fit(
            train_generator,
            epochs=1,
            validation_data=val_generator,
            callbacks=vgg_callbacks,
        )


class concatenate:
    def __init__(self, tokenizer, lstm, vgg16):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16

    def preprocess_image(self, image_path, target_size):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array

    def predict(self, X_train, y_train, new_samples_per_class=50, max_sequence_length=10):
        num_classes = 27  # Adjust according to your actual number of classes

        new_X_train = pd.DataFrame(columns=X_train.columns)
        new_y_train = pd.Series(dtype=int)  # Initialize an empty Series for y_train

        # Resample for each class
        for class_label in range(num_classes):
            indices = np.where(y_train == class_label)[0]
            if len(indices) >= new_samples_per_class:
                sampled_indices = resample(
                    indices, n_samples=new_samples_per_class, replace=False, random_state=42
                )
                new_X_train = pd.concat([new_X_train, X_train.iloc[sampled_indices]])
                new_y_train = pd.concat([new_y_train, pd.Series(y_train[sampled_indices])])
            else:
                print(f"Not enough samples for class {class_label}")

        # Reset index for the new dataframes
        new_X_train = new_X_train.reset_index(drop=True)
        new_y_train = new_y_train.reset_index(drop=True)
        
        print(f"Resampled X_train shape: {new_X_train.shape}")
        print(f"Resampled y_train shape: {new_y_train.shape}")

        # Check if new_y_train has the correct shape before reshaping
        if len(new_y_train) != num_classes * new_samples_per_class:
            raise ValueError(f"Resampled y_train length is {len(new_y_train)}, expected {num_classes * new_samples_per_class}")

        # Convert to the correct shape
        new_y_train = new_y_train.values.reshape((num_classes * new_samples_per_class,)).astype("int")

        # Load the models
        tokenizer = self.tokenizer
        lstm_model = self.lstm
        vgg16_model = self.vgg16

        # Text preprocessing
        train_sequences = tokenizer.texts_to_sequences(new_X_train["description"])
        train_padded_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length, padding="post", truncating="post")

        # Image preprocessing
        target_size = (224, 224, 3)
        images_train = new_X_train["image_path"].apply(lambda x: self.preprocess_image(x, target_size))
        images_train = tf.convert_to_tensor(images_train.tolist(), dtype=tf.float32)

        # Model predictions
        lstm_proba = lstm_model.predict([train_padded_sequences])
        vgg16_proba = vgg16_model.predict([images_train])

        return lstm_proba, vgg16_proba, new_y_train

    def optimize(self, lstm_proba, vgg16_proba, y_train):
        best_weights = None
        best_accuracy = 0.0

        for lstm_weight in np.linspace(0, 1, 101):
            vgg16_weight = 1.0 - lstm_weight
            combined_predictions = (lstm_weight * lstm_proba) + (vgg16_weight * vgg16_proba)
            final_predictions = np.argmax(combined_predictions, axis=1)
            accuracy = accuracy_score(y_train, final_predictions)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = (lstm_weight, vgg16_weight)

        return best_weights