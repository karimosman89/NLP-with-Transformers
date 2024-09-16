# Source Code Directory

This directory contains the core source code for the project, including data processing, feature engineering, model training, and prediction.

## Directory Structure

src/
│
├── init.py # Makes src a Python module
├── main.py # Script for training models
├── predict.py # Script for making predictions
│
├── data/ # Scripts for data processing
│ ├── init.py
│ ├── check_structure.py
│ ├── import_raw_data.py
│ └── make_dataset.py
│
├── features/ # Scripts for feature engineering
│ ├── init.py
│ └── build_features.py
│
├── models/ # Scripts for model training
│ ├── init.py
│ └── train_model.py
│
└── visualization/ # Scripts for data visualization
├── init.py
└── visualize.py

## Key Files

- **`main.py`**: Trains the models using the dataset.
- **`predict.py`**: Makes predictions using the trained models.

## Basic Usage

1. **Data Processing**: Use `src/data/import_raw_data.py` to import raw data.
2. **Feature Engineering**: Use `src/features/build_features.py` to process and engineer features.
3. **Model Training**: Run `python src/main.py` to train the models.
4. **Prediction**: Use `python src/predict.py` to make predictions.

For more detailed instructions, please refer to the main `README.md` in the root of the project.
