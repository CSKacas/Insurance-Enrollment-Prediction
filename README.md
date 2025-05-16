# Insurance Enrollment Prediction

This repository contains a machine learning pipeline to predict whether an employee will opt in to a new voluntary insurance product based on demographic and employment-related data. The project also includes a REST API to serve model predictions in real-time.

---

##  Project Structure

<pre> insurance_enrollment_prediction/ ├── data/ │ └── employee_data.csv # Raw synthetic employee dataset ├── notebooks/ │ ├── 01_data_preprocessing.ipynb # Notebook for preprocessing │ └── 02_model_training.ipynb # Notebook for training and evaluation ├── reports/ # Generated plots and reports │ ├── confusion_matrix.png │ ├── roc_curve.png │ └── report.pdf ├── models/ # Saved model and preprocessor │ ├── model.pkl │ └── preprocessor.pkl ├── src/ │ ├── data_processing.py # Data loading and preprocessing functions │ └── model.py # Model training, evaluation, and saving functions ├── app/ │ ├── main.py # FastAPI app serving the model │ └── requirements.txt # FastAPI-specific dependencies ├── requirements.txt # Full environment dependencies ├── report.md # Technical summary and findings └── README.md # Instructions and documentation </pre>



---

##  Setup & Installation

1. Clone the repo:
    ```bash
    git clone https://github.com/yourusername/insurance_enrollment_prediction.git
    cd insurance_enrollment_prediction
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. (Optional) For serving API, install FastAPI dependencies:
    ```bash
    pip install -r app/requirements.txt
    ```

---

##  Running the Project from Scratch

### 1. Data Preprocessing

- Open and run `notebooks/01_data_preprocessing.ipynb`.
- This notebook:
  - Loads the raw dataset (`employee_data.csv`)
  - Cleans and preprocesses data (encoding categorical variables, scaling numerical features)
  - Splits data into train and test sets
  - Saves the processed datasets for modeling

### 2. Model Training and Evaluation

- Open and run `notebooks/02_model_training.ipynb`.
- This notebook:
  - Loads preprocessed train/test data
  - Trains an XGBoost (or other) model on training data
  - Evaluates model using multiple metrics (accuracy, precision, recall, F1-score)
  - Generates and displays interactive Plotly visualizations:
    - Evaluation metrics bar chart
    - Confusion matrix heatmap
    - ROC curve
  - Saves plots as HTML and PNG in `reports/`
  - Saves the trained model and preprocessing pipeline as `.pkl` files in `models/`

- **Note:** This notebook also includes MLflow experiment tracking and hyperparameter tuning as optional features.

### 3. Serving Predictions with FastAPI

- Start the API server from project root directory:
    
    uvicorn app.main:app --reload
    

- The API exposes a POST endpoint `/predict` which accepts employee data in JSON format and returns enrollment probability.

- Example `curl` request:

    curl --location 'http://127.0.0.1:8000/predict' \
--header 'Content-Type: application/json' \
--data '{"age": 35, "gender": "Male", "marital_status": "Single", "salary": 70000, "employment_type": "Full-time", "region": "West", "has_dependents": "1", "tenure_years": 5}'

    

---

##  Code Overview

- **src/data_processing.py**
  - Functions to load raw CSV data
  - Data cleaning, feature engineering, and preprocessing steps
  - Train/test split generation

- **src/model.py**
  - Functions for training ML models (XGBoost by default)
  - Model evaluation functions returning key metrics
  - Model and preprocessor saving/loading utilities

- **notebooks/**
  - Interactive exploratory and step-by-step pipelines for preprocessing and modeling
  - Visualizations using Plotly for better insights

- **app/main.py**
  - FastAPI application serving the trained model for real-time predictions
  - Loads saved model and preprocessing pipeline
  - Accepts JSON payload, preprocesses input, returns predicted enrollment probability

---

##  Results & Visualizations

- The `reports/` folder contains saved charts for:
  - Model evaluation metrics
  - Confusion matrix heatmap
  - ROC curve

These can be used directly for presentations or included in project reports.

---

##  Additional Notes

- Hyperparameter tuning and experiment tracking are implemented via MLflow (see `02_model_training.ipynb`).
- Model and preprocessing pipeline are saved to avoid retraining on every run.
- The FastAPI app provides a convenient REST interface to integrate the model into other applications or dashboards.

Thank you 
