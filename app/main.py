from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Load saved model and preprocessor
model = joblib.load(os.path.join("models", "model.pkl"))
preprocessor = joblib.load(os.path.join("models", "preprocessor.pkl"))


app = FastAPI()

class EmployeeData(BaseModel):
    age: int
    gender: str
    marital_status: str
    salary: float
    employment_type: str
    region: str
    has_dependents: str
    tenure_years: float

@app.post("/predict")
def predict(data: EmployeeData):
    df_input = pd.DataFrame([data.dict()])
    X_transformed = preprocessor.transform(df_input)
    prediction = model.predict(X_transformed)
    return {"enrolled": int(prediction[0])}