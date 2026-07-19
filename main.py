from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np

import tensorflow as tf
import joblib
import shap
# this is the explainable ai


app = FastAPI()


# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load trained model
model = tf.keras.models.load_model(
    "models/fraud_model.keras"
)


# Load scaler
scaler = joblib.load(
    "models/scaler.pkl"
)


@app.get("/")
def home():
    return {
        "message":"Fraud Detection API Running"
    }



@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Read uploaded CSV

    df = pd.read_csv(file.file)


    # Keep original data
    original = df.copy()


    # Scale input

    X = scaler.transform(df)


    # Prediction

    probabilities = model.predict(X)


    results=[]


    # SHAP explainer

    background = X[:50]

    explainer = shap.DeepExplainer(
        model,
        background
    )


    shap_values = explainer.shap_values(X)


    for i in range(len(X)):


        probability = float(
            probabilities[i][0]
        )


        if probability >=0.5:
            prediction="Fraud"
        else:
            prediction="Genuine"



        # Get feature importance

        values = np.abs(
            shap_values[0][i]
        )


        top_features = np.argsort(
            values
        )[-3:]


        reasons = [
            df.columns[j]
            for j in top_features
        ]


        results.append({

            "transaction":i+1,

            "prediction":prediction,

            "confidence":
                round(
                    probability*100,
                    2
                ),

            "reasons":reasons
        })


    return {

        "results":results

    }