import tensorflow as tf
import joblib
import pandas as pd
import numpy as np



# ==========================
# Load saved model
# ==========================

model = tf.keras.models.load_model(
    "models/fraud_model.keras"
)



# ==========================
# Load scaler
# ==========================

scaler = joblib.load(
    "models/scaler.pkl"
)



# ==========================
# Load feature names
# ==========================

feature_names = joblib.load(
    "models/feature_names.pkl"
)




def predict_transactions(df):

    """
    Takes dataframe of new transactions
    and returns predictions
    """



    # --------------------------------
    # Keep only required features
    # --------------------------------

    df = df[feature_names]



    # --------------------------------
    # Scale data
    # --------------------------------

    X = scaler.transform(df)



    # --------------------------------
    # Model prediction
    # --------------------------------

    probabilities = model.predict(X)



    results = []


    for i, probability in enumerate(probabilities):


        probability = float(
            probability[0]
        )


        if probability >= 0.5:

            label = "Fraud"

        else:

            label = "Genuine"



        results.append({

            "transaction": i + 1,

            "prediction": label,

            "probability": round(
                probability,
                4
            ),

            "confidence": round(
                probability * 100,
                2
            )

        })



    return results