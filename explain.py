import shap
import numpy as np
import tensorflow as tf
import joblib



# Load model

model = tf.keras.models.load_model(
    "models/fraud_model.keras"
)



# Load feature names

feature_names = joblib.load(
    "models/feature_names.pkl"
)




def explain_prediction(X):


    """
    Generates SHAP explanations
    """



    # Background data

    background = X[:50]



    explainer = shap.DeepExplainer(

        model,

        background

    )



    shap_values = explainer.shap_values(
        X
    )



    explanations = []



    for i in range(len(X)):


        # Get SHAP values
        values = np.abs(
            shap_values[0][i]
        )


        # Pick top 3 features

        top_indices = np.argsort(
            values
        )[-3:]



        reasons = []



        for index in top_indices:


            reasons.append({

                "feature":
                    feature_names[index],

                "impact":
                    round(
                        float(
                            shap_values[0][i][index]
                        ),
                        4
                    )

            })



        explanations.append(
            reasons
        )


    return explanations