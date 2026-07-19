import pandas as pd
import numpy as np
import os
import joblib

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import RandomOverSampler

import matplotlib.pyplot as plt


# ===============================
# 1. Load Dataset
# ===============================

print("Loading dataset...")

df = pd.read_csv("creditcard.csv")


print("\nDataset Shape:")
print(df.shape)


print("\nFirst 5 rows:")
print(df.head())



# ===============================
# 2. Basic Data Checking
# ===============================

print("\nMissing Values:")
print(df.isnull().sum())


print("\nClass Distribution:")
print(df["Class"].value_counts())



# ===============================
# 3. Remove Duplicate Rows
# ===============================

duplicates = df.duplicated().sum()

print("\nDuplicate Rows:", duplicates)


df = df.drop_duplicates()



# ===============================
# 4. Separate Features and Labels
# ===============================

X = df.drop(
    "Class",
    axis=1
)


y = df["Class"]



# Save feature names
feature_names = X.columns.tolist()



# ===============================
# 5. Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



print("\nBefore Oversampling:")

print(y_train.value_counts())



# ===============================
# 6. Random Oversampling
# ===============================


ros = RandomOverSampler(
    random_state=42
)



X_train, y_train = ros.fit_resample(

    X_train,

    y_train

)



print("\nAfter Oversampling:")

print(y_train.value_counts())



# ===============================
# 7. Feature Scaling
# ===============================


scaler = StandardScaler()



X_train = scaler.fit_transform(
    X_train
)



X_test = scaler.transform(
    X_test
)



# ===============================
# 8. Save Scaler
# ===============================


os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(

    scaler,

    "models/scaler.pkl"

)



joblib.dump(

    feature_names,

    "models/feature_names.pkl"

)



print("\nScaler saved!")



# ===============================
# 9. Build Neural Network
# ===============================


model = tf.keras.Sequential([


    tf.keras.layers.Dense(

        32,

        activation="relu",

        input_shape=(X_train.shape[1],)

    ),



    tf.keras.layers.Dense(

        16,

        activation="relu"

    ),



    tf.keras.layers.Dense(

        1,

        activation="sigmoid"

    )

])



model.summary()



# ===============================
# 10. Compile Model
# ===============================


model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[

        "accuracy"

    ]

)



# ===============================
# 11. Train Model
# ===============================


history = model.fit(

    X_train,

    y_train,

    epochs=10,

    batch_size=64,

    validation_split=0.2

)



# ===============================
# 12. Evaluate Model
# ===============================


print("\nEvaluating Model...")


loss, accuracy = model.evaluate(

    X_test,

    y_test

)


print(
    "\nTest Accuracy:",
    accuracy
)



# Probability predictions

y_prob = model.predict(

    X_test

)



# Convert probability to classes

y_pred = (

    y_prob > 0.5

).astype(int)



print("\nClassification Report:")

print(

    classification_report(

        y_test,

        y_pred

    )

)



# ===============================
# 13. Confusion Matrix
# ===============================


cm = confusion_matrix(

    y_test,

    y_pred

)


print("\nConfusion Matrix:")

print(cm)



plt.figure(figsize=(5,5))


plt.imshow(cm)


plt.title(
    "Confusion Matrix"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.colorbar()


plt.show()



# ===============================
# 14. ROC AUC Score
# ===============================


auc = roc_auc_score(

    y_test,

    y_prob

)


print(

    "\nROC AUC:",

    auc

)



fpr, tpr, thresholds = roc_curve(

    y_test,

    y_prob

)



plt.figure(figsize=(6,5))


plt.plot(

    fpr,

    tpr

)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(

    "ROC Curve"

)


plt.show()



# ===============================
# 15. Save Model
# ===============================


model.save(

    "models/fraud_model.keras"

)



print("\nModel Saved Successfully!")

print("\nFiles created:")

print(
"""
models/

├── fraud_model.keras

├── scaler.pkl

└── feature_names.pkl

"""
)