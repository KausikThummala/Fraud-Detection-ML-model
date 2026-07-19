import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Load dataset
#basically load the csv file into the memory
df = pd.read_csv("creditcard.csv")

# Features and labels
#drop the class column from the dataframe as to separate the features from the target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
#here we are splitting the data set in 80: 20 ratio
#where the first 80% of the dataset goes into training and the next 20 % goes into  testing
#the same 80 and 20 percent goes into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
#here we are using stratify to balance the fraud ratio between the training and testing

# Balance training data
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)

# Scale features
#as ml algorithms give preference to data which is large in size
#so scale all the feature columns separately as it gives a model a better chance to learn from features
#which have numerically very small data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(30,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Predict
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)
print(predictions[:10])