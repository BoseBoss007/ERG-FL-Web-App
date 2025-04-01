import requests
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# ==============================
# STEP 1: PROCESS INPUT ARGUMENTS
# ==============================

if len(sys.argv) != 3:
    print("Usage: python fl_client.py <server_url> <file_path>")
    sys.exit(1)

server_url = sys.argv[1]  # Server URL from app.py
file_path = sys.argv[2]   # CSV file path from user upload

# ============================
# STEP 2: CREATE TARGET COLUMN
# ============================

def assign_disorder(data):
    """
    Assigns a mental disorder label based on ERG conditions.
    Includes "Healthy" as a target label.
    """
    conditions = [
        (data["b_amp"] < 50) & (data["OP_s_Amp"] < 45),  # Seasonal Affective Disorder (SAD)
        (data["a_amp"] < -40) & (data["b_amp"] > 60),    # Schizophrenia (SZ)
        (data["b_amp"] > 70) & (data["OP_s_Time"] > 150), # Bipolar Disorder (BD)
        (data["b_amp"] > 75) & (data["OP_s_Amp"] < 65),  # Autism Spectrum Disorder (ASD)
        (data["OP_s_Amp"] < 40)                          # Drug Addiction
    ]
    disorders = ["SAD", "SZ", "BD", "ASD", "DA"]

    for i, condition in enumerate(conditions):
        data.loc[condition, "mental_disorder"] = disorders[i]

    data["mental_disorder"].fillna("Healthy", inplace=True)
    return data

# ==============================
# STEP 3: PREPROCESS DATA
# ==============================

def preprocess_data(file_path):
    """
    Reads the dataset, assigns target labels, and preprocesses it for training.
    """
    data = pd.read_csv(file_path)
    
    data = assign_disorder(data)

    X = data.drop(columns=["patient_id", "mental_disorder"]).values
    y = data["mental_disorder"].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    return X, y_categorical, label_encoder, data

# ==============================
# STEP 4: BUILD CLIENT MODEL
# ==============================

def build_model(input_size=8, num_classes=5):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_size,)),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==============================
# STEP 5: TRAINING & FEDERATED UPDATE
# ==============================

X, y, label_encoder, original_data = preprocess_data(file_path)
model = build_model(input_size=8, num_classes=5)

accuracies = []

for epoch in range(50):
    print(f"Epoch {epoch+1}/50: Training...")
    history = model.fit(X, y, epochs=1, batch_size=16, verbose=1)
    accuracies.append(history.history['accuracy'][0])

    weights = model.get_weights()
    response = requests.post(
        f"{server_url}/update",
        json={"weights": json.dumps([w.tolist() for w in weights])}
    )

    if response.status_code == 200:
        global_weights = json.loads(response.json()["global_weights"])
        model.set_weights([np.array(w) for w in global_weights])

# ==============================
# STEP 6: SAVE RESULTS
# ==============================

plt.plot(range(1, 51), accuracies, label="Training Accuracy", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Over Epochs")
plt.legend()
plt.savefig("results/accuracy_graph.png")

predictions = model.predict(X)
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
original_data["Predicted_Disorder"] = predicted_labels

# Add disorder descriptions inside the CSV file
csv_file_path = "results/client1_predictions.csv"
with open(csv_file_path, "w") as f:
    f.write("# ASD: Autism Spectrum Disorder\n")
    f.write("# SZ: Schizophrenia\n")
    f.write("# BD: Bipolar Disorder\n")
    f.write("# SAD: Seasonal Affective Disorder\n")
    f.write("# DA: Drug Addiction\n")
    f.write("# Healthy: No diagnosed disorder\n")
    original_data.to_csv(f, index=False)

# Create training completion flag
with open("results/training_complete.txt", "w") as f:
    f.write("Training completed successfully.")

print("Training completed. Results saved.")
