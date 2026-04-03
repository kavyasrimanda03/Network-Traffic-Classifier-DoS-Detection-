import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
import joblib

# Load the dataset
data = pd.read_csv(r'C:\Network_classifier\dataset2.csv' , nrows=1000)  # No space after r

# Preprocessing
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

X = data.drop(columns=['Label', 'Timestamp'])
y = data['Label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X = pd.get_dummies(X)

X.replace([np.inf, -np.inf], np.nan, inplace=True)
valid_mask = X.notnull().all(axis=1)
X = X[valid_mask]
y_encoded = y_encoded[valid_mask]

original_columns = X.columns  # Save for later use

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_resampled, y_train_resampled)

# GUI Function
def predict_from_gui():
    try:
        # Collect user inputs
        user_input = {
            'Flow Duration': float(entry1.get()),
            'Total Fwd Packets': float(entry2.get()),
            'Total Backward Packets': float(entry3.get()),
            'Flow Bytes/s': float(entry4.get()),
            'Flow Packets/s': float(entry5.get()),
            'Fwd Packet Length Mean': float(entry6.get())
        }

        # Create full feature vector
        input_df = pd.DataFrame([np.zeros(len(original_columns))], columns=original_columns)

        # Fill in the user values
        for feature, value in user_input.items():
            if feature in input_df.columns:
                input_df.at[0, feature] = value

        input_scaled = scaler.transform(input_df)
        prediction = classifier.predict(input_scaled)
        label = label_encoder.inverse_transform(prediction)[0]

        messagebox.showinfo("Prediction Result", f"The network packet is likely **{label.upper()}**")

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numerical values.")

# GUI Layout
root = tk.Tk()
root.title("Network Packet Classifier")

tk.Label(root, text="Flow Duration").grid(row=0, column=0, padx=10, pady=5)
entry1 = tk.Entry(root)
entry1.grid(row=0, column=1)

tk.Label(root, text="Total Fwd Packets").grid(row=1, column=0, padx=10, pady=5)
entry2 = tk.Entry(root)
entry2.grid(row=1, column=1)

tk.Label(root, text="Total Backward Packets").grid(row=2, column=0, padx=10, pady=5)
entry3 = tk.Entry(root)
entry3.grid(row=2, column=1)

tk.Label(root, text="Flow Bytes/s").grid(row=3, column=0, padx=10, pady=5)
entry4 = tk.Entry(root)
entry4.grid(row=3, column=1)

tk.Label(root, text="Flow Packets/s").grid(row=4, column=0, padx=10, pady=5)
entry5 = tk.Entry(root)
entry5.grid(row=4, column=1)

tk.Label(root, text="Fwd Packet Length Mean").grid(row=5, column=0, padx=10, pady=5)
entry6 = tk.Entry(root)
entry6.grid(row=5, column=1)

predict_button = tk.Button(root, text="Predict", command=predict_from_gui, bg="blue", fg="white")
predict_button.grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()
