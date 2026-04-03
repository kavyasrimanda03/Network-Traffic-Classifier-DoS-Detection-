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

# Load and clean data
data = pd.read_csv('dataset2.csv' , nrows=100000)
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

original_columns = X.columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Resample using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_resampled, y_train_resampled)

# Evaluate on test set
y_pred = classifier.predict(X_test)

# 1. Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 2. Classification Report Bar Plot
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report[['precision', 'recall', 'f1-score']].iloc[:-1].plot(kind='bar', figsize=(10, 6))
plt.title("Precision, Recall & F1-Score per Class")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Feature importance
importances = classifier.feature_importances_
importance_df = pd.DataFrame({
    'Feature': original_columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Select top 6 important features
top_features = importance_df['Feature'].head(6).tolist()

# Display top features
print("Top 6 Features Used in GUI:", top_features)

# GUI function
def predict_from_gui():
    try:
        user_input = {}
        for feature, entry in zip(top_features, entry_widgets):
            user_input[feature] = float(entry.get())

        # Create dummy input matching full feature set
        input_df = pd.DataFrame([np.zeros(len(original_columns))], columns=original_columns)

        # Fill in user input
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

entry_widgets = []
for idx, feature in enumerate(top_features):
    tk.Label(root, text=feature).grid(row=idx, column=0, padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=idx, column=1)
    entry_widgets.append(entry)

tk.Button(root, text="Predict", command=predict_from_gui, bg="blue", fg="white")\
    .grid(row=len(top_features), column=0, columnspan=2, pady=10)

root.mainloop()
