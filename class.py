import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('dataset2.csv' , nrows=10000)

# Clean the data
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Feature and target separation
X = data.drop(columns=['Label', 'Timestamp'])
y = data['Label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode categorical variables if any
X = pd.get_dummies(X)

# Replace infs and handle missing
X.replace([np.inf, -np.inf], np.nan, inplace=True)
valid_mask = X.notnull().all(axis=1)
X = X[valid_mask]
y_encoded = y_encoded[valid_mask]

# Save original column names for feature importance
original_columns = X.columns

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_resampled, y_train_resampled)

# Evaluate
y_pred = classifier.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Feature importance analysis
importances = classifier.feature_importances_
importance_df = pd.DataFrame({
    'Feature': original_columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Print top 20 features
print("\nTop 20 Important Features:\n", importance_df.head(20))

# Plot top 15 features
plt.figure(figsize=(10,6))
sns.barplot(data=importance_df.head(15), x='Importance', y='Feature', palette='mako')
plt.title("Top 15 Important Features for Attack Detection")
plt.tight_layout()
plt.show()
