import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset (Assume CSV file)
df = pd.read_csv("credit_card_transactions.csv")

# Display first few rows to understand the structure of the data
print(df.head())

# Handle missing values (drop rows with missing values)
df.dropna(inplace=True)

# Encode categorical variables (if necessary, e.g., one-hot encoding)
df = pd.get_dummies(df, drop_first=True)

# Separate features and target variable
X = df.drop('Fraud', axis=1)  # Features (exclude target variable)
y = df['Fraud']  # Target variable (fraud label)

# Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Initialize the Gaussian Naïve Bayes model
nb_model = GaussianNB()

# Train the Naïve Bayes model using the resampled training data
nb_model.fit(X_train_res, y_train_res)

# Make predictions on the test set
y_pred = nb_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

