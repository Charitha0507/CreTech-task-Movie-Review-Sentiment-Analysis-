# app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Title
st.title("📊 Customer Churn Prediction for Telecom Company")

# Load data
data = pd.read_csv("telecom_churn.csv")
st.subheader("📄 First 5 rows of the dataset")
st.write(data.head())

# Basic EDA: Plot churn distribution
st.subheader("🔍 Churn Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(data=data, x='Churn', palette='Set2', ax=ax1)
st.pyplot(fig1)

# Preprocess
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(inplace=True)

# Encode categorical variables
data_encoded = pd.get_dummies(data.drop(['customerID'], axis=1), drop_first=True)

# Features and target
X = data_encoded.drop('Churn_Yes', axis=1)
y = data_encoded['Churn_Yes']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
st.subheader("✅ Accuracy:")
st.write(f"{accuracy:.2f}")

# Confusion Matrix
st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=ax2)
st.pyplot(fig2)

# Classification Report
st.subheader("📋 Classification Report")
st.text(classification_report(y_test, y_pred))








#python -m streamlit run prediction.py