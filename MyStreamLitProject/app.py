import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE

# Load the dataset (replace this with the actual path to your file)
@st.cache_data
def load_data():
    file_path = r"C:\Users\NAMAN\Downloads\AssignmentData.xlsx"
    transactions = pd.read_excel(file_path, sheet_name='creditcard')  # Specify the correct sheet name
    transactions.columns = transactions.columns.str.strip()
    for col in transactions.select_dtypes(include=['object']).columns:
        transactions[col] = LabelEncoder().fit_transform(transactions[col].astype(str))
    transactions.dropna(inplace=True)
    return transactions

transactions = load_data()

# Display the dataset overview
st.title("Fraud Detection Using Isolation Forest")
st.write("This app uses an Isolation Forest model to detect fraudulent transactions.")

if st.checkbox("Show raw data"):
    st.write(transactions)

# Split data into features and target
X = transactions.drop(columns=['Class'])
y = transactions['Class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_pca, y)

# Train Isolation Forest model
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_resampled)

# Input features for prediction
st.sidebar.header("Input Transaction Data")
input_data = {}

for i, col in enumerate(X.columns):
    input_data[col] = st.sidebar.number_input(f"Enter {col}", value=float(X.iloc[0, i]))

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Scale and apply PCA to the input data
input_scaled = scaler.transform(input_df)
input_pca = pca.transform(input_scaled)

# Predict using the Isolation Forest model
if st.button("Predict"):
    prediction = iso_forest.predict(input_pca)
    prediction = np.where(prediction == -1, "Fraud", "Not Fraud")
    st.write(f"The transaction is predicted as: **{prediction[0]}**")

# Display the PCA variance explained
if st.checkbox("Show PCA Variance Explained"):
    st.write(f"Variance explained by the selected components: {pca.explained_variance_ratio_.sum():.2f}")