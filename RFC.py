import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Sales Prediction", layout="centered")
st.title("📊 Sales Prediction using Random Forest")

# --- Load dataset ---
DATA_PATH = r"C:\Users\Marliya\Desktop\RF Classification\sales_data.csv"
df = pd.read_csv(DATA_PATH)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --- Select target ---
target = st.selectbox("Select Target Column", df.columns)

# Drop missing target rows
df = df.dropna(subset=[target])
X = df.drop(target, axis=1)
y = df[target]

# --- Encode categorical columns safely ---
encoding_maps = {}
for col in X.select_dtypes(include="object").columns:
    mapping = {val: idx for idx, val in enumerate(X[col].unique())}
    X[col] = X[col].map(mapping)
