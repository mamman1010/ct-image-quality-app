import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib  # For ML model

# Load dataset
df = pd.read_csv("data.csv")

# Load trained ML models
snr_model = joblib.load("snr_model.pkl")
cnr_model = joblib.load("cnr_model.pkl")

st.title("CT Image Quality Estimator")
st.markdown("""
This app displays actual and optional ML-predicted **Signal-to-Noise Ratio (SNR)** and **Contrast-to-Noise Ratio (CNR)** values 
from abdominal CT data based on **abdominal volume**.
""")

# Sidebar toggle for ML model
use_ml = st.sidebar.checkbox("Use ML Model for Prediction")

# User input
ab_volume = st.number_input("Enter abdominal volume (e.g., 10783):", min_value=0)

# Filter actual values
filtered = df[df['ab_volume'] == ab_volume]

# Display actual values
if not filtered.empty:
    actual_snr = filtered['snr'].values[0]
    actual_cnr = filtered['cnr'].values[0]
    st.subheader("Actual SNR and CNR")
    st.write(f"**SNR:** {actual_snr:.2f}")
    st.write(f"**CNR:** {actual_cnr:.2f}")
else:
    st.warning("No matching record found in dataset.")

# ML prediction
if use_ml and ab_volume:
    pred_input = np.array([[ab_volume]])
    pred_snr = snr_model.predict(pred_input)[0]
    pred_cnr = cnr_model.predict(pred_input)[0]
    st.subheader("Predicted SNR and CNR (ML Model)")
    st.write(f"**Predicted SNR:** {pred_snr:.2f}")
    st.write(f"**Predicted CNR:** {pred_cnr:.2f}")

# Plotting actual values
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(df['ab_volume'], df['snr'], color='blue', label='Actual SNR')
ax.scatter(df['ab_volume'], df['cnr'], color='green', label='Actual CNR')

# Optional: predicted values as red stars
if use_ml:
    ax.scatter(ab_volume, pred_snr, color='red', marker='*', s=150, label='Predicted SNR')
    ax.scatter(ab_volume, pred_cnr, color='orange', marker='*', s=150, label='Predicted CNR')

ax.set_xlabel("Abdominal Volume")
ax.set_ylabel("SNR / CNR")
ax.set_title("SNR and CNR vs Abdominal Volume")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.markdown("---")
st.caption("Developed by Mamman • Powered by Streamlit & Machine Learning")
