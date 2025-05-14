import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib  # For ML model

# Load dataset
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces in column names

# Load trained ML models
snr_model = joblib.load("snr_model.pkl")
cnr_model = joblib.load("cnr_model.pkl")

st.title("CT Image Quality Estimator")
st.markdown("""
This app displays actual and optional ML-predicted **Signal-to-Noise Ratio (SNR)** and **Contrast-to-Noise Ratio (CNR)** values 
from abdominal CT data based on **abdominal volume** and **kVp**.
""")

# Sidebar toggle for ML model
use_ml = st.sidebar.checkbox("Use ML Model for Prediction")

# Tube voltage selection
kVp = st.selectbox("Select tube voltage (kVp):", options=[80, 120])

# Abdominal volume slider based on data range
min_vol = int(df['ab_volume'].min())
max_vol = int(df['ab_volume'].max())
ab_volume = st.slider("Select abdominal volume:", min_value=min_vol, max_value=max_vol, value=min_vol)
ab_volume_int = int(ab_volume)

# Filter actual value for exact match
exact_filtered = df[(df['ab_volume'] == ab_volume_int) & (df['kVp'] == kVp)]

# Filter nearby values for plotting (±200 cm³)
range_filtered = df[
    (df['ab_volume'] >= ab_volume_int - 200) &
    (df['ab_volume'] <= ab_volume_int + 200) &
    (df['kVp'] == kVp)
]

# Display actual exact values
if not exact_filtered.empty:
    actual_snr = exact_filtered['snr'].values[0]
    actual_cnr = exact_filtered['cnr'].values[0]
    st.subheader("Actual SNR and CNR at Selected Volume")
    st.write(f"**SNR:** {actual_snr:.2f}")
    st.write(f"**CNR:** {actual_cnr:.2f}")
else:
    st.warning("No exact matching record found at this abdominal volume.")

# ML prediction
pred_snr = None
pred_cnr = None
if use_ml:
    pred_input = np.array([[ab_volume_int, kVp]], dtype=np.float32)
    pred_snr = snr_model.predict(pred_input)[0]
    pred_cnr = cnr_model.predict(pred_input)[0]
    st.subheader("Predicted SNR and CNR (ML Model)")
    st.write(f"**Predicted SNR:** {pred_snr:.2f}")
    st.write(f"**Predicted CNR:** {pred_cnr:.2f}")

# Plotting
fig, ax = plt.subplots(figsize=(8, 5))

# Always plot nearby actual data if available
if not range_filtered.empty:
    ax.scatter(range_filtered['ab_volume'], range_filtered['snr'], color='blue', label='Actual SNR')
    ax.scatter(range_filtered['ab_volume'], range_filtered['cnr'], color='green', label='Actual CNR')
else:
    st.warning("No actual data found within ±200 cm³ of selected abdominal volume.")

# Plot predicted point if ML is enabled
if use_ml and pred_snr is not None and pred_cnr is not None:
    ax.scatter(ab_volume_int, pred_snr, color='red', marker='*', s=150, label='Predicted SNR')
    ax.scatter(ab_volume_int, pred_cnr, color='orange', marker='*', s=150, label='Predicted CNR')

# Reference vertical line for current slider value
ax.axvline(ab_volume_int, color='black', linestyle='--', label='Selected Volume')

ax.set_xlabel("Abdominal Volume (cm³)")
ax.set_ylabel("SNR / CNR")
ax.set_title("SNR and CNR vs Abdominal Volume")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.markdown("---")
st.caption("Developed by Mamman • Powered by Streamlit & Machine Learning")
