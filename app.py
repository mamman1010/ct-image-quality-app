import pandas as pd
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white;}
    </style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()
df["ab_volume"] = df["ab_volume"].round().astype(int)
df["ab_circumference"] = df["ab_circumference"].round().astype(int)

# Load trained models
snr_model_volume = joblib.load("snr_model_volume.pkl")
cnr_model_volume = joblib.load("cnr_model_volume.pkl")
snr_model_circ = joblib.load("snr_model_circ.pkl")
cnr_model_circ = joblib.load("cnr_model_circ.pkl")

st.set_page_config(page_title="CT Quality Estimator", layout="centered")
st.title("🧠 CT Image Quality Estimator")

input_method = st.radio("📌 Select input method:", ["Abdominal Volume", "Abdominal Circumference"], help="Choose whether to enter abdominal volume in cm³ or circumference in cm")
kVp = st.selectbox("⚡ Select tube voltage (kVp):", options=[80, 120], help="Select the X-ray tube voltage used during the scan")

# Input field and filtering
if input_method == "Abdominal Volume":
    ab_volume = st.number_input("📏 Enter abdominal volume (rounded, in cm³):", min_value=0, step=1, help="This value should be derived from 3D Slicer segmentation between -50 and 200 HU")
    ab_volume_int = int(round(ab_volume))
    filtered = df[(df['ab_volume'] == ab_volume_int) & (df['kVp'] == kVp)]
elif input_method == "Abdominal Circumference":
    ab_circ = st.number_input("📏 Enter abdominal circumference (rounded, in cm):", min_value=0, step=1, help="Measured from a representative axial slice using the 3D Slicer ruler or segment statistics")
    ab_circ_int = int(round(ab_circ))
    filtered = df[(df['ab_circumference'] == ab_circ_int) & (df['kVp'] == kVp)]

# Show actual values
if not filtered.empty:
    actual_snr = filtered['snr'].values[0]
    actual_cnr = filtered['cnr'].values[0]
    st.success("✅ Matching data found!")
    st.subheader("📊 Actual SNR and CNR")
    st.write(f"**SNR:** {actual_snr:.2f}")
    st.write(f"**CNR:** {actual_cnr:.2f}")
else:
    st.warning("⚠️ No matching record found in dataset.")

# Use ML Prediction
if input_method == "Abdominal Volume":
    input_features = np.array([[ab_volume_int, kVp]], dtype=np.float32)
    pred_snr = snr_model_volume.predict(input_features)[0]
    pred_cnr = cnr_model_volume.predict(input_features)[0]
elif input_method == "Abdominal Circumference":
    input_features = np.array([[ab_circ_int, kVp]], dtype=np.float32)
    pred_snr = snr_model_circ.predict(input_features)[0]
    pred_cnr = cnr_model_circ.predict(input_features)[0]

st.subheader("🤖 Predicted SNR and CNR (ML Model)")
st.write(f"**Predicted SNR:** {pred_snr:.2f}")
st.write(f"**Predicted CNR:** {pred_cnr:.2f}")

# Plotting
fig, ax = plt.subplots(figsize=(8, 5))

x_val = ab_volume_int if input_method == "Abdominal Volume" else ab_circ_int
xlabel = "Abdominal Volume (cm³)" if input_method == "Abdominal Volume" else "Abdominal Circumference (mm)"

if not filtered.empty:
    ax.scatter(x_val, actual_snr, color='deepskyblue', label='Actual SNR', s=100, edgecolor='black')
    ax.scatter(x_val, actual_cnr, color='limegreen', label='Actual CNR', s=100, edgecolor='black')

ax.scatter(x_val, pred_snr, color='crimson', marker='*', s=200, label='Predicted SNR')
ax.scatter(x_val, pred_cnr, color='orange', marker='*', s=200, label='Predicted CNR')

ax.set_xlabel(xlabel, fontsize=12)
ax.set_ylabel("SNR / CNR", fontsize=12)
ax.set_title("📈 SNR and CNR at Selected Input", fontsize=14, fontweight='bold')
ax.legend(frameon=True, fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

st.pyplot(fig)

st.markdown("**Disclaimer**: This app provides SNR & CNR predictions for abdominal CT scans. Consult a radiologist for clinical decisions.")
st.caption("👨‍⚕️ Developed by Mamman • Powered by Streamlit & Machine Learning")
