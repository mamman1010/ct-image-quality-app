import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("CT Image Quality Optimizer (V1.0)")
st.markdown("""
This app displays actual and optional ML-predicted **Signal-to-Noise Ratio (SNR)** and **Contrast-to-Noise Ratio (CNR)** values 
from abdominal CT data based on **abdominal volume** and **kVp**.
""")

# Load your data
df = pd.read_csv("data.csv")

# User inputs
kVp = st.selectbox("Tube Voltage (kVp)", [80, 100, 120])
roi = st.selectbox("Region of Interest", ["LIVER", "KIDNEY"])
ab_volume = st.slider("Abdominal Volume (cm³)", 5000, 15000, 30000, step=1)

# Approximate volume matching
def find_closest(volume, kvp, roi):
    roi = roi.upper()
    subset = df[(df["kVp"] == kvp) & (df["roi"].str.upper() == roi)]
    closest_row = subset.iloc[(subset["ab_volume"] - volume).abs().argsort()[:1]]
    if closest_row.empty:
        return None
    return closest_row.iloc[0]["SNR"], closest_row.iloc[0]["CNR"]

# Get estimated values
snr, cnr = find_closest(ab_volume, kVp, roi) or ("No data", "No data")

# Output
st.subheader("Estimated Image Quality")
st.write(f"**SNR:** {snr}")
st.write(f"**CNR:** {cnr}")

# Plot
st.subheader("SNR & CNR vs. Tube Voltage")
kvps = [80, 100, 120]
snr_vals = []
cnr_vals = []

for k in kvps:
    result = find_closest(ab_volume, k, roi)
    if result:
        snr_vals.append(result[0])
        cnr_vals.append(result[1])
    else:
        snr_vals.append(None)
        cnr_vals.append(None)

fig, ax = plt.subplots()
ax.plot(kvps, snr_vals, label="SNR", marker='o', color='skyblue')
ax.plot(kvps, cnr_vals, label="CNR", marker='s', color='salmon')

# Annotate only the selected kVp point
if isinstance(snr, (int, float)) and isinstance(cnr, (int, float)):
    ax.annotate(f"{snr}", (kVp, snr), textcoords="offset points", xytext=(0,10),
                ha='center', fontsize=9, color='blue')
    ax.annotate(f"{cnr}", (kVp, cnr), textcoords="offset points", xytext=(0,10),
                ha='center', fontsize=9, color='red')

ax.set_xlabel("Tube Voltage (kVp)")
ax.set_ylabel("Value")
ax.set_title(f"SNR & CNR vs. kVp for ROI: {roi.title()}")
ax.legend()
ax.grid(True)
st.pyplot(fig)
