import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()  # Clean column names

# Round volume and circumference to 2 decimal places
df["volume"] = df["volume"].round().astype(int)
df["ab_circumference"] = df["ab_circumference"].round().astype(int)

# --- Volume-based model training ---
X_volume = df[["volume", "kVp"]]
y_snr = df["snr"]
y_cnr = df["cnr"]

Xv_train, Xv_test, y_snr_train, y_snr_test = train_test_split(X_volume, y_snr, test_size=0.2, random_state=42)
_, _, y_cnr_train, y_cnr_test = train_test_split(X_volume, y_cnr, test_size=0.2, random_state=42)

snr_model_volume = RandomForestRegressor(n_estimators=100, random_state=42)
snr_model_volume.fit(Xv_train, y_snr_train)

cnr_model_volume = RandomForestRegressor(n_estimators=100, random_state=42)
cnr_model_volume.fit(Xv_train, y_cnr_train)

print("Volume-based model evaluation:")
print("SNR RMSE:", np.sqrt(mean_squared_error(y_snr_test, snr_model_volume.predict(Xv_test))))
print("CNR RMSE:", np.sqrt(mean_squared_error(y_cnr_test, cnr_model_volume.predict(Xv_test))))
print("SNR R²:", r2_score(y_snr_test, snr_model_volume.predict(Xv_test)))
print("CNR R²:", r2_score(y_cnr_test, cnr_model_volume.predict(Xv_test)))

joblib.dump(snr_model_volume, "snr_model_volume.pkl")
joblib.dump(cnr_model_volume, "cnr_model_volume.pkl")

# --- Circumference-based model training ---
X_circ = df[["ab_circumference", "kVp"]]

Xc_train, Xc_test, y_snr_train, y_snr_test = train_test_split(X_circ, y_snr, test_size=0.2, random_state=42)
_, _, y_cnr_train, y_cnr_test = train_test_split(X_circ, y_cnr, test_size=0.2, random_state=42)

snr_model_circ = RandomForestRegressor(n_estimators=100, random_state=42)
snr_model_circ.fit(Xc_train, y_snr_train)

cnr_model_circ = RandomForestRegressor(n_estimators=100, random_state=42)
cnr_model_circ.fit(Xc_train, y_cnr_train)

print("Circumference-based model evaluation:")
print("SNR RMSE:", np.sqrt(mean_squared_error(y_snr_test, snr_model_circ.predict(Xc_test))))
print("CNR RMSE:", np.sqrt(mean_squared_error(y_cnr_test, cnr_model_circ.predict(Xc_test))))
print("SNR R²:", r2_score(y_snr_test, snr_model_circ.predict(Xc_test)))
print("CNR R²:", r2_score(y_cnr_test, cnr_model_circ.predict(Xc_test)))

joblib.dump(snr_model_circ, "snr_model_circ.pkl")
joblib.dump(cnr_model_circ, "cnr_model_circ.pkl")

print("All models trained and saved successfully.")

