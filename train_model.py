import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Load dataset
df = pd.read_csv("data.csv")


# Feature and target selection
X = df[["ab_volume", "kVp"]]
y_snr = df["SNR"]
y_cnr = df["CNR"]

# Train-test split
X_train, X_test, y_snr_train, y_snr_test = train_test_split(X, y_snr, test_size=0.2, random_state=42)
_, _, y_cnr_train, y_cnr_test = train_test_split(X, y_cnr, test_size=0.2, random_state=42)

# Train models
snr_model = RandomForestRegressor(n_estimators=100, random_state=42)
snr_model.fit(X_train, y_snr_train)

cnr_model = RandomForestRegressor(n_estimators=100, random_state=42)
cnr_model.fit(X_train, y_cnr_train)

# Evaluate
snr_preds = snr_model.predict(X_test)
cnr_preds = cnr_model.predict(X_test)

print("SNR RMSE:", np.sqrt(mean_squared_error(y_snr_test, snr_preds)))
print("CNR RMSE:", np.sqrt(mean_squared_error(y_cnr_test, cnr_preds)))

joblib.dump(snr_model, "snr_model.pkl")
joblib.dump(cnr_model, "cnr_model.pkl")
print("SNR R²:", r2_score(y_snr_test, snr_preds))
print("CNR R²:", r2_score(y_cnr_test, cnr_preds))