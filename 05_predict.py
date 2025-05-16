#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import joblib
import numpy as np
import pandas as pd
import mlflow
from tensorflow.keras.models import load_model

station_id = sys.argv[1]
data_dir = "output"
model_path = os.path.join(data_dir, "models", f"{station_id}_model.keras")
scaler_dir = os.path.join(data_dir, "scalers")
feature_scaler_path = os.path.join(scaler_dir, f"scaler_{station_id}.pkl")
wvht_scaler_path = os.path.join(scaler_dir, f"wvht_scaler_{station_id}.pkl")
feature_cols_path = os.path.join(scaler_dir, f"feature_cols_{station_id}.pkl")
input_path = os.path.join(data_dir, station_id, "last_24_input.csv")
future_path = os.path.join(data_dir, station_id, "future_features.csv")
output_path = os.path.join(data_dir, station_id, "forecast_output.csv")

mlflow.set_experiment("surf_forecast")
with mlflow.start_run(run_name=f"predict_{station_id}"):
    df_input = pd.read_csv(input_path, parse_dates=["datetime"])
    df_future = pd.read_csv(future_path, parse_dates=["datetime"])

    feature_cols = joblib.load(feature_cols_path)
    scaler = joblib.load(feature_scaler_path)
    wvht_scaler = joblib.load(wvht_scaler_path)
    model = load_model(model_path)

    for col in feature_cols:
        if col not in df_future.columns:
            df_future[col] = df_input[col].mean()

    last_seq = df_input[feature_cols].values[-24:]
    scaled_seq = scaler.transform(last_seq)

    predictions = []
    timestamps = []

    for i in range(len(df_future)):
        input_seq = np.array([scaled_seq])
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred_scaled)
        timestamps.append(df_future.iloc[i]["datetime"])

        next_row = df_future[feature_cols].iloc[i].values
        next_scaled = scaler.transform([next_row])[0]
        scaled_seq = np.vstack([scaled_seq[1:], next_scaled])

    wvht_values = wvht_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    df_out = pd.DataFrame({
        "Zeitpunkt": timestamps,
        "Wellenhöhe (m)": wvht_values,
    })

    def tip(val):
        if val < 0.5: return "Flach"
        elif val < 1.0: return "Anfänger"
        elif val < 2.5: return "Fortgeschrittene"
        else: return "Nur Profis"
    df_out["Surf-Tipp"] = df_out["Wellenhöhe (m)"].apply(tip)

    df_out.to_csv(output_path, index=False)
    print("Forecast gespeichert in:", output_path)

    mlflow.log_param("station_id", station_id)
    mlflow.log_param("step", "predict")
    mlflow.log_metric("forecast_count", len(df_out))
    mlflow.log_artifact(output_path)

