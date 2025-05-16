#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import json
import pandas as pd
import joblib
import mlflow
from merge_weather import merge_weather_with_noaa
from weather_fetch import fetch_weather_forecast
from datetime import datetime, timedelta

station_id = sys.argv[1]
data_dir = "output"
data_path = os.path.join(data_dir, f"train_{station_id}.csv")
scaler_path = os.path.join(data_dir, "scalers", f"scaler_{station_id}.pkl")

mlflow.set_experiment("surf_forecast")
with mlflow.start_run(run_name=f"forecast_input_{station_id}"):
    df = pd.read_csv(data_path, parse_dates=["datetime"])
    feature_cols = [col for col in df.columns if col not in ["datetime", "wvht"]]
    df_input = df.sort_values("datetime").iloc[-24:].copy()

    meta_file = os.path.join("output", "surf_stations_valid.json")
    with open(meta_file, "r") as f:
        stations = json.load(f)

    station_meta = next((v for v in stations.values() if v["id"] == station_id), None)
    if not station_meta:
        raise ValueError(f"Station {station_id} nicht in surf_stations_valid.json gefunden.")
    lat, lon = station_meta["lat"], station_meta["lon"]

    try:
        future_weather = fetch_weather_forecast(lat, lon, days=1)
        print(f"Wetterdaten Zukunft: {len(future_weather)} Einträge")
    except Exception as e:
        print(f"Wetterdaten konnten nicht geladen werden: {e}")
        future_weather = pd.DataFrame(columns=["datetime"])

    start_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    future_times = [start_time + timedelta(hours=i) for i in range(12)]
    future_base = pd.DataFrame({"datetime": future_times})
    future_df = pd.merge(future_base, future_weather, on="datetime", how="left")

    available_cols = [col for col in feature_cols if col in future_df.columns]
    if not available_cols or future_df[available_cols].dropna(how="all").empty:
        print("Wetterdaten komplett leer – fülle mit Mittelwerten der letzten 24h")
        mean_values = df_input[feature_cols].mean()
        for col in feature_cols:
            future_df[col] = mean_values.get(col, 0.0)

    future_df["wvht"] = float("nan")

    output_dir = os.path.join(data_dir, station_id)
    os.makedirs(output_dir, exist_ok=True)
    input_path = os.path.join(output_dir, "last_24_input.csv")
    future_path = os.path.join(output_dir, "future_features.csv")
    df_input.to_csv(input_path, index=False)
    future_df.to_csv(future_path, index=False)

    mlflow.log_param("station_id", station_id)
    mlflow.log_param("step", "forecast_input")
    mlflow.log_metric("future_rows", len(future_df))
    mlflow.log_artifact(input_path)
    mlflow.log_artifact(future_path)

    print("Inputdaten für Forecast gespeichert.")

