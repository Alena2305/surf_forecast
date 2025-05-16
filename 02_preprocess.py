#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import sys
import json
import time
import pandas as pd
import numpy as np
import joblib
import mlflow
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from merge_weather import merge_weather_with_noaa

def deduplicate_columns(columns):
    counts = {}
    new_cols = []
    for col in columns:
        if col not in counts:
            counts[col] = 0
            new_cols.append(col)
        else:
            counts[col] += 1
            new_cols.append(f"{col}.{counts[col]}")
    return new_cols

def load_station_metadata(station_id, meta_path="output/surf_stations_valid.json"):
    with open(meta_path, "r") as f:
        stations = json.load(f)
    for info in stations.values():
        if info["id"] == station_id:
            return info["lat"], info["lon"]
    raise ValueError(f"Koordinaten für Station {station_id} nicht gefunden.")

def preprocess_data(data_dir, station_id, years, output_dir="output"):
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "surf_forecast")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"preprocess_{station_id}"):
        start_time = time.time()

        file_paths = [os.path.join(data_dir, f"{station_id}_{year}.txt") for year in years]
        file_paths = [path for path in file_paths if os.path.exists(path)]
        if not file_paths:
            raise FileNotFoundError(f"Keine gültigen NOAA-Dateien gefunden für {station_id}.")

        dfs = [pd.read_csv(path, sep='\s+', skiprows=[1]) for path in file_paths]
        df = pd.concat(dfs)

        df.columns = deduplicate_columns(df.columns)
        df.columns = [col.lower() for col in df.columns]

        time_cols = ['#yy', 'mm', 'dd', 'hh']
        has_minute = 'minute' in df.columns
        rename_cols = {'#yy': 'year', 'mm': 'month', 'dd': 'day', 'hh': 'hour'}
        if has_minute:
            time_cols.append('minute')
            rename_cols['minute'] = 'minute'

        time_df = df[time_cols].copy().rename(columns=rename_cols)
        time_df = time_df.loc[:, ~time_df.columns.duplicated()]
        if has_minute:
            df['datetime'] = pd.to_datetime(time_df[['year', 'month', 'day', 'hour', 'minute']])
        else:
            df['datetime'] = pd.to_datetime(time_df[['year', 'month', 'day', 'hour']])

        df = df.set_index('datetime')
        df = df.drop(columns=time_cols, errors='ignore')

        invalid_vals = [99.0, 999.0, 9999.0]
        df = df.replace(invalid_vals, np.nan)

        last_train_year = max(years)
        train_df = df[df.index.year < last_train_year].copy()
        test_df = df[df.index.year == last_train_year].copy()

        train_df = train_df.dropna(subset=["wvht"])
        test_df = test_df.dropna(subset=["wvht"])

        numeric_cols = train_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        train_df[numeric_cols] = train_df[numeric_cols].ffill().bfill()
        test_df[numeric_cols] = test_df[numeric_cols].ffill().bfill()

        train_df = train_df.dropna(axis=1, how="all")
        test_df = test_df.dropna(axis=1, how="all")

        nunique = train_df.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        train_df = train_df.drop(columns=constant_cols)
        test_df = test_df.drop(columns=[col for col in constant_cols if col in test_df.columns])

        lat, lon = load_station_metadata(station_id)
        train_df = train_df.reset_index()
        test_df = test_df.reset_index()

        train_df = merge_weather_with_noaa(train_df, lat, lon, days=365)
        test_df = merge_weather_with_noaa(test_df, lat, lon, days=30)

        all_features = [col for col in train_df.columns if col not in ["datetime", "wvht"]]

        for col in all_features:
            if col not in train_df.columns:
                print(f"Feature '{col}' fehlt im Trainingssatz – wird mit Mittelwert ersetzt.")
                train_df[col] = train_df[all_features].mean().get(col, 0.0)
            if col not in test_df.columns:
                print(f"Feature '{col}' fehlt im Testsatz – wird mit Mittelwert ersetzt.")
                test_df[col] = train_df[col].mean()

        if train_df.empty or len(train_df[all_features]) == 0:
            raise ValueError("Trainingsdaten fehlerhaft oder leer – Abbruch.")

        scaler = StandardScaler()
        train_df[all_features] = scaler.fit_transform(train_df[all_features])
        test_df[all_features] = scaler.transform(test_df[all_features])

        scaler_path = os.path.join(output_dir, "scalers")
        os.makedirs(scaler_path, exist_ok=True)
        scaler_file = os.path.join(scaler_path, f"scaler_{station_id}.pkl")
        joblib.dump(scaler, scaler_file)

        wvht_scaler = MinMaxScaler()
        train_df["wvht"] = wvht_scaler.fit_transform(train_df[["wvht"]])
        test_df["wvht"] = wvht_scaler.transform(test_df[["wvht"]])
        wvht_scaler_file = os.path.join(scaler_path, f"wvht_scaler_{station_id}.pkl")
        joblib.dump(wvht_scaler, wvht_scaler_file)

        train_path = os.path.join(output_dir, f"train_{station_id}.csv")
        test_path = os.path.join(output_dir, f"test_{station_id}.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        mlflow.log_param("station_id", station_id)
        mlflow.log_param("step", "preprocess")
        mlflow.log_param("years", years)
        mlflow.log_metric("train_rows", len(train_df))
        mlflow.log_metric("test_rows", len(test_df))
        mlflow.log_metric("feature_count", len(all_features))
        mlflow.log_metric("preprocess_duration", time.time() - start_time)
        mlflow.log_artifact(train_path)
        mlflow.log_artifact(test_path)
        mlflow.log_artifact(scaler_file)
        mlflow.log_artifact(wvht_scaler_file)

        print(f"Preprocessing abgeschlossen: {len(train_df)} Trainingszeilen, {len(test_df)} Testzeilen")
        return train_df, test_df, all_features

if __name__ == "__main__":
    station_id = sys.argv[1]
    years = list(map(int, sys.argv[2:])) if len(sys.argv) > 2 else [2021, 2022, 2023]
    preprocess_data(data_dir="output/data", station_id=station_id, years=years)


# In[ ]:




