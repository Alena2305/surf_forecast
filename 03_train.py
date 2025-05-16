#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import time
import pandas as pd
import numpy as np
import mlflow
import joblib
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

station_id = sys.argv[1]
data_dir = "output"
model_dir = os.path.join(data_dir, "models")
scaler_dir = os.path.join(data_dir, "scalers")
train_path = os.path.join(data_dir, f"train_{station_id}.csv")
test_path = os.path.join(data_dir, f"test_{station_id}.csv")
model_path = os.path.join(model_dir, f"{station_id}_model.keras")


if os.path.exists(model_path):
    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path)).date()
    if mod_time == datetime.today().date():
        print(f"Modell für {station_id} bereits heute vorhanden – Training wird übersprungen.")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT", "surf_forecast")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"skip_train_{station_id}"):
            mlflow.log_param("station_id", station_id)
            mlflow.log_param("step", "train_skipped")
            mlflow.log_param("reason", "existing_model")
        sys.exit(0)


experiment_name = os.getenv("MLFLOW_EXPERIMENT", "surf_forecast")
mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name=f"train_{station_id}"):
    start_time = time.time()

    train_df = pd.read_csv(train_path, parse_dates=["datetime"])
    test_df = pd.read_csv(test_path, parse_dates=["datetime"])
    all_features = [col for col in train_df.columns if col not in ["datetime", "wvht"]]

    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(all_features, os.path.join(scaler_dir, f"feature_cols_{station_id}.pkl"))

    def make_sequences(df, feature_cols, seq_len=24, max_samples=2000):
        X, y = [], []
        df = df.reset_index(drop=True)
        feature_df = df[feature_cols]
        wvht_series = df["wvht"]

        max_i = min(len(df) - seq_len, max_samples)

        for i in range(max_i):
            window = feature_df.iloc[i:i+seq_len].values
            target = wvht_series.iloc[i+seq_len]
            X.append(window)
            y.append(target)

        return np.array(X), np.array(y)

    X_train, y_train = make_sequences(train_df, all_features, max_samples=1000)
    X_test, y_test = make_sequences(test_df, all_features, max_samples=500)

    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False)) 
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")

    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        epochs=5,                   
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    os.makedirs(model_dir, exist_ok=True)
    model.save(model_path)
    mlflow.log_param("station_id", station_id)
    mlflow.log_param("step", "train")
    mlflow.log_metric("train_samples", len(X_train))
    mlflow.log_metric("val_samples", len(X_test))
    mlflow.log_metric("train_duration", time.time() - start_time)
    mlflow.log_artifact(model_path)

    print(f"Modell gespeichert: {model_path}")

