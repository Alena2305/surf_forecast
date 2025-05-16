#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import time
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

station_id = sys.argv[1]
data_dir = "output"
forecast_path = os.path.join(data_dir, station_id, "forecast_output.csv")
actual_path = os.path.join(data_dir, f"test_{station_id}.csv")

experiment_name = os.getenv("MLFLOW_EXPERIMENT", "surf_forecast")
mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name=f"evaluate_{station_id}"):
    start = time.time()
    mlflow.log_param("station_id", station_id)
    mlflow.log_param("step", "evaluate")

    df_forecast = pd.read_csv(forecast_path, parse_dates=["Zeitpunkt"])
    df_actual = pd.read_csv(actual_path, parse_dates=["datetime"])

    df_merged = pd.merge(df_forecast, df_actual, left_on="Zeitpunkt", right_on="datetime", how="inner")

    if df_merged.empty:
        print("Keine Überlappung von Vorhersage- und Testzeitraum gefunden.")
        mlflow.log_param("eval_status", "no_overlap")
        sys.exit(1)

    y_true = df_merged["wvht"]
    y_pred = df_merged["Wellenhöhe (m)"]

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("eval_duration", time.time() - start)

    result_path = os.path.join(data_dir, station_id, "eval_result.csv")
    df_merged[["Zeitpunkt", "Wellenhöhe (m)", "wvht"]].to_csv(result_path, index=False)
    mlflow.log_artifact(result_path)

    print("Evaluation abgeschlossen:")
    print("RMSE:", rmse, ", MAE:", mae, ", R²:", r2)

