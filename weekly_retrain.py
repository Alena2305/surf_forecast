#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import os
import subprocess
import json
import traceback
import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_EXPERIMENT"] = "surf_forecast_retrain"

def run_step(cmd):
    try:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        print(f"Fehler bei Befehl: {cmd}")
        traceback.print_exc()
        return False

def get_latest_metrics(station_id, experiment_name="surf_forecast_retrain"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.station_id = '{station_id}' and params.step = 'evaluate'",
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        return None
    return runs[0].data.metrics

def get_best_previous_rmse(station_id, experiment_name="surf_forecast_retrain"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.station_id = '{station_id}' and params.step = 'evaluate'",
        order_by=["metrics.rmse ASC"],
        max_results=1
    )
    if not runs:
        return None
    return runs[0].data.metrics.get("rmse", None)

run_step("python build_station_index.py")

with open("output/surf_stations_valid.json", "r") as f:
    stations = json.load(f)

for name, info in stations.items():
    station_id = info["id"]
    print(f"\n Starte Verarbeitung für {station_id} ({name})")

    try:
        run_step(f"python 01_collect_data.py {station_id}")
        run_step(f"python 02_preprocess.py {station_id} 2021 2022 2023 2024")
        run_step(f"python 03_train.py {station_id}")
        run_step(f"python 04_evaluate.py {station_id}")

        new_metrics = get_latest_metrics(station_id)
        best_rmse = get_best_previous_rmse(station_id)

        if new_metrics and best_rmse:
            new_rmse = new_metrics.get("rmse")
            print(f"Neues RMSE: {new_rmse:.4f} | Bestes bisheriges RMSE: {best_rmse:.4f}")
            if new_rmse > best_rmse:
                print("Neues Modell ist schlechter! Überprüfung oder Rollback empfohlen.")
            else:
                print("Neues Modell ist besser oder gleich gut.")

    except Exception as e:
        print(f"Fehler bei Station {station_id}: {e}")
        continue

