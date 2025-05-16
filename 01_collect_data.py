#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import requests
import gzip
import shutil
import mlflow
import json
from datetime import datetime

def get_recent_years(num_years=5):
    current_year = datetime.now().year
    return [current_year - i for i in range(num_years)][::-1]

def download_noaa_data(station_id, years, target_dir="output/data"):
    os.makedirs(target_dir, exist_ok=True)
    available_years = []
    download_paths = []
    for year in years:
        url = f"https://www.ndbc.noaa.gov/data/historical/stdmet/{station_id}h{year}.txt.gz"
        local_gz = os.path.join(target_dir, f"{station_id}_{year}.txt.gz")
        local_txt = os.path.join(target_dir, f"{station_id}_{year}.txt")
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_gz, "wb") as f:
                f.write(response.content)
            with gzip.open(local_gz, 'rb') as f_in:
                with open(local_txt, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            available_years.append(year)
            download_paths.append(local_txt)
    return available_years, download_paths

if __name__ == "__main__":
    station_id = sys.argv[1] if len(sys.argv) > 1 else "46042"
    years = get_recent_years()

    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "surf_forecast")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"collect_{station_id}"):
        mlflow.log_param("station_id", station_id)
        mlflow.log_param("step", "collect")
        mlflow.log_param("requested_years", years)

        available_years, paths = download_noaa_data(station_id, years)
        mlflow.log_param("downloaded_years", available_years)

        if not available_years:
            raise RuntimeError(f"Für Station {station_id} konnten keine NOAA-Daten heruntergeladen werden!")

        for path in paths:
            mlflow.log_artifact(path)

        years_path = os.path.join("output", "data", f"years_{station_id}.json")
        with open(years_path, "w") as f:
            json.dump(available_years, f)

        mlflow.log_artifact(years_path)
        print(f"NOAA-Daten gespeichert für Jahre: {available_years}")

