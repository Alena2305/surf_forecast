#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model

app = FastAPI()

class PredictRequest(BaseModel):
    station_id: str

@app.post("/predict")
def predict_wave_height(request: PredictRequest):
    station_id = request.station_id
    base_dir = "output"
    try:
        input_path = os.path.join(base_dir, station_id, "last_24_input.csv")
        model_path = os.path.join(base_dir, "models", f"{station_id}_model.keras")
        feature_path = os.path.join(base_dir, "scalers", f"feature_cols_{station_id}.pkl")
        scaler_path = os.path.join(base_dir, "scalers", f"scaler_{station_id}.pkl")
        wvht_scaler_path = os.path.join(base_dir, "scalers", f"wvht_scaler_{station_id}.pkl")

        df_input = pd.read_csv(input_path, parse_dates=["datetime"])
        model = load_model(model_path)
        feature_cols = joblib.load(feature_path)
        scaler = joblib.load(scaler_path)
        wvht_scaler = joblib.load(wvht_scaler_path)

        last_seq = df_input[feature_cols].values[-24:]
        scaled_seq = scaler.transform(last_seq).reshape(1, 24, -1)

        pred_scaled = model.predict(scaled_seq, verbose=0)[0][0]
        pred = float(wvht_scaler.inverse_transform([[pred_scaled]])[0][0])

        if pred < 0.5:
            tip = "Flach"
        elif pred < 1.0:
            tip = "Anfänger"
        elif pred < 2.5:
            tip = "Fortgeschrittene"
        else:
            tip = "Nur Profis"

        return {
            "station_id": station_id,
            "vorhersage_m": round(pred, 2),
            "surf_tipp": tip
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

