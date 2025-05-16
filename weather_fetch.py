#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_weather_forecast(lat, lon, start=None, days=7):
   
    if not start:
        start = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    MAX_API_DATE = datetime(2025, 5, 28)
    end_candidate = start + timedelta(days=days)
    end = min(end_candidate, MAX_API_DATE)

    if end <= start:
        raise ValueError("Ungültiger Wetterzeitraum: start liegt nach dem API-Limit")

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=wind_speed_10m,wind_direction_10m,temperature_2m"
        f"&start_date={start.date()}&end_date={end.date()}"
        f"&timezone=UTC"
    )

    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Fehler beim Abrufen der Wetterdaten: {response.text}")

    data = response.json()["hourly"]
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    return df

if __name__ == "__main__":
    df = fetch_weather_forecast(36.96, -122.02, days=10)
    print("Vorschau Wetterdaten:")
    print(df.head())

