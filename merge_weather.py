#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime
import pandas as pd
from weather_fetch import fetch_weather_forecast

def merge_weather_with_noaa(noaa_df, lat, lon, days=7):
    if noaa_df["datetime"].max() < datetime.utcnow():
        return noaa_df

    days = min(days, 16) 
    weather_df = fetch_weather_forecast(lat, lon, days=days)

    merged = pd.merge(noaa_df, weather_df, on="datetime", how="inner")

    if merged.empty:
        print("Merge ergab keine gemeinsamen Zeitpunkte – nutze nur NOAA-Daten.")
        return noaa_df

    return merged

if __name__ == "__main__":
    lat, lon = 36.96, -122.02
    weather_df = fetch_weather_forecast(lat, lon, days=7)
    print("Wetterdaten geladen:")
    print(weather_df.head())

