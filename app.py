#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import json
import subprocess
import datetime
import os
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Surf Forecast", layout="wide")
st.title("🏄 Surf Forecast Webapp")
st.markdown("Vorhersage der Wellenhöhe + Surf-Tipps für deinen Spot")

VALID_STATION_FILE = "output/surf_stations_valid.json"

@st.cache_data
def load_station_data():
    with open(VALID_STATION_FILE, "r") as f:
        return json.load(f)

stations = load_station_data()

if not stations:
    st.error("Keine Stationen gefunden.")
    st.stop()

station_labels = list(stations.keys())

m = folium.Map(location=[36.0, -122.0], zoom_start=4)
for label, info in stations.items():
    folium.Marker([info["lat"], info["lon"]], popup=label).add_to(m)
st_data = st_folium(m, height=350)

station_label = st.selectbox("Messstation auswählen:", station_labels)
station_id = stations[station_label]["id"]
lat = stations[station_label]["lat"]
lon = stations[station_label]["lon"]

forecast_days = st.slider("Vorhersagezeitraum (Tage)", 1, 7, 3)
col1, col2 = st.columns(2)
start_date = col1.date_input("Startdatum", datetime.date.today())
end_date = col2.date_input("Enddatum", datetime.date.today() + datetime.timedelta(days=forecast_days))

if st.button("Starte Vorhersage"):
    with st.spinner("Starte Forecast-Pipeline..."):
        try:
            subprocess.run(["python", "01_collect_data.py", station_id], check=True)

            years_file = f"output/data/years_{station_id}.json"
            if not os.path.exists(years_file):
                raise FileNotFoundError(f"Kein Jahresverzeichnis für Station {station_id} gefunden.")
            with open(years_file, "r") as f:
                years = json.load(f)

            subprocess.run(["python", "02_preprocess.py", station_id] + [str(y) for y in years], check=True)
            subprocess.run(["python", "03_train.py", station_id], check=True)
            subprocess.run(["python", "06_forecast_input.py", station_id], check=True)
            subprocess.run(["python", "05_predict.py", station_id], check=True)

            forecast_path = f"output/{station_id}/forecast_output.csv"
            if os.path.exists(forecast_path):
                df = pd.read_csv(forecast_path, parse_dates=["Zeitpunkt"])
                df = df[(df["Zeitpunkt"].dt.date >= start_date) & (df["Zeitpunkt"].dt.date <= end_date)]
                st.success("Forecast erfolgreich erstellt!")
                st.dataframe(df[["Zeitpunkt", "Wellenhöhe (m)", "Surf-Tipp"]])
            else:
                st.error("Forecast-Datei wurde nicht gefunden.")
        except subprocess.CalledProcessError as e:
            st.error(f"Fehler bei der Pipeline-Ausführung: {e}")
        except Exception as e:
            st.error(f"Unerwarteter Fehler: {e}")

