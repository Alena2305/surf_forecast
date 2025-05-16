#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import json
import re
import os
import datetime
from pathlib import Path

NOAA_JSON_URL = "https://www.ndbc.noaa.gov/ndbcmapstations.json"
DATA_BASE = Path("output")
VALID_STATION_FILE = DATA_BASE / "surf_stations_valid.json"

MIN_VALID_YEARS = 3
CURRENT_YEAR = datetime.datetime.now().year


def fetch_all_stations():
    try:
        response = requests.get(NOAA_JSON_URL)
        response.raise_for_status()
        raw_text = response.text
        match = re.search(r'(\[\{.*?\}\])', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception as e:
        print(f"Fehler beim Laden der NOAA-Stationsliste: {e}")
    return []


def has_enough_data(station_id):
    base_url = f"https://www.ndbc.noaa.gov/view_text_file.php?filename={station_id}h.txt.gz&dir=data/historical/stdmet/"
    available_years = []

    for year in range(CURRENT_YEAR - 5, CURRENT_YEAR):
        try:
            url = base_url.replace(".txt", f"_{year}.txt")
            r = requests.head(url)
            if r.status_code == 200:
                available_years.append(year)
        except Exception:
            continue

    return len(available_years) >= MIN_VALID_YEARS


def has_realtime_data(station_id):
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    try:
        r = requests.head(url)
        return r.status_code == 200
    except Exception:
        return False


def build_valid_station_index():
    all_stations = fetch_all_stations()
    valid = {}

    for s in all_stations:
        st_id = s.get("id", "").strip()
        st_name = s.get("name", "").strip()
        st_type = s.get("type", "").lower()
        lat = float(s.get("lat", 0))
        lon = float(s.get("lon", 0))

        if st_type == "buoy" and (15 <= lat <= 50) and (-160 <= lon <= -60):
            if has_enough_data(st_id) and has_realtime_data(st_id):
                label = f"{st_name} ({st_id})"
                valid[label] = {"id": st_id, "lat": lat, "lon": lon}
                print(f"✅ {label}")
            else:
                print(f"Nicht genug oder keine Realtime-Daten: {st_id}")

    DATA_BASE.mkdir(parents=True, exist_ok=True)
    with open(VALID_STATION_FILE, "w", encoding="utf-8") as f:
        json.dump(valid, f, indent=2)
    print(f"\nGültige Stationen gespeichert in {VALID_STATION_FILE}")


if __name__ == "__main__":
    build_valid_station_index()

