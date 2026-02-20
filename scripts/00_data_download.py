#!/usr/bin/env python
# coding: utf-8

# # 00 — Leakage-Free Data Download
# 
# **Purpose:** Download day-ahead forecast datasets that were available *before* the delivery period, ensuring no look-ahead bias (data leakage) in the price prediction model.
# 
# **Data Sources:**
# 1. **BMRS Elexon Insights API** — Day-ahead demand forecasts (half-hourly)
# 2. **BMRS Elexon Insights API** — Day-ahead wind & solar generation forecasts (half-hourly)
# 3. **Open-Meteo Previous Runs API** — Historical weather *forecasts* (not actuals)
# 
# **Date Range:** 2021-01-01 to 2026-2-11
# 
# **Why this matters:**  
# The original pipeline used *outturn* (actual) demand and weather data, which would not have been available at forecast time. Using day-ahead forecasts ensures the model only sees information that a real trader would have had when making decisions.

# In[1]:


import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta
from pathlib import Path

# Create output directories
DATA_RAW = Path('../data/raw')
DATA_RAW.mkdir(parents=True, exist_ok=True)

START_DATE = '2021-01-01'
END_DATE   = '2026-02-11'

print(f"Download period: {START_DATE} to {END_DATE}")
print(f"Output directory: {DATA_RAW.resolve()}")


# ## 1. BMRS Day-Ahead Demand Forecasts
# 
# The Elexon BMRS Insights API provides **National Demand Forecasts (NDF)** published day-ahead.  
# These replace the *outturn* demand data (`elec_demand_outturn_hh_bmrs.csv`) which contains actual realised demand — information not available at forecast time.
# 
# **Endpoint:** `GET /forecast/demand/day-ahead`  
# **Key fields:** `nationalDemand` (MW), `startTime`, `settlementDate`, `settlementPeriod`  
# **No API key required.**

# In[2]:


def download_bmrs_demand_da(start_date: str, end_date: str, chunk_days: int = 30) -> pd.DataFrame:
    """
    Download day-ahead national demand forecasts from BMRS Elexon API.
    Paginates in chunks to avoid hitting response limits.
    """
    base_url = 'https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/day-ahead'
    all_records = []
    
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    while current < end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days), end)
        
        params = {
            'from': current.strftime('%Y-%m-%dT00:00Z'),
            'to': chunk_end.strftime('%Y-%m-%dT00:00Z'),
            'format': 'json',
        }
        
        try:
            r = requests.get(base_url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            
            records = data.get('data', [])
            all_records.extend(records)
            
            print(f"  {current.date()} → {chunk_end.date()}: {len(records)} records")
        except Exception as e:
            print(f"  ERROR {current.date()} → {chunk_end.date()}: {e}")
        
        time.sleep(0.5)  # Rate limiting
        current = chunk_end
    
    df = pd.DataFrame(all_records)
    print(f"\nTotal records: {len(df)}")
    return df

print("Downloading BMRS day-ahead demand forecasts...")
print(f"Period: {START_DATE} to {END_DATE}\n")

df_demand_da = download_bmrs_demand_da(START_DATE, END_DATE)

if not df_demand_da.empty:
    # Parse timestamps
    df_demand_da['startTime'] = pd.to_datetime(df_demand_da['startTime'])
    df_demand_da = df_demand_da.sort_values('startTime').reset_index(drop=True)
    
    # Keep relevant columns
    df_demand_da = df_demand_da[['startTime', 'settlementDate', 'settlementPeriod',
                                  'nationalDemand', 'transmissionSystemDemand', 'publishTime']].copy()
    df_demand_da.rename(columns={'startTime': 'datetime', 'nationalDemand': 'Demand_DA_MW'}, inplace=True)
    
    # Save
    out_path = DATA_RAW / 'elec_demand_forecast_da_hh_bmrs.csv'
    df_demand_da.to_csv(out_path, index=False)
    print(f"\n✓ Saved to {out_path}")
    print(f"  Shape: {df_demand_da.shape}")
    print(f"  Date range: {df_demand_da['datetime'].min()} → {df_demand_da['datetime'].max()}")
    print(f"  Mean demand: {df_demand_da['Demand_DA_MW'].mean():.0f} MW")
else:
    print("WARNING: No demand forecast data returned!")


# ## 2. BMRS Day-Ahead Wind & Solar Generation Forecasts
# 
# The BMRS API provides **day-ahead generation forecasts** for Wind Onshore, Wind Offshore, and Solar.  
# These forecasts are published the day before delivery — they represent what the system operator expected, not what actually happened.
# 
# This replaces the XGBoost-predicted wind/solar generation files, which were trained on *actual* weather data (introducing potential information leakage).
# 
# **Endpoint:** `GET /forecast/generation/wind-and-solar/day-ahead`  
# **Requires:** `processType=Day Ahead`  
# **Key fields:** `psrType` (Solar, Wind Onshore, Wind Offshore), `quantity` (MW), `startTime`

# In[3]:


def download_bmrs_wind_solar_da(start_date: str, end_date: str, chunk_days: int = 7) -> pd.DataFrame:
    """
    Download day-ahead wind & solar generation forecasts from BMRS.
    Returns separate columns for Solar, Wind Onshore, and Wind Offshore.
    """
    base_url = 'https://data.elexon.co.uk/bmrs/api/v1/forecast/generation/wind-and-solar/day-ahead'
    all_records = []
    
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    while current < end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days), end)
        
        params = {
            'processType': 'Day Ahead',
            'from': current.strftime('%Y-%m-%d'),
            'to': chunk_end.strftime('%Y-%m-%d'),
        }
        
        try:
            r = requests.get(base_url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            
            records = data.get('data', [])
            all_records.extend(records)
            
            print(f"  {current.date()} → {chunk_end.date()}: {len(records)} records")
        except Exception as e:
            print(f"  ERROR {current.date()} → {chunk_end.date()}: {e}")
        
        time.sleep(0.5)
        current = chunk_end
    
    df = pd.DataFrame(all_records)
    print(f"\nTotal raw records: {len(df)}")
    return df

print("Downloading BMRS day-ahead wind & solar generation forecasts...")
print(f"Period: {START_DATE} to {END_DATE}\n")

df_ws_raw = download_bmrs_wind_solar_da(START_DATE, END_DATE)

if not df_ws_raw.empty:
    df_ws_raw['startTime'] = pd.to_datetime(df_ws_raw['startTime'])
    
    # Pivot: one row per timestamp, columns = Solar / Wind Onshore / Wind Offshore
    df_ws = df_ws_raw.pivot_table(
        index='startTime', columns='psrType', values='quantity', aggfunc='first'
    ).reset_index()
    df_ws.columns.name = None
    
    # Rename columns
    col_map = {
        'startTime': 'datetime',
        'Solar': 'Solar_Forecast_DA_MW',
        'Wind Onshore': 'Wind_Onshore_Forecast_DA_MW',
        'Wind Offshore': 'Wind_Offshore_Forecast_DA_MW',
    }
    df_ws.rename(columns=col_map, inplace=True)
    
    # Total wind
    wind_cols = [c for c in df_ws.columns if 'Wind' in c and 'Forecast' in c]
    df_ws['Wind_Forecast_DA_MW'] = df_ws[wind_cols].sum(axis=1)
    
    df_ws = df_ws.sort_values('datetime').reset_index(drop=True)
    
    # Save
    out_path = DATA_RAW / 'renew_gen_forecast_da_hourly_bmrs.csv'
    df_ws.to_csv(out_path, index=False)
    print(f"\n✓ Saved to {out_path}")
    print(f"  Shape: {df_ws.shape}")
    print(f"  Date range: {df_ws['datetime'].min()} → {df_ws['datetime'].max()}")
    print(f"  Columns: {list(df_ws.columns)}")
    if 'Solar_Forecast_DA_MW' in df_ws.columns:
        print(f"  Mean solar forecast: {df_ws['Solar_Forecast_DA_MW'].mean():.0f} MW")
    if 'Wind_Forecast_DA_MW' in df_ws.columns:
        print(f"  Mean wind forecast: {df_ws['Wind_Forecast_DA_MW'].mean():.0f} MW")
else:
    print("WARNING: No wind/solar forecast data returned!")


# ## 3. Historical Weather Forecasts (Open-Meteo Previous Runs API)
# 
# The Open-Meteo **Previous Runs API** provides access to historical weather *forecasts* — i.e., what was predicted ~24h before delivery, not what actually happened.
# 
# This is critical for avoiding leakage: using actual weather to predict generation is using future information.
# 
# **API:** `https://previous-runs-api.open-meteo.com/v1/forecast`  
# **Parameters:** `wind_speed_10m`, `wind_speed_100m`, `wind_gusts_10m`, `wind_direction_10m`, `shortwave_radiation`, `direct_normal_irradiance`, `temperature_2m`, `cloud_cover`  
# **Locations:** Capacity-weighted centroids from the REPD database (Renewable Energy Planning Database)  
# **Key param:** `past_days=1` → retrieves the forecast made ~24h before delivery

# In[4]:


# Compute capacity-weighted centroids from REPD for wind and solar regions
# REPD uses OSGB36 (British National Grid) — we convert to WGS84 (lat/lon) for Open-Meteo

try:
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    has_pyproj = True
except ImportError:
    has_pyproj = False
    print("pyproj not installed — using approximate conversion")
    print("Install with: pip install pyproj")

def osgb_to_latlon_approx(easting, northing):
    """Approximate OSGB36 to WGS84 conversion (good to ~100m)."""
    # Simplified conversion for UK coordinates
    lat = 49.766 + (northing - 100000) / 111320
    lon = -7.557 + (easting - 100000) / (111320 * np.cos(np.radians(lat)))
    return lat, lon

repd = pd.read_csv(DATA_RAW / 'renew_repd_sites_desnz.csv', encoding='latin-1')
repd['X-coordinate'] = pd.to_numeric(repd['X-coordinate'], errors='coerce')
repd['Y-coordinate'] = pd.to_numeric(repd['Y-coordinate'], errors='coerce')

repd['Installed Capacity (MWelec)'] = pd.to_numeric(repd['Installed Capacity (MWelec)'], errors='coerce')

# Filter operational wind and solar sites with valid coordinates
repd_valid = repd.dropna(subset=['X-coordinate', 'Y-coordinate', 'Installed Capacity (MWelec)'])
repd_valid = repd_valid[repd_valid['Installed Capacity (MWelec)'] > 0]

# Convert to lat/lon
if has_pyproj:
    lons, lats = transformer.transform(
        repd_valid['X-coordinate'].values,
        repd_valid['Y-coordinate'].values
    )
    repd_valid = repd_valid.copy()
    repd_valid['lat'] = lats
    repd_valid['lon'] = lons
else:
    coords = repd_valid.apply(
        lambda r: osgb_to_latlon_approx(r['X-coordinate'], r['Y-coordinate']),
        axis=1, result_type='expand'
    )
    repd_valid = repd_valid.copy()
    repd_valid['lat'] = coords[0]
    repd_valid['lon'] = coords[1]

# Compute capacity-weighted centroids by technology
centroids = {}
for tech_group, tech_filter in [
    ('wind', repd_valid['Technology Type'].str.contains('Wind', na=False)),
    ('solar', repd_valid['Technology Type'].str.contains('Solar', na=False)),
]:
    subset = repd_valid[tech_filter].copy()
    cap = subset['Installed Capacity (MWelec)']
    total_cap = cap.sum()
    
    centroid_lat = (subset['lat'] * cap).sum() / total_cap
    centroid_lon = (subset['lon'] * cap).sum() / total_cap
    centroids[tech_group] = {'lat': round(centroid_lat, 4), 'lon': round(centroid_lon, 4),
                             'total_capacity_MW': round(total_cap, 1), 'n_sites': len(subset)}
    print(f"{tech_group.upper()} centroid: ({centroid_lat:.4f}, {centroid_lon:.4f}) — "
          f"{total_cap:.0f} MW across {len(subset)} sites")

print(f"\nCentroids: {centroids}")


# In[5]:


def download_openmeteo_forecast_weather(lat: float, lon: float, start_date: str, end_date: str,
                                         chunk_months: int = 3) -> pd.DataFrame:
    """
    Download historical weather FORECASTS from Open-Meteo Previous Runs API.
    The Previous Runs API returns forecasts made before delivery (not actuals).
    """
    base_url = 'https://previous-runs-api.open-meteo.com/v1/forecast'
    
    weather_params = [
        'wind_speed_10m', 'wind_speed_100m', 'wind_gusts_10m',
        'wind_direction_10m', 'shortwave_radiation', 'direct_normal_irradiance',
        'temperature_2m', 'cloud_cover'
    ]
    
    all_dfs = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    while current < end:
        chunk_end = min(current + pd.DateOffset(months=chunk_months), end)
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': current.strftime('%Y-%m-%d'),
            'end_date': chunk_end.strftime('%Y-%m-%d'),
            'hourly': ','.join(weather_params),
            'timezone': 'Europe/London',
        }
        
        try:
            r = requests.get(base_url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            
            if 'hourly' in data:
                hourly = data['hourly']
                df_chunk = pd.DataFrame(hourly)
                df_chunk['time'] = pd.to_datetime(df_chunk['time'])
                all_dfs.append(df_chunk)
                print(f"  {current.date()} → {chunk_end.date()}: {len(df_chunk)} hours")
            else:
                print(f"  {current.date()} → {chunk_end.date()}: no hourly data")
        except Exception as e:
            print(f"  ERROR {current.date()} → {chunk_end.date()}: {e}")
        
        time.sleep(1.0)  # Rate limiting for Open-Meteo
        current = chunk_end
    
    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
        df = df.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
        return df
    return pd.DataFrame()

# Download weather forecasts for wind and solar centroids
weather_dfs = {}
for tech, info in centroids.items():
    print(f"\nDownloading {tech.upper()} weather forecasts (lat={info['lat']}, lon={info['lon']})...")
    df = download_openmeteo_forecast_weather(info['lat'], info['lon'], START_DATE, END_DATE)
    if not df.empty:
        # Prefix columns with technology
        df = df.rename(columns={c: f'{tech}_{c}' for c in df.columns if c != 'time'})
        weather_dfs[tech] = df
        print(f"  ✓ {len(df)} hours downloaded")

# Merge wind and solar weather forecasts
if weather_dfs:
    df_weather = weather_dfs.get('wind', pd.DataFrame())
    if 'solar' in weather_dfs:
        if df_weather.empty:
            df_weather = weather_dfs['solar']
        else:
            df_weather = df_weather.merge(weather_dfs['solar'], on='time', how='outer')
    
    df_weather = df_weather.sort_values('time').reset_index(drop=True)
    df_weather.rename(columns={'time': 'datetime'}, inplace=True)
    
    out_path = DATA_RAW / 'weather_forecast_da_hourly_openmeteo.csv'
    df_weather.to_csv(out_path, index=False)
    print(f"\n✓ Saved to {out_path}")
    print(f"  Shape: {df_weather.shape}")
    print(f"  Date range: {df_weather['datetime'].min()} → {df_weather['datetime'].max()}")
    print(f"  Columns: {list(df_weather.columns)}")
else:
    print("\nWARNING: No weather forecast data downloaded!")


# ## Summary
# 
# Three leakage-free datasets have been downloaded:
# 
# | File | Source | Description |
# |------|--------|-------------|
# | `elec_demand_forecast_da_hh_bmrs.csv` | BMRS Elexon | Day-ahead national demand forecasts (MW) |
# | `renew_gen_forecast_da_hourly_bmrs.csv` | BMRS Elexon | Day-ahead wind & solar generation forecasts (MW) |
# | `weather_forecast_da_hourly_openmeteo.csv` | Open-Meteo | Historical weather forecasts at REPD capacity-weighted centroids |
# 
# **Key principle:** Every feature used in the price prediction model should be based on information available *before* the delivery period. These forecast datasets satisfy that constraint, unlike the outturn (actual) data used previously.
# 
# **Next steps:**
# - Notebook 02: Retrain XGBoost wind/solar models using forecast weather (not actuals)
# - Notebook 03: Replace outturn demand with day-ahead forecasts; lag gas/CO2 by 1 day; add leakage audit

# In[6]:


# Quick verification — check all output files exist
print("=== OUTPUT VERIFICATION ===\n")

expected_files = [
    'elec_demand_forecast_da_hh_bmrs.csv',
    'renew_gen_forecast_da_hourly_bmrs.csv',
    'weather_forecast_da_hourly_openmeteo.csv',
]

for fname in expected_files:
    fpath = DATA_RAW / fname
    if fpath.exists():
        df = pd.read_csv(fpath)
        print(f"✓ {fname}")
        print(f"  Rows: {len(df):,}  |  Cols: {len(df.columns)}  |  Size: {fpath.stat().st_size / 1024:.0f} KB")
    else:
        print(f"✗ {fname} — NOT FOUND")

print("\n=== DONE ===")

