#!/usr/bin/env python
# coding: utf-8

# 
# UK WEATHER DATA PIPELINE FOR ELECTRICITY PRICE PREDICTION
# ==========================================================
# By ANI
# 
# WORKFLOW:
# 1. Load REPD (Renewable Energy Planning Database) - UK govt database of all renewable energy projects
# 2. Filter for operational wind and solar projects (>1 MW)
# 3. Convert coordinates from British National Grid to lat/lon
# 4. Calculate capacity-weighted centroids for each UK region
# 5. Select top regions covering 95% of installed capacity
# 6. Fetch historical weather data from Open-Meteo API (2021-2025)
# 7. Save required wind and solar data per region into different folders and csv's
# 
# DATA SOURCES:
# 
# REPD: https://www.gov.uk/government/publications/renewable-energy-planning-database-monthly-extract
# 
# Weather: Open-Meteo Historical API (ERA5 reanalysis)
# 
# Data Period: 2021-01-01 to 2026-02-11
# 

# In[1]:


import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import time
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

print("Imports complete")


# In[2]:


# Time period
start_date = '2021-01-01'
end_date = '2026-02-11'

# Technology filters
WIND_ONSHORE = 'Wind Onshore'
WIND_OFFSHORE = 'Wind Offshore'
SOLAR = 'Solar Photovoltaics'

# Minimum capacity threshold (MW) - filter out tiny installations
MIN_CAPACITY_MW = 1.0

print(f"Configuration set: {start_date} to {end_date}")


# In[3]:


#Load REPD CSV
import os
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    REPD_PATH = '/content/drive/MyDrive/Colab Notebooks/Ani_Data/repd.csv'
else:
    REPD_PATH = '../data/raw/renew_repd_sites_desnz.csv'

# Check if the file exists before trying to read it
if not os.path.exists(REPD_PATH):
    print(f"File not found: {REPD_PATH}")
else:
    repd_raw = pd.read_csv(REPD_PATH, encoding='latin-1')
    print(f"Loaded {len(repd_raw)} records from REPD")
    print(repd_raw.head())


# In[4]:


#Filter and cleaning the REPD csv

def clean_repd(df):
    """Filter REPD for operational wind and solar projects"""

    # Check actual column names
    print("Checking column names...")
    print(df.columns.tolist())

    keep_cols = [
        'Site Name', 'Region', 'Technology Type', 'Development Status',
        'Installed Capacity (MWelec)', 'X-coordinate', 'Y-coordinate'
    ]
    print(keep_cols)

    # Filter for operational projects
    df_operational = df[df['Development Status'] == 'Operational'].copy()
    print(f"Operational projects: {len(df_operational)}")

    # Filter for wind and solar
    tech_filter = df_operational['Technology Type'].isin([WIND_ONSHORE, WIND_OFFSHORE, SOLAR])
    df_filtered = df_operational[tech_filter].copy()
    print(f"Wind/Solar operational: {len(df_filtered)}")

    # Convert 'Installed Capacity (MWelec)' to numeric, coercing errors, then drop NaNs
    df_filtered['Installed Capacity (MWelec)'] = pd.to_numeric(df_filtered['Installed Capacity (MWelec)'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['Installed Capacity (MWelec)']).copy()

    # Filter minimum capacity
    df_filtered = df_filtered[df_filtered['Installed Capacity (MWelec)'] >= MIN_CAPACITY_MW]
    print(f"After {MIN_CAPACITY_MW}MW threshold: {len(df_filtered)}")

    # Remove missing coordinates
    df_filtered = df_filtered.dropna(subset=['X-coordinate', 'Y-coordinate'])
    print(f"With valid coordinates: {len(df_filtered)}")

    df_filtered = df_filtered[keep_cols]

    # Summary by technology
    print("\n" + "="*50)
    print("CAPACITY SUMMARY BY TECHNOLOGY")
    print("="*50)
    summary = df_filtered.groupby('Technology Type')['Installed Capacity (MWelec)'].agg(['count', 'sum'])
    summary.columns = ['Projects', 'Total MW']
    print(summary)
    print("="*50)

    return df_filtered

repd_clean = clean_repd(repd_raw)


# In[5]:


def add_lat_lon(df):

    # 1. (EPSG:27700 -> EPSG:4326)
    # always_xy=True ensures the output is (Longitude, Latitude)
    transformer = Transformer.from_crs("epsg:27700", "epsg:4326", always_xy=True)

    # 2. Vectorized Transform
    lon, lat = transformer.transform(df['X-coordinate'].values, df['Y-coordinate'].values)

    # 3. Assign new columns
    df = df.copy()
    df['longitude'] = lon
    df['latitude'] = lat

    return df

# --- Execution Flow ---
repd_clean = clean_repd(repd_raw)

# 2. Run the conversion
repd_final = add_lat_lon(repd_clean)

# 3. Check results
print(repd_final[['Site Name', 'latitude', 'longitude']].head())


# In[6]:


#Work out weighted capacity for each location
def calculate_all_weighted_locations(df, tech_types):

    # Filter Tech
    df_tech = df[df['Technology Type'].isin(tech_types)].copy()

    # Create Weighted Coordinates
    cap = 'Installed Capacity (MWelec)'
    df_tech['w_lat'] = df_tech['latitude'] * df_tech[cap]
    df_tech['w_lon'] = df_tech['longitude'] * df_tech[cap]

    # Group & Sort
    regions = df_tech.groupby('Region').agg(
        total_capacity_mw=(cap, 'sum'),
        lat_sum=('w_lat', 'sum'),
        lon_sum=('w_lon', 'sum')
    ).sort_values('total_capacity_mw', ascending=False)

    # Calculate Cumulative %
    total_uk = regions['total_capacity_mw'].sum()
    regions['cumulative%'] = regions['total_capacity_mw'].cumsum() / total_uk
    regions['global_share'] = regions['total_capacity_mw'] / total_uk

    # Final Centroids
    regions['latitude'] = regions['lat_sum'] / regions['total_capacity_mw']
    regions['longitude'] = regions['lon_sum'] / regions['total_capacity_mw']

    return regions.reset_index()[['Region', 'latitude', 'longitude', 'total_capacity_mw', 'cumulative%', 'global_share']]


# In[7]:


wind_all = calculate_all_weighted_locations(repd_final, [WIND_ONSHORE, WIND_OFFSHORE])
solar_all = calculate_all_weighted_locations(repd_final, [SOLAR])
for name, df in [("WIND", wind_all), ("SOLAR", solar_all)]:
    print(f"\n{'='*80}\n{name} CAPACITY-WEIGHTED LOCATIONS\n{'='*80}")
    print(df.to_string(index=False, formatters={
        'latitude': '{:.4f}'.format,
        'longitude': '{:.4f}'.format,
        'total_capacity_mw': '{:.1f}'.format,
        'global_share': '{:.2%}'.format,
        'cumulative%': '{:.2%}'.format
    }))


# In[8]:


# --- 2. Filter (Top 95%) ---
wind_final = wind_all[wind_all['cumulative%'] <= 0.95].copy()
solar_final = solar_all[solar_all['cumulative%'] <= 0.95].copy()

# --- 3. Print Evidence (Wind) ---
print("--- WIND JUSTIFICATION ---")
print(f"Original Regions: {len(wind_all)}")
print(f"Final Regions:    {len(wind_final)}")
print(f"Noise Removed:    {len(wind_all) - len(wind_final)} regions")
print(f"Capacity Kept:    {wind_final['global_share'].sum():.1%}")
print("\n", wind_final)

# --- 4. Print Evidence (Solar) ---
print("\n\n--- SOLAR JUSTIFICATION ---")
print(f"Original Regions: {len(solar_all)}")
print(f"Final Regions:    {len(solar_final)}")
print(f"Noise Removed:    {len(solar_all) - len(solar_final)} regions")
print(f"Capacity Kept:    {solar_final['global_share'].sum():.1%}")
print("\n", solar_final)


# In[9]:


#Open-Meteo API

def fetch_weather_for_one_location(lat, lon, start_date, end_date):

    #Define the API URL
    url = "https://archive-api.open-meteo.com/v1/archive"

    #Define what is needed from the API
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': [
            'wind_speed_10m',           # Wind speed at 10 meters (m/s)
            'wind_speed_100m',          # Wind speed at 100m - turbine hub height (m/s)
            'wind_gusts_10m',           # Wind gusts (m/s)
            'wind_direction_100m',      # Wind direction in degrees (0-360)
            'shortwave_radiation',      # Solar radiation GHI (W/m²)
            'direct_normal_irradiance', # Solar radiation DNI (W/m²)
            'cloud_cover',              # Cloud cover (%)
            'temperature_2m',           # Temperature at 2m (°C)
        ],
        'timezone': 'Europe/London'
    }

    #Send request to API
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    #Convert response to JSON (Python dictionary)
    data = response.json()

    # Step 5: Extract the hourly data into a DataFrame
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(data['hourly']['time']),
        'wind_speed_10m': data['hourly']['wind_speed_10m'],
        'wind_speed_100m': data['hourly']['wind_speed_100m'],
        'wind_gusts': data['hourly']['wind_gusts_10m'],
        'wind_direction': data['hourly']['wind_direction_100m'],
        'ghi': data['hourly']['shortwave_radiation'],
        'dni': data['hourly']['direct_normal_irradiance'],
        'cloud_cover': data['hourly']['cloud_cover'],
        'temperature': data['hourly']['temperature_2m'],
    })

    # Step 6: Return the DataFrame
    return df


# In[10]:


import os
import time
import pandas as pd

def fetch_all_locations(locations_df, start_date, end_date):
    """
    Fetch weather data, SAVE Files, resume progress if crash.
    """
    if IN_COLAB:
        folder_path = r"/content/drive/MyDrive/Colab Notebooks/Ani_Data/"
    else:
        folder_path = os.path.join('..', 'data', 'processed', 'regions')

    # Ensure folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    all_data = {}

    for index, row in locations_df.iterrows():
        name = row['Region']
        lat = row['latitude']
        lon = row['longitude']

        # Check if file already saved before a crash
        if os.path.exists(os.path.join(folder_path, f"{name}_Solar.csv")):
            print(f"Skipping {name} (Already Saved)")
            continue
        # ----------------------

        print(f"Fetching {name}...")

        try:
            # 1. Fetch from API
            df = fetch_weather_for_one_location(lat, lon, start_date, end_date)

            # Save Solar
            solar_cols = ['timestamp', 'ghi', 'dni', 'cloud_cover', 'temperature']
            df[solar_cols].to_csv(os.path.join(folder_path, f"{name}_Solar.csv"), index=False)

            # Save Wind
            wind_cols = ['timestamp', 'wind_speed_10m', 'wind_speed_100m', 'wind_gusts', 'wind_direction']
            df[wind_cols].to_csv(os.path.join(folder_path, f"{name}_Wind.csv"), index=False)

            print(f"   -> Saved {name}")

            # 3. Add to memory
            all_data[name] = df

            time.sleep(10) #take time between API requests

        except Exception as e:
            print(f"FAIL on {name}: {e}")
            # If 429 hits, stop the loop to save IP
            if "429" in str(e) or "Client Error" in str(e):
                print("429 ERROR DETECTED. Stopping loop ")
                break

    print("Process Complete (or Stopped). Check output folder.")
    return all_data


# In[11]:


#Fetch Wind data
print("--- Fetching Wind Sites ---")
raw_wind_data = fetch_all_locations(wind_final, start_date, end_date)


# In[12]:


#Fetch Solar Data
print("\n--- Fetching Solar Sites ---")
raw_solar_data = fetch_all_locations(solar_final, start_date, end_date)

