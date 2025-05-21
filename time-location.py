import pandas as pd
from datetime import timedelta
import os
import numpy as np

def lsoa():
    # Set the path and file name
    data_folder = r'C:/Users/1940120/OneDrive - TU Eindhoven/Documents/Homework/y2/CBL Multidisciplinary/CBL 2025 burglary/Github/CBL---group-23'
    filename = 'all_burglary_2020+.csv'
    file_path = os.path.join(data_folder, filename)

    # Load the data
    df = pd.read_csv(file_path, parse_dates=['Month'])  # expects 'Month', 'LSOA code'

    # Sort data by LSOA code and Month
    df.sort_values(by=['LSOA code', 'Month'], inplace=True)

    # Time window for repeat definition
    repeat_window = timedelta(days=30)

    # Calculate the difference between consecutive rows within each LSOA group
    df['prev_month'] = df.groupby('LSOA code')['Month'].shift(1)
    df['time_diff'] = df['Month'] - df['prev_month']

    # Create a new column to flag repeats within the time window
    df['repeat_within_30_days'] = (df['time_diff'] <= repeat_window) & df['time_diff'].notna()

    # Report the repeat burglary rate
    repeat_rate = df['repeat_within_30_days'].mean()
    print(f"Repeat burglary rate within 30 days (by LSOA): {repeat_rate:.2%}")

def coordinate():
    # Set the path and file name
    data_folder = r'C:/Users/1940120/OneDrive - TU Eindhoven/Documents/Homework/y2/CBL Multidisciplinary/CBL 2025 burglary/Github/CBL---group-23'
    filename = 'all_burglary_2020+.csv'
    file_path = os.path.join(data_folder, filename)

    # Load the data
    df = pd.read_csv(file_path, parse_dates=['Month'])  # expects 'Month', 'latitude', 'longitude'

    # Sort data by LSOA code, Month, and location
    df.sort_values(by=['LSOA code', 'Month'], inplace=True)

    # Time window for repeat definition
    repeat_window = timedelta(days=30)
    distance_threshold_km = 0.1  # 100 meters

    # Shift latitude, longitude, and date to compare with previous entry
    df['prev_lat'] = df.groupby('LSOA code')['Latitude'].shift(1)
    df['prev_lon'] = df.groupby('LSOA code')['Longitude'].shift(1)
    df['prev_month'] = df.groupby('LSOA code')['Month'].shift(1)

    # Calculate the time difference between consecutive events
    df['time_diff'] = df['Month'] - df['prev_month']

    # Calculate the distance between consecutive events using Haversine formula
    df['distance'] = df.apply(lambda row: haversine(row['latitude'], row['longitude'], row['prev_lat'], row['prev_lon']) 
                            if pd.notna(row['prev_lat']) else None, axis=1)

    # Mark as repeat if within both time window and distance threshold
    df['repeat_within_30_days'] = (df['time_diff'] <= repeat_window) & \
                                (df['distance'] <= distance_threshold_km) & \
                                df['time_diff'].notna()

    # Report repeat burglary rate
    repeat_rate = df['repeat_within_30_days'].mean()
    print(f"Repeat burglary rate within 30 days and 100 meters (by LSOA): {repeat_rate:.2%}")

# Haversine function to calculate the distance between two lat/long points
def haversine(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Distance in kilometers
    return R * c

coordinate()