# Import required libraries
import streamlit as st
from streamlit.components.v1 import html as st_html
import pandas as pd
import geopandas as gpd
import folium
import altair as alt
import os
import branca
from branca.element import Element, MacroElement
from folium import MacroElement
from branca.element import Template, MacroElement
import other.constants as const
from shapely.geometry import mapping
import numpy as np

# Configure Streamlit layout
st.set_page_config(page_title="Force Allocation", layout="wide")


# Load and filter raw burglary data
@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Month"], dayfirst=True)
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["LSOA11CD"] = df["LSOA code"]
    df = df[df.Longitude.between(-0.55, 0.30) & df.Latitude.between(51.25, 51.70)].copy()
    return df

# Load predicted burglary counts
@st.cache_data(show_spinner=False)
def load_predicted(path):
    df = pd.read_csv(path)
    df["Month"] = pd.to_datetime(df["Month"], dayfirst=False)
    return df

# Load LSOA shapefiles from directory
@st.cache_data(show_spinner=False)
def load_lsoa_boundaries(shp_dir):
    shp_files = [f for f in os.listdir(shp_dir) if f.lower().endswith(".shp")]
    gdfs = []
    for fname in shp_files:
        full = os.path.join(shp_dir, fname)
        try:
            g = gpd.read_file(full)
            gdfs.append(g)
        except Exception:
            continue
    lsoa = pd.concat(gdfs, ignore_index=True)
    code_cols = [c for c in lsoa.columns if "LSOA" in c.upper() and c.upper().endswith("CD")]
    if code_cols:
        lsoa = lsoa.rename(columns={code_cols[0]: "LSOA11CD"})
    return lsoa.to_crs(epsg=4326)

# Load ward boundaries from a zipped shapefile
@st.cache_data(show_spinner=False)
def load_ward_boundaries(zip_path, inside_path):
    gdf = gpd.read_file(f"zip://{zip_path}!{inside_path}")
    return gdf[["NAME", "geometry"]].rename(columns={"NAME": "Ward"}).to_crs(epsg=4326)

# Load borough boundaries from a zipped shapefile
@st.cache_data(show_spinner=False)
def load_borough_boundaries(zip_path, inside_path):
    gdf = gpd.read_file(f"zip://{zip_path}!{inside_path}")
    return gdf[["NAME", "geometry"]].rename(columns={"NAME": "Borough"}).to_crs(epsg=4326)

# Load ward-level monthly burglary counts and build colormap
@st.cache_data(show_spinner=False)
def load_ward_timegeojson():
    feather_path = const.PROCESSED_DATA_PATH + "/ward_burglary.feather"
    wards_timesliced = gpd.read_feather(feather_path)

    vmin = int(wards_timesliced["Ward_Burglary_Count"].min())
    vmax = int(wards_timesliced["Ward_Burglary_Count"].max())
    colormap = branca.colormap.LinearColormap(
        colors=["#ffffcc", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"],
        vmin=vmin,
        vmax=vmax,
        caption="Ward‐level Burglary Count"
    )

    features = []
    for _, row in wards_timesliced.iterrows():
        iso_time = row["Month"].strftime("%Y-%m-%dT00:00:00")
        count_val = int(row["Ward_Burglary_Count"])

        feature = {
            "type": "Feature",
            "geometry": mapping(row["geometry"]),
            "properties": {
                "Ward": row["Ward"],
                "Month": iso_time,
                "Ward_Burglary_Count": count_val
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    return geojson, colormap
