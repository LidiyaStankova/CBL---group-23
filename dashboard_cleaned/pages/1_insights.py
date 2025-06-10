import streamlit as st
from streamlit.components.v1 import html as st_html
import pandas as pd
import geopandas as gpd
import folium
import os
from branca.element import Element, MacroElement
from folium import MacroElement
from branca.element import Template, MacroElement
import other.constants as const

st.set_page_config(page_title="Force Allocation", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Month"], dayfirst=True)
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["LSOA11CD"] = df["LSOA code"]
    df = df[
        df.Longitude.between(-0.55, 0.30) &
        df.Latitude.between(51.25, 51.70)
    ].copy()
    return df

@st.cache_data(show_spinner=False)
def load_predicted(path):
    df = pd.read_csv(path)
    df["Month"] = pd.to_datetime(df["Month"], dayfirst=False)

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

@st.cache_data(show_spinner=False)
def load_ward_boundaries(zip_path, inside_path):
    gdf = gpd.read_file(f"zip://{zip_path}!{inside_path}")
    return gdf[["NAME", "geometry"]].rename(columns={"NAME": "Ward"}).to_crs(epsg=4326)

@st.cache_data(show_spinner=False)
def load_borough_boundaries(zip_path, inside_path):
    gdf = gpd.read_file(f"zip://{zip_path}!{inside_path}")
    return gdf[["NAME", "geometry"]].rename(columns={"NAME": "Borough"}).to_crs(epsg=4326)



df_raw       = load_data(const.DATA_PATH)
pred_df      = load_predicted(const.PRED_PATH)

lsoa_gdf     = load_lsoa_boundaries(const.LSOA_SHP_DIR)
ward_gdf     = load_ward_boundaries(const.GEO_ZIP_PATH, const.WARD_IN_ZIP)
borough_gdf  = load_borough_boundaries(const.GEO_ZIP_PATH, const.BOROUGH_IN_ZIP)

st.markdown("## Top 10 Boroughs by Total Burglaries (2021–2025)")
pts = gpd.GeoDataFrame(
    df_raw,
    geometry=gpd.points_from_xy(df_raw.Longitude, df_raw.Latitude),
    crs="EPSG:4326"
)
joined_by_borough = gpd.sjoin(pts, borough_gdf, how="left", predicate="within")
borough_counts = (
    joined_by_borough
    .groupby("Borough")
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
    .head(10)
)
st.bar_chart(borough_counts.set_index("Borough")["Count"].sort_values(ascending=False))