import streamlit as st
from streamlit.components.v1 import html as st_html
import pandas as pd
import geopandas as gpd
import folium
import os

st.set_page_config(page_title="Predictions", layout="wide")

DATA_PATH      = r"C:/Users/20233537/OneDrive - TU Eindhoven/Documents/Homework/y2/CBL Multidisciplinary/CBL 2025 burglary/Github/CBL---group-23/burglary_2021_2025.csv"
LSOA_SHP_DIR   = r"C:/Users/20233537/OneDrive - TU Eindhoven/Documents/Homework/y2/CBL Multidisciplinary/CBL 2025 burglary/Github/CBL---group-23/dashboard/LB_shp"
GEO_ZIP_PATH   = r"C:/Users/20233537/OneDrive - TU Eindhoven/Documents/Homework/y2/CBL Multidisciplinary/CBL 2025 burglary/Github/CBL---group-23/dashboard/stats/statistical-gis-boundaries-london.zip"
WARD_IN_ZIP    = "statistical-gis-boundaries-london/ESRI/London_Ward_CityMerged.shp"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Month"], dayfirst=True)
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["LSOA11CD"] = df["LSOA code"]
    return df

@st.cache_data
def load_lsoa_boundaries(shp_dir):
    gdfs = [gpd.read_file(os.path.join(shp_dir, f))
            for f in os.listdir(shp_dir) if f.lower().endswith(".shp")]
    lsoa = pd.concat(gdfs, ignore_index=True)
    cands = [c for c in lsoa.columns if "LSOA" in c.upper() and c.upper().endswith("CD")]
    lsoa = lsoa.rename(columns={cands[0]: "LSOA11CD"})
    return lsoa.to_crs(epsg=4326)

@st.cache_data
def load_shape_from_zip(zip_path, inside_path):
    gdf = gpd.read_file(f"zip://{zip_path}!{inside_path}")
    return gdf.to_crs(epsg=4326)

def load_prediction_data(month, ward=None):
    file_path = fr"C:/Users/20233537/OneDrive - TU Eindhoven/Documents/Homework/y2/CBL Multidisciplinary/CBL 2025 burglary/Github/CBL---group-23/dashboard/stats/allocation/2026/{ward}/{month}.csv"
    return pd.read_csv(file_path)

df_raw      = load_data(DATA_PATH)
lsoa_gdf    = load_lsoa_boundaries(LSOA_SHP_DIR)
ward_gdf    = load_shape_from_zip(GEO_ZIP_PATH, WARD_IN_ZIP)[["NAME", "geometry"]].rename(columns={"NAME": "Ward"})

# Layout for dropdowns side by side
col1, col2 = st.columns(2)

with col1:
    month_options = ["january", "february", "march", "april", "may", "june",
                     "july", "august", "september", "october", "november", "december"]
    selected_month = st.selectbox("Select month:", month_options)

with col2:
    ward_list = ward_gdf.Ward.sort_values().unique()
    selected_ward = st.selectbox("Select ward:", ward_list)

ward_geom = ward_gdf.loc[ward_gdf.Ward == selected_ward, "geometry"].iloc[0]

pts = gpd.GeoDataFrame(
        df_raw,
        geometry=gpd.points_from_xy(df_raw.Longitude, df_raw.Latitude),
        crs="EPSG:4326"
    )

pts_in = pts[pts.within(ward_geom)].copy()

joined = gpd.sjoin(pts_in, lsoa_gdf[["LSOA11CD","geometry"]],
                    how="left", predicate="within")

hours = load_prediction_data(selected_month, selected_ward)

hours.columns = hours.columns.str.strip().str.upper()

if "LSOA" in hours.columns:
    hours = hours.rename(columns={"LSOA": "LSOA11CD"})

lsoas_in = lsoa_gdf[lsoa_gdf.geometry.centroid.within(ward_geom)].copy()

print(lsoas_in['LSOA11CD'].unique().tolist())

display_gdf = (
    lsoas_in
    .merge(hours[["LSOA11CD","PATROL_HOURS"]], on="LSOA11CD", how="left")
    .fillna({"PATROL_HOURS": 0})
)
display_gdf["PATROL_HOURS"] = display_gdf["PATROL_HOURS"].astype(int)

m2 = folium.Map(
    location=[0, 0],  # temporary; will be overridden by fit_bounds
    zoom_start=13,
    tiles="cartodbpositron"
)

# Fit the map to the exact bounds of the selected ward
bounds = ward_geom.bounds  # (minx, miny, maxx, maxy)
m2.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

folium.GeoJson(ward_geom, style_function=lambda f: {"fillOpacity":0,"color":"black","weight":2}).add_to(m2)
folium.Choropleth(
    geo_data=display_gdf,
    data=display_gdf,
    columns=["LSOA11CD","PATROL_HOURS"],
    key_on="feature.properties.LSOA11CD",
    fill_color="YlGnBu",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=f"Predicted patrol hours in {selected_ward}"
).add_to(m2)
folium.GeoJson(
    display_gdf,
    style_function=lambda f: {"fillOpacity":0},
    highlight_function=lambda f: {
        "fillOpacity": 0.1,
        "color": None,           # removes border on highlight
        "weight": 0              # no border thickness
    },
    tooltip=folium.features.GeoJsonTooltip(
        fields=["LSOA11CD","PATROL_HOURS"],
        aliases=["LSOA code","Needed patrol hours"],
        localize=True
    )
).add_to(m2)
folium.LayerControl().add_to(m2)

st.subheader(f"LSOAs in {selected_ward}")
st_html(m2._repr_html_(), height=600)

df_pred = load_prediction_data(selected_month, selected_ward)

# 🔧 Clean column names
df_pred.columns = df_pred.columns.str.strip().str.lower()

df_pred = df_pred[["lsoa", "patrol_hours"]].rename(
    columns={"lsoa": "LSOA", "patrol_hours": "Patrol Hours"}
)

# Header
st.title("📈 Police force allocation")
st.markdown("This page shows the allocation of police patrol hours for upcoming months based on historical data.")

# Display table
st.dataframe(df_pred, use_container_width=True)

# Optional: Chart
if "Year" in df_pred.columns and "Month" in df_pred.columns:
    df_pred["YM"] = df_pred["Year"].astype(str) + "-" + df_pred["Month"]
    chart_data = df_pred.copy()
    chart_data["YM"] = pd.to_datetime(chart_data["YM"])
    st.line_chart(chart_data.set_index("YM")["Predicted Incidents"])
