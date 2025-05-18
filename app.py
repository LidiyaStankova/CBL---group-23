import os
import pandas as pd
import geopandas as gpd
import folium

import streamlit as st
from streamlit.components.v1 import html as st_html
import pydeck as pdk
import altair as alt

# ----- Configuration -----
DATA_PATH   = r"C:\Users\Gabri\Documents\GitHub\CBL---group-23\burglary_2021_2025.csv"
LSOA_SHP_DIR = r"C:\Users\Gabri\Documents\GitHub\CBL---group-23\LB_shp"

st.set_page_config(page_title="Burglaries Map (2021–2025)", layout="wide")

# ----- Cache loaders -----
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    # Parse DD/MM/YYYY correctly
    df = pd.read_csv(path, parse_dates=["Month"], dayfirst=True)
    return df.dropna(subset=["Latitude", "Longitude"])


@st.cache_data
def load_lsoa_boundaries(shp_dir: str):
    # read all borough LSOA shapefiles
    gdf_list = []
    for fname in os.listdir(shp_dir):
        if fname.lower().endswith(".shp"):
            full = os.path.join(shp_dir, fname)
            gdf_list.append(gpd.read_file(full))
    lsoa_gdf = pd.concat(gdf_list, ignore_index=True)
    # rename the code column to "LSOA11CD"
    candidates = [c for c in lsoa_gdf.columns if "LSOA" in c.upper() and "CD" in c.upper()]
    if "LSOA11CD" not in lsoa_gdf.columns and len(candidates) == 1:
        lsoa_gdf = lsoa_gdf.rename(columns={candidates[0]: "LSOA11CD"})
    return lsoa_gdf.to_crs(epsg=4326)

# ----- Load data -----
df_raw   = load_data(DATA_PATH)
lsoa_gdf = load_lsoa_boundaries(LSOA_SHP_DIR)

# create a join-field in burglary data
df_raw["LSOA11CD"] = df_raw["LSOA code"]

# restrict to London bounding box
df_raw = df_raw[
    df_raw["Longitude"].between(-0.55, 0.30) &
    df_raw["Latitude"].between(51.25, 51.70)
].copy()

# ----- Sidebar Mode -----
mode = st.sidebar.radio(
    "Display mode",
    ("All incidents", "By calendar month", "LSOAs")
)

# ----- Header -----
st.title("Burglaries in London (2021–2025)")
if mode == "All incidents":
    st.subheader("All burglaries (2021–2025)")
elif mode == "By calendar month":
    st.subheader("Filter by calendar month")
else:
    st.subheader("LSOA boundaries with burglary counts")

# ----- Month filtering -----
df = df_raw.copy()
if mode == "By calendar month":
    months = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]
    sel_month = st.sidebar.selectbox("Select month", months)
    idx = months.index(sel_month) + 1
    df = df[df["Month"].dt.month == idx]
    st.markdown(f"**Showing all {sel_month}s (2021–2025)**")

st.markdown(f"**Total incidents displayed:** {len(df):,}")

# ----- Scatter & Stats modes -----
if mode in ("All incidents", "By calendar month"):

    # Pydeck scatter
    mid_lat = df["Latitude"].mean()
    mid_lon = df["Longitude"].mean()
    scatter = pdk.Layer(
        "ScatterplotLayer", data=df,
        get_position=["Longitude","Latitude"],
        get_fill_color=[10,132,180,140],
        get_radius=10, pickable=True
    )
    view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=10)
    deck = pdk.Deck(
        layers=[scatter],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10",
        tooltip={
            "html": (
                "<b>Location:</b> {Location}<br/>"
                "<b>LSOA:</b> {LSOA name}<br/>"
                "<b>Date:</b> {Month:%Y-%m}"
            )
        }
    )
    st.pydeck_chart(deck)

    if mode == "All incidents":
        # KPI cards
        total = len(df)
        monthly = df.groupby(df["Month"].dt.to_period("M")).size()
        avg_pm = monthly.mean()
        peak_mo = monthly.idxmax().strftime("%Y-%m")
        yearly = df.groupby(df["Month"].dt.year).size()
        peak_yr = yearly.idxmax()
        unique_lsoas = df["LSOA code"].nunique()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total incidents", f"{total:,}")
        c2.metric("Avg incidents/month", f"{avg_pm:.1f}")
        c3.metric("Peak month", peak_mo)
        c4.metric("Peak year", peak_yr)
        c5.metric("Unique LSOAs", f"{unique_lsoas:,}")

        # Top 10 LSOAs
        st.markdown("### Top 10 LSOAs by incident count")
        top10 = df.groupby("LSOA name").size().nlargest(10).rename("Count")
        st.bar_chart(top10)

        # Seasonal bar chart
        st.markdown("### Seasonal Distribution of Burglaries by Month")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        df_season = (
            df.assign(MN=df["Month"].dt.month_name().str[:3])
              .groupby("MN").size()
              .reindex(month_order)
              .reset_index(name="Incidents")
        )
        bar = (
            alt.Chart(df_season)
               .mark_bar()
               .encode(
                   x=alt.X("MN:N", sort=month_order, axis=alt.Axis(labelAngle=0)),
                   y=alt.Y("Incidents:Q", axis=alt.Axis(format=",")),
                   tooltip=["MN","Incidents"]
               )
               .properties(width=700, height=300)
        )
        st.altair_chart(bar, use_container_width=True)

        # Box-plot of monthly totals
        st.markdown("### Distribution of Monthly Totals Across Years")
        df_ym = (
            df_raw.assign(Y=df_raw["Month"].dt.year, M=df_raw["Month"].dt.month)
                  .groupby(["Y","M"]).size()
                  .reset_index(name="Count")
                  .assign(Mo=lambda d: d["M"].apply(lambda m: month_order[m-1]))
        )
        box = (
            alt.Chart(df_ym)
               .mark_boxplot(extent="min-max", size=50)
               .encode(
                   x=alt.X("Mo:N", sort=month_order, axis=alt.Axis(labelAngle=0)),
                   y=alt.Y("Count:Q", axis=alt.Axis(format=",.0f")),
                   color=alt.Color("Mo:N", scale=alt.Scale(scheme="category20"), legend=None)
               )
               .properties(height=350)
        )
        st.altair_chart(box, use_container_width=True)

# ----- LSOAs mode: choropleth -----
else:
    # compute burglary counts per LSOA
    counts = df_raw.groupby("LSOA11CD").size().reset_index(name="Burglary_Count")
    merged = lsoa_gdf.merge(counts, on="LSOA11CD", how="left")
    merged["Burglary_Count"] = merged["Burglary_Count"].fillna(0).astype(int)

    # --- NEW: drop any extra cols and simplify geometry ---
    merged = merged[["LSOA11CD", "Burglary_Count", "geometry"]].copy()
    # tolerance is in degrees; try 0.001 (≈100 m) or even 0.002
    merged["geometry"] = merged["geometry"].simplify(tolerance=0.0001, preserve_topology=True)

    # Folium map
    center = [51.5074, -0.1278]
    m = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=merged,
        name="Burglary Count by LSOA",
        data=merged,
        columns=["LSOA11CD","Burglary_Count"],
        key_on="feature.properties.LSOA11CD",
        fill_color="YlOrRd", fill_opacity=0.7, line_opacity=0.2,
        legend_name="Total Burglaries (2021–2025)"
    ).add_to(m)
    folium.GeoJson(
        merged,
        style_function=lambda feat: {"fillOpacity":0, "color":"#444", "weight":0.5},
        tooltip=folium.features.GeoJsonTooltip(
            fields=["LSOA11CD","Burglary_Count"],
            aliases=["LSOA Code","Burglary Count"],
            localize=True
        )
    ).add_to(m)
    folium.LayerControl().add_to(m)

    st_html(m._repr_html_(), height=600)

