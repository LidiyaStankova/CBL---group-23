import os
import pandas as pd
import geopandas as gpd
import folium

import streamlit as st
from streamlit.components.v1 import html as st_html
import pydeck as pdk
import altair as alt

# ----- Configuration -----
PREPROCESSED_DATA_PATH = "stats\processed_burglaries.parquet"
PREPROCESSED_LSOA_PATH = "stats\processed_lsoas.geojson"

st.set_page_config(page_title="Burglaries Map (2021–2025)", layout="wide")

# ----- Cache data -----
@st.cache_data
def load_data():
    return pd.read_parquet(PREPROCESSED_DATA_PATH)

@st.cache_data
def load_lsoas():
    return gpd.read_file(PREPROCESSED_LSOA_PATH)

@st.cache_data
def load_all_stats():
    monthly = pd.read_parquet("stats/monthly_totals.parquet")
    yearly = pd.read_parquet("stats/yearly_totals.parquet")
    top10 = pd.read_parquet("stats/top10_lsoas.parquet").set_index("LSOA name")
    seasonal = pd.read_parquet("stats/seasonal_months.parquet")
    boxplot = pd.read_parquet("stats/boxplot_monthly.parquet")
    return monthly, yearly, top10, seasonal, boxplot

# ----- Load data -----
df_raw   = load_data()
lsoa_gdf = load_lsoas()
monthly, yearly, top10, df_season, df_ym = load_all_stats()

# create a join-field in burglary data
df_raw["LSOA11CD"] = df_raw["LSOA code"]

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
df = []
if mode == "By calendar month":
    month_names = pd.date_range("2000-01-01", periods=12, freq="MS").strftime("%B").tolist()
    sel_month = st.sidebar.selectbox("Select month", month_names)

    # Use integer filtering instead of string (faster comparison)
    month_num = month_names.index(sel_month) + 1
    df = df_raw[df_raw["Month"].dt.month == month_num].copy()
    
    st.markdown(f"**Showing all {sel_month}s (2021–2025)**")
else:
    df = df_raw

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

        total = monthly["Total"].sum()
        avg_pm = monthly["Total"].mean()
        peak_mo = monthly.loc[monthly["Total"].idxmax(), "Month_Period"].strftime("%Y-%m")
        peak_yr = yearly.loc[yearly["Total"].idxmax(), "Year"]
        unique_lsoas = df["LSOA code"].nunique()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total incidents", f"{total:,}")
        c2.metric("Avg incidents/month", f"{avg_pm:.1f}")
        c3.metric("Peak month", peak_mo)
        c4.metric("Peak year", peak_yr)
        c5.metric("Unique LSOAs", f"{unique_lsoas:,}")

        # Top 10 LSOAs
        st.markdown("### Top 10 LSOAs by incident count")
        top10 = top10
        st.bar_chart(top10)

        # Seasonal bar chart
        st.markdown("### Seasonal Distribution of Burglaries by Month")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
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
    # Pre-aggregate only necessary burglary counts
    counts = df_raw["LSOA11CD"].value_counts().rename_axis("LSOA11CD").reset_index(name="Burglary_Count")

    # Merge with GeoDataFrame, avoid unnecessary columns and .copy()
    merged = lsoa_gdf[["LSOA11CD", "geometry"]].merge(counts, on="LSOA11CD", how="left")
    merged["Burglary_Count"] = merged["Burglary_Count"].fillna(0).astype(int)

    # Convert to GeoJSON string only once (avoid Python object overhead in folium)
    geojson_str = merged.to_json()

    # Folium setup
    center = [51.5074, -0.1278]
    m = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")

    # Choropleth (faster if GeoJSON is pre-converted)
    folium.Choropleth(
        geo_data=geojson_str,
        name="Burglary Count by LSOA",
        data=merged[["LSOA11CD", "Burglary_Count"]],
        columns=["LSOA11CD", "Burglary_Count"],
        key_on="feature.properties.LSOA11CD",
        fill_color="YlOrRd", fill_opacity=0.7, line_opacity=0.2,
        legend_name="Total Burglaries (2021–2025)"
    ).add_to(m)

    # Lightweight GeoJson for tooltips
    folium.GeoJson(
        geojson_str,
        style_function=lambda _: {"fillOpacity": 0, "color": "#444", "weight": 0.5},
        tooltip=folium.GeoJsonTooltip(
            fields=["LSOA11CD", "Burglary_Count"],
            aliases=["LSOA Code", "Burglary Count"],
            localize=True
        )
    ).add_to(m)

    folium.LayerControl().add_to(m)

    st_html(m._repr_html_(), height=600)


