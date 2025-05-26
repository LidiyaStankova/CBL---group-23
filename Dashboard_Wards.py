import os
import pandas as pd
import geopandas as gpd
import folium

import streamlit as st
from streamlit.components.v1 import html as st_html
import pydeck as pdk
import altair as alt

# Configuration
DATA_PATH      = r"C:\Users\Gabri\Documents\GitHub\CBL---group-23\burglary_2021_2025.csv"
LSOA_SHP_DIR   = r"C:\Users\Gabri\Documents\GitHub\CBL---group-23\LB_shp"
GEO_ZIP_PATH   = r"C:\Users\Gabri\Downloads\statistical-gis-boundaries-london2.zip"
WARD_IN_ZIP    = "statistical-gis-boundaries-london/ESRI/London_Ward_CityMerged.shp"
BOROUGH_IN_ZIP = "statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp"

st.set_page_config(page_title="Burglaries Map (2021–2025)", layout="wide")

# Data loaders
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

# Load data and shapes
df_raw      = load_data(DATA_PATH)
lsoa_gdf    = load_lsoa_boundaries(LSOA_SHP_DIR)
ward_gdf    = load_shape_from_zip(GEO_ZIP_PATH, WARD_IN_ZIP)[["NAME", "geometry"]].rename(columns={"NAME": "Ward"})
borough_gdf = load_shape_from_zip(GEO_ZIP_PATH, BOROUGH_IN_ZIP)[["NAME", "geometry"]].rename(columns={"NAME": "Borough"})

df_raw = df_raw[
    df_raw.Longitude.between(-0.55, 0.30) &
    df_raw.Latitude.between(51.25, 51.70)
].copy()

# Mode selection
mode = st.sidebar.radio("Display mode",
    ("All incidents", "By calendar month", "Aggregated by geography", "Wards")
)

# Title and subtitle
st.title("Burglaries in London (2021–2025)")
if mode == "All incidents":
    st.subheader("All burglaries (2021–2025)")
elif mode == "By calendar month":
    st.subheader("Filter by month")
elif mode == "Aggregated by geography":
    st.subheader("Choropleth view")
else:
    st.subheader("Ward explorer")

# Month filter
df = df_raw.copy()
if mode == "By calendar month":
    months = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]
    sel_month = st.sidebar.selectbox("Select month", months)
    df = df[df.Month.dt.month == months.index(sel_month) + 1]
    st.markdown(f"**Showing all {sel_month}s (2021–2025)**")

# Point maps and charts
if mode in ("All incidents", "By calendar month"):
    st.markdown(f"**Total incidents displayed:** {len(df):,}")

    mid_lat, mid_lon = df.Latitude.mean(), df.Longitude.mean()
    scatter = pdk.Layer("ScatterplotLayer", data=df,
                        get_position=["Longitude", "Latitude"],
                        get_fill_color=[10, 132, 180, 140],
                        get_radius=10, pickable=True)
    view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=10)
    deck = pdk.Deck(layers=[scatter],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/light-v10",
                    tooltip={"html":
                             "<b>Location:</b> {Location}<br/>"
                             "<b>LSOA:</b> {LSOA name}<br/>"
                             "<b>Date:</b> {Month:%Y-%m}"})
    st.pydeck_chart(deck)

    if mode == "All incidents":
        total = len(df)
        monthly = df.groupby(df.Month.dt.to_period("M")).size()
        avg_pm = monthly.mean()
        peak_mo = monthly.idxmax().strftime("%Y-%m")
        yearly = df.groupby(df.Month.dt.year).size()
        peak_yr = yearly.idxmax()
        unique_lsoas = df["LSOA code"].nunique()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total incidents", f"{total:,}")
        c2.metric("Avg incidents/month", f"{avg_pm:.1f}")
        c3.metric("Peak month", peak_mo)
        c4.metric("Peak year", peak_yr)
        c5.metric("Unique LSOAs", f"{unique_lsoas:,}")

        st.markdown("### Top 10 LSOAs by incident count")
        top10 = df.groupby("LSOA name").size().nlargest(10).rename("Count")
        st.bar_chart(top10)

        st.markdown("### Seasonal distribution")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        df_season = (
            df.assign(MN=df.Month.dt.month_name().str[:3])
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
                   tooltip=["MN", "Incidents"]
               )
               .properties(width=700, height=300)
        )
        st.altair_chart(bar, use_container_width=True)

        st.markdown("### Monthly totals distribution")
        df_ym = (
            df_raw.assign(Y=df_raw.Month.dt.year, M=df_raw.Month.dt.month)
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
                   color=alt.Color("Mo:N", legend=None)
               )
               .properties(height=350)
        )
        st.altair_chart(box, use_container_width=True)

# Ward view
elif mode == "Wards":
    ward_list = ward_gdf.Ward.sort_values().unique()
    sel_ward  = st.sidebar.selectbox("Choose a ward", ward_list)
    ward_geom = ward_gdf.loc[ward_gdf.Ward == sel_ward, "geometry"].iloc[0]

    pts = gpd.GeoDataFrame(
        df_raw,
        geometry=gpd.points_from_xy(df_raw.Longitude, df_raw.Latitude),
        crs="EPSG:4326"
    )
    pts_in = pts[pts.within(ward_geom)].copy()

    joined = gpd.sjoin(pts_in, lsoa_gdf[["LSOA11CD","geometry"]],
                       how="left", predicate="within")

    counts = joined.groupby("index_right").size().reset_index(name="Burglary_Count")
    counts["LSOA11CD"] = counts["index_right"].map(lsoa_gdf.LSOA11CD)

    lsoas_in = lsoa_gdf[lsoa_gdf.geometry.centroid.within(ward_geom)].copy()
    display_gdf = (
        lsoas_in
        .merge(counts[["LSOA11CD","Burglary_Count"]], on="LSOA11CD", how="left")
        .fillna({"Burglary_Count": 0})
    )
    display_gdf["Burglary_Count"] = display_gdf["Burglary_Count"].astype(int)

    m2 = folium.Map(
        location=[
            display_gdf.geometry.centroid.y.mean(),
            display_gdf.geometry.centroid.x.mean()
        ],
        zoom_start=12,
        tiles="cartodbpositron"
    )
    folium.GeoJson(ward_geom, style_function=lambda f: {"fillOpacity":0,"color":"black","weight":2}).add_to(m2)
    folium.Choropleth(
        geo_data=display_gdf,
        data=display_gdf,
        columns=["LSOA11CD","Burglary_Count"],
        key_on="feature.properties.LSOA11CD",
        fill_color="YlGnBu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"Burglary count in {sel_ward} (2021–2025)"
    ).add_to(m2)
    folium.GeoJson(
        display_gdf,
        style_function=lambda f: {"fillOpacity":0},
        tooltip=folium.features.GeoJsonTooltip(
            fields=["LSOA11CD","Burglary_Count"],
            aliases=["LSOA code","Count"],
            localize=True
        )
    ).add_to(m2)
    folium.LayerControl().add_to(m2)

    st.subheader(f"LSOAs in {sel_ward}")
    st_html(m2._repr_html_(), height=600)

# Geography aggregation
else:
    unit = st.sidebar.selectbox("Level", ["LSOA","Ward","Borough"])

    if unit == "LSOA":
        base    = lsoa_gdf
        grp     = df_raw.groupby("LSOA11CD").size().reset_index(name="Burglary_Count")
        geo_key = "LSOA11CD"
    elif unit == "Ward":
        pts = gpd.GeoDataFrame(
            df_raw,
            geometry=gpd.points_from_xy(df_raw.Longitude, df_raw.Latitude),
            crs="EPSG:4326"
        )
        joined = gpd.sjoin(pts, ward_gdf, how="left", predicate="within")
        grp    = joined.groupby("Ward").size().reset_index(name="Burglary_Count")
        base   = ward_gdf
        geo_key= "Ward"
    else:
        pts    = gpd.GeoDataFrame(
            df_raw,
            geometry=gpd.points_from_xy(df_raw.Longitude, df_raw.Latitude),
            crs="EPSG:4326"
        )
        joined = gpd.sjoin(pts, borough_gdf, how="left", predicate="within")
        grp    = joined.groupby("Borough").size().reset_index(name="Burglary_Count")
        base   = borough_gdf
        geo_key= "Borough"

    merged = base.merge(grp, on=geo_key, how="left")
    merged["Burglary_Count"] = merged["Burglary_Count"].fillna(0).astype(int)
    merged["geometry"] = merged.geometry.simplify(tolerance=0.0001, preserve_topology=True)

    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=merged,
        name=f"{unit} choropleth",
        data=merged,
        columns=[geo_key,"Burglary_Count"],
        key_on=f"feature.properties.{geo_key}",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"Total Burglaries by {unit} (2021–2025)"
    ).add_to(m)
    folium.GeoJson(
        merged,
        style_function=lambda f: {"fillOpacity":0,"color":"#444","weight":0.5},
        tooltip=folium.features.GeoJsonTooltip(
            fields=[geo_key,"Burglary_Count"],
            aliases=[unit,"Count"],
            localize=True
        )
    ).add_to(m)
    folium.LayerControl().add_to(m)
    st_html(m._repr_html_(), height=600)
