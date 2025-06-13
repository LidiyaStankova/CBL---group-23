import os
import pandas as pd
import geopandas as gpd
import folium
import branca
from shapely.geometry import mapping
from folium.plugins import TimestampedGeoJson
import numpy as np
import streamlit as st
from streamlit.components.v1 import html as st_html
import altair as alt
from datetime import datetime
import matplotlib.pyplot as plt
import other.constants as const

st.set_page_config(page_title="Burglaries Overview (All‐in‐One)", layout="wide")

# ─────────────────────────────────────────────────────────────────────
# 1) Data loaders (cached)
# ─────────────────────────────────────────────────────────────────────
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
    return df

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

# ─────────────────────────────────────────────────────────────────────
# 2) Build “Ward × Month” GeoJSON for Actual data (cached)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_ward_timegeojson():
    """
    Loads precomputed ward-level burglary data and generates GeoJSON + colormap.
    """
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
        ward_name = row["Ward"]
        color = colormap(count_val)
        popup_html = (
            f"<strong>{ward_name}</strong><br>"
            f"Month: {row['Month'].strftime('%Y-%m')}<br>"
            f"Burglaries: {count_val}"
        )
        features.append({
            "type": "Feature",
            "geometry": mapping(row["geometry"]),
            "properties": {
                "time": iso_time,
                "style": {
                    "color": color,
                    "fillColor": color,
                    "fillOpacity": 0.7,
                    "weight": 1
                },
                "popup": popup_html
            }
        })

    return {"type": "FeatureCollection", "features": features}, colormap, wards_timesliced

# ─────────────────────────────────────────────────────────────────────
# 3) Build “Ward × Month” GeoJSON for Predicted data (cached)  ← REPLACE
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_predicted_timegeojson(pred_df, _ward_gdf):
    """
    Build a month-by-month GeoJSON so that the Folium time-slider
    actually changes when you press play.
    """
    # ── 1) make sure the dataframe has one row per Ward × Month ─────────
    df = pred_df.copy()
    if "Predicted_Count" in df.columns:
        df = df.rename(columns={"Predicted_Count": "Ward_Burglary_Count"})
    elif "Predicted_Burglaries" in df.columns:
        df = df.rename(columns={"Predicted_Burglaries": "Ward_Burglary_Count"})
    else:
        raise KeyError("Need a column called Predicted_Count or Predicted_Burglaries")

    df["Month"] = df["Month"].dt.to_period("M").dt.to_timestamp()

    # ── 2) build a *complete* Ward × Month matrix so every ward exists every month ──
    all_months = np.sort(df["Month"].unique())
    months_df  = pd.DataFrame({"Month": all_months, "key": 1})

    ward_simple = _ward_gdf[["Ward", "geometry"]].copy()
    ward_simple["geometry"] = ward_simple.geometry.simplify(0.001, preserve_topology=True)
    ward_simple["key"] = 1

    cartesian = (
        ward_simple[["Ward", "geometry", "key"]]
        .merge(months_df, on="key")
        .drop(columns="key")
    )

    merged = cartesian.merge(
        df[["Ward", "Month", "Ward_Burglary_Count"]],
        on=["Ward", "Month"],
        how="left"
    )
    merged["Ward_Burglary_Count"] = merged["Ward_Burglary_Count"].fillna(0)

    # ── 3) GLOBAL colour scale so colours are comparable month-to-month ──
    vmin = float(merged["Ward_Burglary_Count"].min())
    vmax = float(merged["Ward_Burglary_Count"].max())
    colormap = branca.colormap.LinearColormap(
        colors=["#ffffcc", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"],
        vmin=vmin, vmax=vmax,
        caption="Ward-level Predicted Burglaries per month"
    )

    # ── 4) assemble one Feature **per Ward × Month** ─────────────────────
    features = []
    for _, row in merged.iterrows():
        iso_time = row["Month"].strftime("%Y-%m-%dT00:00:00")   # folium time-slider needs iso
        val      = float(row["Ward_Burglary_Count"])
        color    = colormap(val)

        features.append({
            "type": "Feature",
            "geometry": mapping(row["geometry"]),
            "properties": {
                "time": iso_time,
                "style": {
                    "color":      color,
                    "fillColor":  color,
                    "fillOpacity": 0.7,
                    "weight": 1
                },
                "popup": (
                    f"<strong>{row['Ward']}</strong><br>"
                    f"Month: {row['Month'].strftime('%Y-%m')}<br>"
                    f"Predicted: {val:,.0f}"
                )
            }
        })

    return {"type": "FeatureCollection", "features": features}, colormap, merged


# 5) Load data and precompute everything
df_raw       = load_data(const.DATA_PATH)
pred_df      = load_predicted(const.PRED_PATH)


pred_df = pred_df.rename(columns={
    "Predicted_Burglaries": "Predicted_Count"
})

lsoa_gdf     = load_lsoa_boundaries(const.LSOA_SHP_DIR)
ward_gdf     = load_ward_boundaries(const.GEO_ZIP_PATH, const.WARD_IN_ZIP)
borough_gdf  = load_borough_boundaries(const.GEO_ZIP_PATH, const.BOROUGH_IN_ZIP)

time_geojson, ward_colormap, wards_timesliced_df = load_ward_timegeojson()
pred_geojson, pred_colormap, preds_timesliced_df = compute_predicted_timegeojson(pred_df, ward_gdf)

st.warning("This page contains data that is predicted using several models. These predicted values should therefore be treated with caution and not be considered factual.")

# ─────────────────────────────────────────────────────────────────────
# 4) “Overview” (single‐page with in‐page table of contents)
# ─────────────────────────────────────────────────────────────────────
# ——— 4A) Map Section ———
st.title("Overview: Ward‐level Time‐Slider")

data_source = st.radio(
    "Map Data source:",
    ("Actual", "Predicted"),
    horizontal=True
)

selected_year = st.slider(
    "Select year to view ward‐level distribution:",
    min_value=2021,
    max_value=2025,
    value=2021,
    step=1
)

m_time = folium.Map(
    location=[51.5074, -0.1278],
    zoom_start=10,
    tiles="cartodbpositron"
)


geojson_data = time_geojson if data_source == "Actual" else pred_geojson
cmap         = ward_colormap  if data_source == "Actual" else pred_colormap

TimestampedGeoJson(
    data=geojson_data,
    transition_time=200,
    period="P1M",
    add_last_point=True,
    auto_play=False,
    loop=True,
    max_speed=1,
    loop_button=True,
    date_options="YYYY-MM",
    time_slider_drag_update=True
).add_to(m_time)
cmap.add_to(m_time)
folium.LayerControl().add_to(m_time)
st_html(m_time._repr_html_(), height=1200)

# ——— 4B) Yearly Ward‐Level Distribution Section ———
st.markdown("## Ward‐Level Distribution of Burglary Counts (Yearly)")
df_year = wards_timesliced_df[wards_timesliced_df["Month"].dt.year == selected_year].copy()
df_year["Month_Num"] = df_year["Month"].dt.month
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
df_year["Month_Name"] = df_year["Month_Num"].apply(lambda m: month_labels[m - 1])

monthly_groups = [
    df_year.loc[df_year["Month_Num"] == m, "Ward_Burglary_Count"].values
    for m in range(1, 13)
]

fig, ax = plt.subplots(figsize=(12, 6))
boxprops = dict(linewidth=1.2)
medianprops = dict(linewidth=2.0, color="black")

bp = ax.boxplot(
    monthly_groups,
    patch_artist=True,
    boxprops=boxprops,
    medianprops=medianprops
)
colors = plt.cm.tab20.colors[:12]
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)

ax.set_title(
    f"Monthly Burglary Counts Across Wards in {selected_year}",
    fontsize=16,
    pad=15
)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Burglary Count (per Ward)", fontsize=12)
ax.set_xticklabels(month_labels, fontsize=10)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()
st.pyplot(fig)

st.caption(
    f"Each box shows how burglary counts vary across wards in {selected_year}. "
    "E.g., the “Jan” box represents every ward’s January count for that year."
)

    # ——— 4C) Top Wards by Month Section ———
st.markdown("## Top 5 Wards for a Selected Month (2021–2025)")


top5_source = st.radio(
    "Data source:",
    ("Actual", "Predicted"),
    horizontal=True,
    key="top5_data_source"
)


month_names_full = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
name_to_num = {name: i+1 for i, name in enumerate(month_names_full)}
selected_month_full = st.selectbox(
    "Select month:",
    month_names_full,
    index=0,
    key="top_wards_month"
)
selected_month_num = name_to_num[selected_month_full]

if top5_source == "Actual":

    pts = gpd.GeoDataFrame(
        df_raw.assign(Month_Num=df_raw.Month.dt.month),
        geometry=gpd.points_from_xy(df_raw.Longitude, df_raw.Latitude),
        crs="EPSG:4326"
    )
    pts = pts[pts["Month_Num"] == selected_month_num]
    joined = gpd.sjoin(
        pts,
        ward_gdf[["Ward", "geometry"]],
        how="left",
        predicate="within"
    )
    counts = (
        joined
        .groupby("Ward")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    label = f"Total {selected_month_full} (Actual)"

else:  # Predicted
    df_sel = pred_df[pred_df["Month"].dt.month == selected_month_num]
    counts = (
        df_sel
        .groupby("Ward")["Predicted_Count"]
        .sum()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    label = f"Total {selected_month_full} (Predicted next 12 m)"

# Top 5 table
top5 = counts.head(5).copy()
if top5.empty:
    st.write(f"No {top5_source.lower()} data found for {selected_month_full}.")
else:
    top5["Rank"] = range(1, len(top5) + 1)

    top5 = top5[["Rank", "Ward", "Count"]]
    top5 = top5.rename(columns={"Ward": "Ward_Name", "Count": label})

    st.dataframe(top5, use_container_width=True)

    bar = (
        alt.Chart(top5)
           .mark_bar()
           .encode(
               x=alt.X(f"{label}:Q", title=label),
               y=alt.Y("Ward_Name:N", sort="-x", title="Ward"),
               tooltip=["Ward_Name", f"{label}:Q"]
           )
           .properties(width=700, height=300)
    )
    st.altair_chart(bar, use_container_width=True)

# 4D) Top 5 Boroughs by Total Burglaries

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
