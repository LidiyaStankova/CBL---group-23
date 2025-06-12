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

@st.cache_data(show_spinner=False)
def prepare_monthly_comparison(_df_raw, _pred_df):
    # ---  HISTORICAL  -------------------------------------------------
    hist = _df_raw[_df_raw["Crime type"].str.lower() == "burglary"].copy()
    hist["Month"] = hist["Month"].dt.to_period("M").dt.to_timestamp()
    hist["Year"]  = hist["Month"].dt.year
    hist["Month_Num"] = hist["Month"].dt.month
    hist = hist[hist["Year"].between(2021, 2024)]

    hist_avg = (
        hist.groupby("Month_Num").size().div(4)          # 4 years → average
             .round(1)
             .reset_index(name="Historical_Avg")
    )

    # ---  PREDICTED  --------------------------------------------------
    pred = _pred_df.groupby("Month_Num")["Predicted_Burglaries"].sum().reset_index()
    pred = pred.rename(columns={"Predicted_Burglaries": "Predicted"})

    # ---  MERGE & LABELS  --------------------------------------------
    month_names = {
        1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
        7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"
    }
    lookup = (
        pd.DataFrame({"Month_Num": range(1, 13)})
        .merge(hist_avg, on="Month_Num", how="left")
        .merge(pred,      on="Month_Num", how="left")
    )
    lookup["Month_Name"] = lookup["Month_Num"].map(month_names)
    return lookup

st.warning("This page contains data that is predicted using several models. These predicted values should therefore be treated with caution and not be considered factual.")

df_raw       = load_data(const.DATA_PATH)
pred_df      = load_predicted(const.PRED_PATH)
borough_gdf  = load_borough_boundaries(const.GEO_ZIP_PATH, const.BOROUGH_IN_ZIP)
ward_gdf     = load_ward_boundaries(const.GEO_ZIP_PATH, const.WARD_IN_ZIP)
time_geojson, ward_colormap, wards_timesliced_df = load_ward_timegeojson()
pred_geojson, pred_colormap, preds_timesliced_df = compute_predicted_timegeojson(pred_df, ward_gdf)

st.markdown("## Compare Two Wards Over Time")

# choose actual vs predicted
line_source = st.radio(
    "Data source:",
    ("Actual", "Predicted"),
    horizontal=True,
    key="line_compare_source"
)

# two dropdowns for exactly two wards
wards = sorted(ward_gdf["Ward"].unique())
col1, col2 = st.columns(2)
ward1 = col1.selectbox("Ward 1", wards, index=0, key="ward_compare_1")
ward2 = col2.selectbox("Ward 2", wards, index=1, key="ward_compare_2")

# pick the right timesliced DataFrame and rename count column
if line_source == "Actual":
    df_plot = wards_timesliced_df.rename(
        columns={"Ward_Burglary_Count": "Count"}
    ).copy()
else:
    df_plot = preds_timesliced_df.rename(
        columns={"Ward_Burglary_Count": "Count"}
    ).copy()

# filter to the two selected wards
df_plot = df_plot[df_plot["Ward"].isin([ward1, ward2])]

# DROP the geometry column so Streamlit/Arrow can serialize it
df_plot = df_plot[["Ward", "Month", "Count"]]

# build and show an Altair line chart
chart = (
    alt.Chart(df_plot)
       .mark_line(point=True)
       .encode(
           x=alt.X("Month:T", title="Month"),
           y=alt.Y("Count:Q", title="Burglaries"),
           color=alt.Color("Ward:N", title="Ward"),
           tooltip=[
             alt.Tooltip("Ward:N"),
             alt.Tooltip("Month:T", title="Month"),
             alt.Tooltip("Count:Q", title="Count"),
           ]
       )
       .properties(
           width=800,
           height=400
       )
)
st.altair_chart(chart, use_container_width=True)

st.markdown("## Historical Average (2021 – 2024) vs. 12-Month Prediction (Mar-25 → Feb-26)")

monthly_lookup = prepare_monthly_comparison(df_raw, pred_df)

chart_df = monthly_lookup.melt(
    id_vars=["Month_Num", "Month_Name"],
    value_vars=["Historical_Avg", "Predicted"],
    var_name="Series",
    value_name="Burglaries"
).dropna()

monthly_lookup["Delta"] = (monthly_lookup["Predicted"] - monthly_lookup["Historical_Avg"]) / monthly_lookup["Historical_Avg"] * 100

chart_df = chart_df.merge(
    monthly_lookup[["Month_Num", "Delta"]],
    on="Month_Num",
    how="left"
)
chart_df["Delta"] = chart_df.apply(
    lambda row: f"{row['Delta']:.1f}%" if row["Series"] == "Predicted" else "",
    axis=1
)

base = (
    alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Month_Name:N",
                    sort=list(monthly_lookup["Month_Name"]),
                    title=None),
            y=alt.Y("Burglaries:Q", title="Burglaries"),
            color=alt.Color("Series:N", legend=None),
            tooltip=[
                "Series:N",
                "Burglaries:Q",
                alt.Tooltip("Delta:N", title="Delta")
            ]
        )
        .properties(width=350, height=300)   
)

chart = base.facet(column=alt.Column("Series:N", header=alt.Header(title="")))

st.altair_chart(chart, use_container_width=True)