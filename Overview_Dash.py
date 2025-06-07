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
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────
# 1) File‐paths (adjust these to wherever your files live)
# ─────────────────────────────────────────────────────────────────────
DATA_PATH         = r"C:\Users\Gabri\Documents\GitHub\CBL---group-23\burglary_2021_2025.csv"
PRED_PATH         = r"C:\Users\Gabri\Documents\GitHub\CBL---group-23\ward_burglary_predictions_12months.csv"
LSOA_SHP_DIR      = r"C:\Users\Gabri\Documents\GitHub\CBL---group-23\LB_shp"
GEO_ZIP_PATH      = r"C:\Users\Gabri\Downloads\statistical-gis-boundaries-london2.zip"
WARD_IN_ZIP       = "statistical-gis-boundaries-london/ESRI/London_Ward_CityMerged.shp"
BOROUGH_IN_ZIP    = "statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp"

st.set_page_config(page_title="Burglaries Overview (All‐in‐One)", layout="wide")

# ─────────────────────────────────────────────────────────────────────
# 2) Sidebar: Top‐level mode only
# ─────────────────────────────────────────────────────────────────────
mode = st.sidebar.radio(
    "Display mode",
    ("Overview",)  # only one mode for now
)

# ─────────────────────────────────────────────────────────────────────
# 3) Data loaders (cached)
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
# 4) Build “Ward × Month” GeoJSON for Actual data (cached)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_ward_timegeojson(_df_raw, _lsoa_gdf, _ward_gdf):
    df_burglary = _df_raw[_df_raw["Crime type"].str.lower() == "burglary"].copy()
    df_burglary["Month"] = df_burglary["Month"].dt.to_period("M").dt.to_timestamp()

    lsoa_month_counts = (
        df_burglary
        .groupby(["LSOA11CD", "Month"])
        .size()
        .reset_index(name="LSOA_Burglary_Count")
    )

    gdf_lsoa_counts = _lsoa_gdf.merge(
        lsoa_month_counts,
        on="LSOA11CD",
        how="left"
    )
    gdf_lsoa_counts["LSOA_Burglary_Count"] = (
        gdf_lsoa_counts["LSOA_Burglary_Count"]
        .fillna(0)
        .astype(int)
    )

    gdf_ward_small = _ward_gdf[["Ward", "geometry"]].copy()
    gdf_lsoa_counts = gpd.sjoin(
        gdf_lsoa_counts,
        gdf_ward_small,
        how="left",
        predicate="intersects"
    )

    ward_month = (
        gdf_lsoa_counts
        .drop(columns=["index_right", "geometry"], errors="ignore")
        .groupby(["Ward", "Month"])
        .agg({"LSOA_Burglary_Count": "sum"})
        .reset_index()
        .rename(columns={"LSOA_Burglary_Count": "Ward_Burglary_Count"})
    )

    all_months = ward_month["Month"].sort_values().unique()
    months_df = pd.DataFrame({"Month": all_months})
    months_df["key"] = 1

    gdf_ward_simple = _ward_gdf.copy()
    gdf_ward_simple["geometry"] = (
        gdf_ward_simple.geometry.simplify(tolerance=0.001, preserve_topology=True)
    )
    gdf_ward_simple["key"] = 1

    wards_cartesian = (
        gdf_ward_simple[["Ward", "geometry", "key"]]
        .merge(months_df, on="key")
        .drop(columns=["key"])
    )

    wards_timesliced = wards_cartesian.merge(
        ward_month,
        on=["Ward", "Month"],
        how="left"
    )
    wards_timesliced["Ward_Burglary_Count"] = (
        wards_timesliced["Ward_Burglary_Count"]
        .fillna(0)
        .astype(int)
    )

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

# 5) Build “Ward × Month” GeoJSON for Predicted data (cached)  ← REPLACE
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


# 6) Load data and precompute everything
df_raw       = load_data(DATA_PATH)
pred_df      = load_predicted(PRED_PATH)


pred_df = pred_df.rename(columns={
    "Predicted_Burglaries": "Predicted_Count"
})

lsoa_gdf     = load_lsoa_boundaries(LSOA_SHP_DIR)
ward_gdf     = load_ward_boundaries(GEO_ZIP_PATH, WARD_IN_ZIP)
borough_gdf  = load_borough_boundaries(GEO_ZIP_PATH, BOROUGH_IN_ZIP)


time_geojson, ward_colormap, wards_timesliced_df = compute_ward_timegeojson(df_raw, lsoa_gdf, ward_gdf)
pred_geojson, pred_colormap, preds_timesliced_df = compute_predicted_timegeojson(pred_df, ward_gdf)

# ─────────────────────────────────────────────────────────────────────
# 7) “Overview” (single‐page with in‐page table of contents)
# ─────────────────────────────────────────────────────────────────────
if mode == "Overview":
    # ——— 7A) Table of Contents (anchor links) ———
    st.markdown(
        """
        <div style="font-weight:600; margin-bottom:8px;">
          Jump to:
          <a href="#map" style="margin-right:12px; color:#e63946;">Map</a>
          <a href="#monthly" style="margin-right:12px; color:#e63946;">Yearly Ward Distribution</a>
          <a href="#borough" style="margin-right:12px; color:#e63946;">Top Boroughs</a>
          <a href="#compare" style="margin-right:12px; color:#e63946;">Compare Wards</a>
          <a href="#topwards" style="color:#e63946;">Top Wards by Month</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ——— 7B) Anchor + Map Section ———
    st.markdown('<a id="map"></a>', unsafe_allow_html=True)
    st.title("Overview: Ward‐level Time‐Slider")

    data_source = st.radio(
        "Map Data source:",
        ("Actual", "Predicted"),
        horizontal=True
    )

    st.subheader("Map ► Use the slider below to see how burglaries in each ward changed over time")

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

    # ——— 7C) Anchor + Yearly Ward‐Level Distribution Section ———
    st.markdown('<a id="monthly"></a>', unsafe_allow_html=True)
    st.markdown("## Yearly Ward‐Level Distribution of Burglary Counts (Box‐Plot)")

    selected_year = st.slider(
        "Select year to view ward‐level distribution:",
        min_value=2021,
        max_value=2025,
        value=2021,
        step=1
    )
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
        f"Distribution of Monthly Burglary Counts Across Wards in {selected_year}",
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

    # ——— 7D) Anchor + Top Boroughs Section ———
    st.markdown('<a id="borough"></a>', unsafe_allow_html=True)
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
    st.bar_chart(borough_counts.set_index("Borough")["Count"])
# ─────────────────────────────────────────────────────────────────────
# 7E) Anchor + Historical-vs-Predicted Monthly Comparison  ← NEW
# ─────────────────────────────────────────────────────────────────────
st.markdown('<a id="compare"></a>', unsafe_allow_html=True)
st.markdown("## Historical (2021 – 2024) Average vs. 12-Month Prediction (Mar-25 → Feb-26)")

# 7E-1) Prepare a look-up table (cached so it only runs once)
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
    pred = _pred_df.groupby("Month_Num")["Predicted_Count"].sum().reset_index()
    pred = pred.rename(columns={"Predicted_Count": "Predicted"})

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

monthly_lookup = prepare_monthly_comparison(df_raw, pred_df)

# 7E-2) Interactive widgets
sel_month = st.selectbox(
    "Select month to compare:",
    options=monthly_lookup["Month_Name"],
    index=0
)

row = monthly_lookup.loc[monthly_lookup["Month_Name"] == sel_month].iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("2021-2024 Avg", f"{row.Historical_Avg:,.1f}")
if pd.isna(row.Predicted):
    col2.metric("Predicted 25-26", "n/a")
    col3.write("")      
else:
    col2.metric("Predicted 25-26", f"{int(row.Predicted):,}")
    delta = (row.Predicted - row.Historical_Avg) / row.Historical_Avg * 100
    col3.metric("Δ vs. Avg", f"{delta:+.1f}%")

# 7E-3) Optional full-year bar chart  ← replace this whole block
st.subheader("Monthly Totals – Average vs. Prediction")
chart_df = monthly_lookup.melt(
    id_vars=["Month_Num", "Month_Name"],
    value_vars=["Historical_Avg", "Predicted"],
    var_name="Series",
    value_name="Burglaries"
).dropna()
base = (
    alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Month_Name:N",
                    sort=list(monthly_lookup["Month_Name"]),
                    title=None),
            y=alt.Y("Burglaries:Q", title="Burglaries"),
            color=alt.Color("Series:N", legend=None),
            tooltip=["Series:N", "Burglaries:Q"]
        )
        .properties(width=350, height=300)   
)

chart = base.facet(column=alt.Column("Series:N", header=alt.Header(title="")))

st.altair_chart(chart, use_container_width=True)


    # ——— 7F) Anchor + Top Wards by Month Section ———
st.markdown('<a id="topwards"></a>', unsafe_allow_html=True)
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

    st.write("#### Top 5 Wards")
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
