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

def load_prediction_data(month, year, ward=None):
    file_path = const.ALLOCATION_PATH + fr"/{year}/{ward}/{month}.csv"
    return pd.read_csv(file_path)

required_columns = [
    "LSOA11CD", "patrol_hours", "total_burglaries", "total_burglaries_cur_year",
    "average_burglaries_month", "average_burglaries_year",
    "peak_month", "low_month", "risk_score"
]

info_panel_template = """
{% macro html(this, kwargs) %}

<!-- Info panel (hidden initially) -->
<div id="info-panel" style="
    display: none;
    position: fixed;
    bottom: 50px;
    left: 50px;
    width: 280px;
    background-color: white;
    border: 2px solid #444;
    z-index: 9999;
    padding: 12px;
    font-size: 14px;
    box-shadow: 3px 3px 6px rgba(0,0,0,0.3);
">
    <b>LSOA Information</b><br>
    Click on an LSOA to see patrol details.
</div>

<script>
function onEachFeature(feature, layer) {
    layer.on('click', function (e) {
        console.log("Feature clicked:", feature);
        var props = feature.properties;
        var panel = document.getElementById('info-panel');
        panel.style.display = "block";
        panel.innerHTML = 
            "<b>LSOA Code:</b> " + props.LSOA11CD + "<br>" +
            "<b>Patrol Hours:</b> " + props.patrol_hours + "<br>" +
            "<b>Total Burglaries:</b> " + props.total_burglaries + "<br>" +
            "<b>Total Burglaries This Year:</b> " + props.total_burglaries_cur_year + "<br>" +
            "<b>Avg Burglaries / Month:</b> " + props.average_burglaries_month.toFixed(2) + "<br>" +
            "<b>Avg Burglaries / Year:</b> " + props.average_burglaries_year.toFixed(2) + "<br>" +
            "<b>Peak Month:</b> " + props.peak_month + "<br>" +
            "<b>Lowest Month:</b> " + props.low_month + "<br>" +
            "<b>Risk Score:</b> " + props.risk_score;
    });
}
</script>

{% endmacro %}
"""

df_raw      = load_data(const.DATA_PATH)
lsoa_gdf    = load_lsoa_boundaries(const.LSOA_SHP_DIR)
ward_gdf    = load_shape_from_zip(const.GEO_ZIP_PATH, const.WARD_IN_ZIP)[["NAME", "geometry"]].rename(columns={"NAME": "Ward"})

# Layout for dropdowns side by side
col1, col2, col3 = st.columns(3)

with col1:
    year_options = [2025, 2026]
    selected_year = st.selectbox("Select year:", year_options)

with col2:
    month_options = ["january", "february", "march", "april", "may", "june",
                     "july", "august", "september", "october", "november", "december"]
    selected_month = st.selectbox("Select month:", month_options)

with col3:
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

hours = load_prediction_data(selected_month, selected_year, selected_ward)

if "LSOA" in hours.columns:
    hours = hours.rename(columns={"LSOA": "LSOA11CD"})

lsoas_in = lsoa_gdf[lsoa_gdf.geometry.centroid.within(ward_geom)].copy()

available_columns = [col for col in required_columns if col in hours.columns]
display_gdf = (
    lsoas_in
    .merge(hours[required_columns], on="LSOA11CD", how="left")
    .fillna({
        "patrol_hours": 0,
        "total_burglaries": 0,
        "total_burglaries_cur_year": 0,
        "average_burglaries_month": 0,
        "average_burglaries_year": 0,
        "peak_month": "N/A",
        "low_month": "N/A",
        "risk_score": "N/A"
    })
)
display_gdf["patrol_hours"] = display_gdf["patrol_hours"].astype(int)

m2 = folium.Map(
    location=[0, 0],  # temporary; will be overridden by fit_bounds
    zoom_start=13,
    tiles="cartodbpositron"
)

macro = MacroElement()
macro._template = Template(info_panel_template)
m2.get_root().add_child(macro)


# Fit the map to the exact bounds of the selected ward
bounds = ward_geom.bounds  # (minx, miny, maxx, maxy)
m2.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

folium.GeoJson(ward_geom, style_function=lambda f: {"fillOpacity":0,"color":"black","weight":2}).add_to(m2)
folium.Choropleth(
    geo_data=display_gdf,
    data=display_gdf,
    columns=required_columns,
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
        fields=["LSOA11CD","patrol_hours"],
        aliases=["LSOA code","Needed patrol hours"],
        localize=True
    ),
    on_each_feature="onEachFeature"
).add_to(m2)


folium.LayerControl().add_to(m2)

st.subheader(f"LSOAs in {selected_ward}")
st_html(m2._repr_html_(), height=600)

df_pred = load_prediction_data(selected_month, selected_year, selected_ward)

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
