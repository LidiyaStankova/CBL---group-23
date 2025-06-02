import streamlit as st
import pandas as pd
import folium

# Page config
st.set_page_config(page_title="Predictions", layout="wide")

mode = st.sidebar.radio(
    "Display mode",
    ("By borough", "By Ward")
)

# Layout for dropdowns side by side
if mode == "By borough":
    col1, col2 = st.columns(2)
else:
    col1, col2, col3 = st.columns(3)

with col1:
    month_options = ["january", "february", "march", "april", "may", "june",
                     "july", "august", "september", "october", "november", "december"]
    selected_month = st.selectbox("Select month:", month_options)

with col2:
    borough_options = ["Borough_1", "Borough_2"]
    selected_borough = st.selectbox("Select borough:", borough_options)

if mode == "By Ward":
    with col3:
        ward_options = ["Ward_1", "Ward_2"]
        selected_ward = st.selectbox("Select ward:", ward_options)

# Load prediction data
def load_prediction_data(month, borough, ward=None):
    if (mode == "By borough"):
        file_path = fr"C:\Users\20233537\OneDrive - TU Eindhoven\Documents\Homework\y2\CBL Multidisciplinary\CBL 2025 burglary\Github\CBL---group-23\dashboard\stats\predicted\2026\{borough}\{month}.csv"
    else:
        file_path = fr"C:\Users\20233537\OneDrive - TU Eindhoven\Documents\Homework\y2\CBL Multidisciplinary\CBL 2025 burglary\Github\CBL---group-23\dashboard\stats\predicted\2026\{borough}\{ward}\{month}.csv"
    return pd.read_csv(file_path)

# Load data
df_pred = load_prediction_data(selected_month, selected_borough, selected_ward if mode == "By Ward" else None)

# 🔧 Clean column names
df_pred.columns = df_pred.columns.str.strip().str.lower()

if mode == "By Ward":
    # Optional: clean & format
    df_pred = df_pred[["lsoa", "predicted_burglaries"]].rename(
        columns={"lsoa": "LSOA", "predicted_burglaries": "Predicted Burglaries"}
    )

# Header
st.title("📈 Predicted Burglaries")
st.markdown("This page shows forecasted burglary incidents for upcoming months based on historical data.")

# Display table
st.dataframe(df_pred, use_container_width=True)

# Optional: Chart
if "Year" in df_pred.columns and "Month" in df_pred.columns:
    df_pred["YM"] = df_pred["Year"].astype(str) + "-" + df_pred["Month"]
    chart_data = df_pred.copy()
    chart_data["YM"] = pd.to_datetime(chart_data["YM"])
    st.line_chart(chart_data.set_index("YM")["Predicted Incidents"])
