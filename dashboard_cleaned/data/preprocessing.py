import os
import sys
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dashboard_cleaned.other import constants as const
from calendar import month_name
import numpy as np
import shutil

# ─────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────

def load_shape_from_zip(zip_path, inside_path):
    gdf = gpd.read_file(f"zip://{zip_path}!{inside_path}")
    return gdf.to_crs(epsg=4326)

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

def load_ward_boundaries(zip_path, inside_path):
    gdf = gpd.read_file(f"zip://{zip_path}!{inside_path}")
    return gdf[["NAME", "geometry"]].rename(columns={"NAME": "Ward"}).to_crs(epsg=4326)

def load_data(path):
    df = pd.read_csv(path, parse_dates=["Month"], dayfirst=True)
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["LSOA11CD"] = df["LSOA code"]
    df = df[
        df.Longitude.between(-0.55, 0.30) &
        df.Latitude.between(51.25, 51.70)
    ].copy()
    return df

# ─────────────────────────────────────────────────────────────────────
# Preprocessing functions
# ─────────────────────────────────────────────────────────────────────

def preprocess_data(raw_path: str, preprocessed_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, parse_dates=["Month"], dayfirst=True)
    df = df.dropna(subset=["Latitude", "Longitude"])
    df.to_parquet(preprocessed_path, index=False)

    # restrict to London bounding box
    df = df[
        df["Longitude"].between(-0.55, 0.30) &
        df["Latitude"].between(51.25, 51.70)
    ].copy()
    return df

def lsoa_allocation_data():
    month_order = [m.lower() for m in list(month_name)[1:]]

    # Load LSOA and Ward GeoDataFrames
    lsoa_gdf = load_shape_from_zip(const.GEO_ZIP_PATH, const.LSOA_IN_ZIP)[["LSOA11CD", "geometry"]]
    ward_gdf = load_shape_from_zip(const.GEO_ZIP_PATH, const.WARD_IN_ZIP)[["NAME", "geometry"]].rename(columns={"NAME": "Ward"})

    # Ensure consistent CRS
    ward_gdf = ward_gdf.to_crs(epsg=27700)
    lsoa_gdf = lsoa_gdf.to_crs(epsg=27700)

    # Load burglary data
    all_burglary_data = pd.read_parquet(
        const.PROCESSED_DATA_PATH + "/processed_burglaries.parquet"
    )

    # Normalize date information
    all_burglary_data['Month'] = pd.to_datetime(all_burglary_data['Month'])
    all_burglary_data['month_name'] = all_burglary_data['Month'].dt.month_name().str.lower()
    all_burglary_data['year'] = all_burglary_data['Month'].dt.year

    ward_list = ward_gdf.Ward.sort_values().unique()
    year_list = all_burglary_data['year'].unique()

    allocation_data = pd.read_csv(const.ALLOCATION_DATA_PATH)
    risk_scores_data = pd.read_csv(const.RISK_SCORES_PATH)

    for year in np.append(year_list, max(year_list) + 1):
        if os.path.exists(const.ALLOCATION_PATH + f"/{year}"):
            shutil.rmtree(const.ALLOCATION_PATH + f"/{year}", ignore_errors=True)
        year_dir = const.ALLOCATION_PATH + f"/{year}"
        os.makedirs(year_dir, exist_ok=True)

        for ward_name in ward_list:
            ward_dir = const.ALLOCATION_PATH + f"/{year}/{ward_name}"
            os.makedirs(ward_dir, exist_ok=True)

            # Get ward polygon and find LSOAs within it
            ward_poly = ward_gdf.loc[ward_gdf["Ward"] == ward_name, "geometry"].values[0]
            lsoas_in_ward = lsoa_gdf[lsoa_gdf.centroid.within(ward_poly)]

            for _, lsoa_row in lsoas_in_ward.iterrows():
                lsoa_code = lsoa_row["LSOA11CD"]
                lsoa_burglaries = all_burglary_data[all_burglary_data["LSOA code"] == lsoa_code]

                total_burglaries = len(lsoa_burglaries)

                # Group by year and month to get count per (year, month)
                monthly_year_counts = (
                    lsoa_burglaries
                    .groupby(["year", "month_name"])
                    .size()
                    .unstack(fill_value=0)
                )

                # Compute average per month across years
                average_monthly = monthly_year_counts.mean().reindex(month_order)

                # Compute peak and low based on average
                peak_month = average_monthly.idxmax()
                low_month = average_monthly.idxmin()

                # Overall averages
                average_burglaries_month = average_monthly.mean()
                average_burglaries_year = total_burglaries / lsoa_burglaries['year'].nunique() if lsoa_burglaries['year'].nunique() > 0 else 0

                average_burglaries_month = round(average_burglaries_month, 1)
                average_burglaries_year = round(average_burglaries_year, 1)


                # Filter for 2025 once
                lsoa_2025_burglaries = lsoa_burglaries[lsoa_burglaries['year'] == year].copy()
                lsoa_2025_burglaries['month_name'] = lsoa_2025_burglaries['Month'].dt.month_name().str.lower()

                patrol_hours = allocation_data.loc[
                    allocation_data["LSOA_Code"] == lsoa_code, "Allocation"
                ].values[0]

                risk_score_row = risk_scores_data.loc[
                    risk_scores_data["LSOAcode"] == lsoa_code, "Exp_Risk_Score"
                ]
                risk_score = risk_score_row.values[0] if not risk_score_row.empty else np.nan
                if not np.isnan(risk_score):
                    risk_score = round(risk_score, 1)

                for i, month in enumerate(month_order):
                    # Only process if this month exists in the data
                    total_burglaries_cur_year = 0
                    if month not in lsoa_2025_burglaries['month_name'].values:
                        total_burglaries_cur_year = 0
                    else:
                        months_until_now = month_order[:i + 1]
                        total_burglaries_cur_year = lsoa_2025_burglaries[
                            lsoa_2025_burglaries["month_name"].isin(months_until_now)
                        ].shape[0]

                    data = {
                        "LSOA": lsoa_code,
                        "patrol_hours": patrol_hours,  # Placeholder
                        "average_burglaries_month": average_burglaries_month,
                        "average_burglaries_year": average_burglaries_year,
                        "peak_month": peak_month,
                        "low_month": low_month,
                        "risk_score": risk_score,  # Placeholder
                        "total_burglaries": total_burglaries,
                        "total_burglaries_cur_year": total_burglaries_cur_year
                    }

                    month_file = os.path.join(ward_dir, f"{month}.csv")
                    df = pd.DataFrame([data])

                    if os.path.exists(month_file):
                        df.to_csv(month_file, mode='a', index=False, header=False)
                    else:
                        df.to_csv(month_file, index=False)

def ward_burglary(_df_raw, _lsoa_gdf, _ward_gdf):
    """
    Precomputes ward-level burglary counts over time and saves the result to a file.
    """
    save_path = const.PROCESSED_DATA_PATH + "/ward_burglary.feather"

    df_burglary = _df_raw[_df_raw["Crime type"].str.lower() == "burglary"].copy()
    df_burglary["Month"] = df_burglary["Month"].dt.to_period("M").dt.to_timestamp()

    lsoa_month_counts = (
        df_burglary
        .groupby(["LSOA11CD", "Month"])
        .size()
        .reset_index(name="LSOA_Burglary_Count")
    )

    gdf_lsoa_counts = _lsoa_gdf.merge(lsoa_month_counts, on="LSOA11CD", how="left")
    gdf_lsoa_counts["LSOA_Burglary_Count"] = gdf_lsoa_counts["LSOA_Burglary_Count"].fillna(0).astype(int)

    gdf_ward_small = _ward_gdf[["Ward", "geometry"]].copy()
    gdf_lsoa_counts = gpd.sjoin(gdf_lsoa_counts, gdf_ward_small, how="left", predicate="intersects")
    gdf_lsoa_counts = gdf_lsoa_counts.drop(columns=["index_right"], errors="ignore")

    ward_month = (
        gdf_lsoa_counts
        .drop(columns=["geometry"], errors="ignore")
        .groupby(["Ward", "Month"])
        .agg({"LSOA_Burglary_Count": "sum"})
        .reset_index()
        .rename(columns={"LSOA_Burglary_Count": "Ward_Burglary_Count"})
    )

    all_months = ward_month["Month"].sort_values().unique()
    months_df = pd.DataFrame({"Month": all_months})
    months_df["key"] = 1

    gdf_ward_simple = _ward_gdf.copy()
    gdf_ward_simple["geometry"] = gdf_ward_simple.geometry.simplify(tolerance=0.001, preserve_topology=True)
    gdf_ward_simple["key"] = 1

    wards_cartesian = gdf_ward_simple[["Ward", "geometry", "key"]].merge(months_df, on="key").drop(columns=["key"])
    wards_timesliced = wards_cartesian.merge(ward_month, on=["Ward", "Month"], how="left")
    wards_timesliced["Ward_Burglary_Count"] = wards_timesliced["Ward_Burglary_Count"].fillna(0).astype(int)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    wards_timesliced.to_feather(save_path)

preprocess_data(const.DATA_PATH, const.PROCESSED_DATA_PATH + "/processed_burglaries.parquet")
lsoa_allocation_data()

df_raw       = load_data(const.DATA_PATH)
lsoa_gdf     = load_lsoa_boundaries(const.LSOA_SHP_DIR)
ward_gdf     = load_ward_boundaries(const.GEO_ZIP_PATH, const.WARD_IN_ZIP)
#ward_burglary(df_raw, lsoa_gdf, ward_gdf)
