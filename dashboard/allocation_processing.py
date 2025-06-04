import os
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq

DATA_PATH   = r"C:\Users\20233537\OneDrive - TU Eindhoven\Documents\Homework\y2\CBL Multidisciplinary\CBL 2025 burglary\Github\CBL---group-23\burglary_2021_2025.csv"
PREPROCESSED_DATA_PATH = "processed_burglaries.parquet"
GEO_ZIP_PATH   = r"C:/Users/20233537/OneDrive - TU Eindhoven/Documents/Homework/y2/CBL Multidisciplinary/CBL 2025 burglary/Github/CBL---group-23/dashboard/stats/statistical-gis-boundaries-london.zip"
WARD_IN_ZIP    = "statistical-gis-boundaries-london/ESRI/London_Ward_CityMerged.shp"
LSOA_IN_ZIP    = "statistical-gis-boundaries-london/ESRI/LSOA_2011_London_gen_MHW.shp"

def load_shape_from_zip(zip_path, inside_path):
    gdf = gpd.read_file(f"zip://{zip_path}!{inside_path}")
    return gdf.to_crs(epsg=4326)

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
    import os
    from calendar import month_name

    current_year = 2025
    month_order = [m.lower() for m in list(month_name)[1:]]

    # Load LSOA and Ward GeoDataFrames
    lsoa_gdf = load_shape_from_zip(GEO_ZIP_PATH, LSOA_IN_ZIP)[["LSOA11CD", "geometry"]]
    ward_gdf = load_shape_from_zip(GEO_ZIP_PATH, WARD_IN_ZIP)[["NAME", "geometry"]].rename(columns={"NAME": "Ward"})

    # Ensure consistent CRS
    ward_gdf = ward_gdf.to_crs(epsg=27700)
    lsoa_gdf = lsoa_gdf.to_crs(epsg=27700)

    # Load burglary data
    all_burglary_data = pd.read_parquet(
        "C:/Users/20233537/OneDrive - TU Eindhoven/Documents/Homework/y2/CBL Multidisciplinary/CBL 2025 burglary/Github/CBL---group-23/dashboard/stats/processed_burglaries.parquet"
    )

    # Normalize date information
    all_burglary_data['Month'] = pd.to_datetime(all_burglary_data['Month'])
    all_burglary_data['month_name'] = all_burglary_data['Month'].dt.month_name().str.lower()
    all_burglary_data['year'] = all_burglary_data['Month'].dt.year

    ward_list = ward_gdf.Ward.sort_values().unique()

    for ward_name in ward_list:
        ward_dir = f"dashboard/stats/allocation/2026/{ward_name}"
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


            # Filter for 2025 once
            lsoa_2025_burglaries = lsoa_burglaries[lsoa_burglaries['year'] == current_year].copy()
            lsoa_2025_burglaries['month_name'] = lsoa_2025_burglaries['Month'].dt.month_name().str.lower()

            for i, month in enumerate(month_order):
                months_until_now = month_order[:i + 1]
                total_burglaries_cur_year = lsoa_2025_burglaries[
                    lsoa_2025_burglaries["month_name"].isin(months_until_now)
                ].shape[0]

                data = {
                    "LSOA": lsoa_code,
                    "patrol_hours": 0,  # Placeholder
                    "average_burglaries_month": average_burglaries_month,
                    "average_burglaries_year": average_burglaries_year,
                    "peak_month": peak_month,
                    "low_month": low_month,
                    "risk_score": "xx,xx",  # Placeholder
                    "total_burglaries": total_burglaries,
                    "total_burglaries_cur_year": total_burglaries_cur_year
                }

                month_file = os.path.join(ward_dir, f"{month}.csv")
                df = pd.DataFrame([data])

                if os.path.exists(month_file):
                    df.to_csv(month_file, mode='a', index=False, header=False)
                else:
                    df.to_csv(month_file, index=False)

preprocess_data(DATA_PATH, PREPROCESSED_DATA_PATH)
lsoa_allocation_data()