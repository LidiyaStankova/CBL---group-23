import os
import pandas as pd
import geopandas as gpd

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


def preprocess_lsoas(shp_dir: str, out_path: str) -> gpd.GeoDataFrame:
    gdf_list = []
    for fname in os.listdir(shp_dir):
        if fname.lower().endswith(".shp"):
            full = os.path.join(shp_dir, fname)
            gdf_list.append(gpd.read_file(full))
    lsoa_gdf = pd.concat(gdf_list, ignore_index=True)

    # Rename to standard column name
    candidates = [c for c in lsoa_gdf.columns if "LSOA" in c.upper() and "CD" in c.upper()]
    if "LSOA11CD" not in lsoa_gdf.columns and len(candidates) == 1:
        lsoa_gdf = lsoa_gdf.rename(columns={candidates[0]: "LSOA11CD"})

    lsoa_gdf = lsoa_gdf.to_crs(epsg=4326)
    lsoa_gdf["geometry"] = lsoa_gdf["geometry"].simplify(tolerance=0.0001, preserve_topology=True)
    lsoa_gdf[["LSOA11CD", "geometry"]].to_file(out_path, driver="GeoJSON")

    return lsoa_gdf

def preprocess_stats(df: pd.DataFrame, output_dir: str = "."):
    # Make sure required columns exist
    if "Month" not in df.columns or "LSOA name" not in df.columns:
        raise ValueError("Input DataFrame must include 'Month' and 'LSOA name' columns")

    # Add helper columns
    df["Year"] = df["Month"].dt.year
    df["Month_Num"] = df["Month"].dt.month
    df["Month_Abbr"] = df["Month"].dt.month_name().str[:3]
    df["Month_Period"] = df["Month"].dt.to_period("M")

    # Monthly totals (e.g., 2021-01, 2021-02, ...)
    monthly_totals = df.groupby("Month_Period").size().rename("Total").reset_index()
    monthly_totals.to_parquet(os.path.join(output_dir, "monthly_totals.parquet"), index=False)

    # Yearly totals
    yearly_totals = df.groupby("Year").size().rename("Total").reset_index()
    yearly_totals.to_parquet(os.path.join(output_dir, "yearly_totals.parquet"), index=False)

    # Top 10 LSOAs
    top10_lsoas = df.groupby("LSOA name").size().rename("Count").nlargest(10).reset_index()
    top10_lsoas.to_parquet(os.path.join(output_dir, "top10_lsoas.parquet"), index=False)

    # Seasonal: Total per month abbreviation (e.g., Jan, Feb)
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    seasonal = (
        df.groupby("Month_Abbr").size()
        .reindex(month_order)
        .reset_index(name="Incidents")
        .rename(columns={"Month_Abbr": "MN"})
    )
    seasonal.to_parquet(os.path.join(output_dir, "seasonal_months.parquet"), index=False)

    # Boxplot data: Monthly counts by year and month
    df_ym = (
        df.groupby(["Year", "Month_Num"]).size()
          .reset_index(name="Count")
          .assign(Mo=lambda d: d["Month_Num"].apply(lambda m: month_order[m-1]))
    )
    df_ym.to_parquet(os.path.join(output_dir, "boxplot_monthly.parquet"), index=False)

    print("All statistical aggregates saved.")

def preprocess_per_borough():


    lsoa_path = r"\ESRI\LSOA_2011_London_gen_MHW.shp"
    ward_path = r"\ESRI\London_Ward_CityMerged.shp"
    borough_path = r"\ESRI\London_Borough_Excluding_MHW.shp"
    csv_path = r"..\burglary_2021_2025.csv"

    gdf_lsoa = gpd.read_file(lsoa_path)
    gdf_ward = gpd.read_file(ward_path)
    gdf_borough = gpd.read_file(borough_path)

    print(gdf_lsoa.columns)
    print(gdf_ward.columns)
    print(gdf_borough.columns)


    df = pd.read_csv(csv_path, parse_dates=["Month"])
    df = df.rename(columns={"LSOA code": "LSOA11CD"})
    df_burglary = df[df["Crime type"].str.lower() == "burglary"]
    burglary_counts = df_burglary.groupby("LSOA11CD").size().reset_index(name="Burglary_Count")

    #Merge
    gdf_lsoa = gdf_lsoa.merge(burglary_counts, on="LSOA11CD", how="left")
    gdf_lsoa["Burglary_Count"] = gdf_lsoa["Burglary_Count"].fillna(0)

    #Join for optaining wards names
    gdf_lsoa = gdf_lsoa.to_crs(gdf_ward.crs)  
    gdf_borough = gdf_borough[["NAME", "geometry"]] 
    gdf_lsoa = gpd.sjoin(gdf_lsoa, gdf_borough, how="left", predicate="intersects")
    gdf_lsoa = gdf_lsoa.rename(columns={"NAME": "Borough"})
    # Save the resulting GeoDataFrame to CSV (geometry will be WKT)
    gdf_lsoa.to_csv("./stats/lsoa_burglary_with_boroughs.csv", index=False)


DATA_PATH   = r"..\burglary_2021_2025.csv"
LSOA_SHP_DIR = r".\LB_shp"

PREPROCESSED_DATA_PATH = "processed_burglaries.parquet"
PREPROCESSED_LSOA_PATH = "processed_lsoas.geojson"

df_raw   = preprocess_data(DATA_PATH, PREPROCESSED_DATA_PATH)
lsoa_gdf = preprocess_lsoas(LSOA_SHP_DIR, PREPROCESSED_LSOA_PATH)
preprocess_stats(df_raw, output_dir="stats")
preprocess_per_borough()
preprocess_and_save_choropleth()