#!/usr/bin/env python3
"""
Merge.py

Scans all “street”-level CSVs under the `data` folder,
extracts only the “Burglary” records, and writes them to a single consolidated file.

Input directory:
    C:/Users/Gabri/Documents/GitHub/CBL---group-23/data

Output file:
    C:/Users/Gabri/Documents/GitHub/CBL---group-23/all_burglary.csv

Usage:
    1. Ensure pandas is installed:
         pip install pandas
    2. From project root run:
         python Merge.py
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# ——— Logging setup ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def find_street_csvs(base_dir: Path):
    """
    Recursively find all CSV files under `base_dir`
    whose names end with '-street.csv'.
    """
    logging.info(f"Searching for '*-street.csv' files in {base_dir}")
    files = list(base_dir.rglob("*-street.csv"))
    logging.info(f"Found {len(files)} street-level CSV file(s).")
    return files

def filter_burglary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows where the 'Crime type' column equals 'Burglary'.
    Raises KeyError if the column is missing.
    """
    if "Crime type" not in df.columns:
        raise KeyError("Missing required column: 'Crime type'")
    return df[df["Crime type"].str.strip().eq("Burglary")]

def main():
    # 1) Where to look for your monthly street-level CSVs
    input_dir = Path("C:/Users/Gabri/Documents/GitHub/CBL---group-23/data")
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # 2) Where to save the combined burglary file
    output_path = Path("C:/Users/Gabri/Documents/GitHub/CBL---group-23") / "all_burglary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for csv_path in find_street_csvs(input_dir):
        try:
            df = pd.read_csv(csv_path, dtype=str)
        except Exception as e:
            logging.warning(f"Could not read {csv_path.name}: {e}")
            continue

        try:
            df_burglary = filter_burglary(df)
        except KeyError as e:
            logging.warning(f"Skipping {csv_path.name}: {e}")
            continue

        if not df_burglary.empty:
            all_dfs.append(df_burglary)
            logging.info(f"  → {len(df_burglary)} burglary record(s) from {csv_path.name}")

    if not all_dfs:
        logging.warning("No burglary records found in any street-level CSV. Exiting.")
        sys.exit(0)

    # 3) Concatenate and write out
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_path, index=False)
    logging.info(f"Successfully wrote {len(combined)} total records to {output_path}")

if __name__ == "__main__":
    main()
