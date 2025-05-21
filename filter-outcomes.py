import pandas as pd
import glob
import os

# --- Parameters ---
folder_path = "data"
search_pattern = os.path.join(folder_path, "*outcome*.csv")
output_file = "all_outcomes.csv"

# --- Step 1: Find all matching CSV files ---
files = glob.glob(search_pattern)

if not files:
    print("No files found matching pattern:", search_pattern)
    exit()

# --- Step 2: Load and filter each file ---
dataframes = []

for file in files:
    try:
        df = pd.read_csv(file)
        if 'Outcome type' in df.columns:
            filtered_df = df[df['Outcome type'] != ""]
            dataframes.append(filtered_df)
            print(f"Loaded and filtered {file}, kept {len(filtered_df)} rows.")
        else:
            print(f"Skipped {file}: 'Outcome type' column not found.")
    except Exception as e:
        print(f"Error processing {file}: {e}")

# --- Step 3: Combine and export ---
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"\n✅ Merged {len(dataframes)} files. Saved to '{output_file}' ({len(combined_df)} rows).")
else:
    print("No data to save. All files may have been empty or filtered out.")
