import os
import pandas as pd

data_folder = 'data'
dfs = []

# Loop through all data files
for file_name in os.listdir(data_folder):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_folder, file_name)
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Could not read {file_name}: {e}")

#Combine all dataframes into one
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv('combined_dataset.csv', index=False)
    print(f" Successfully combined {len(dfs)} CSV files.")