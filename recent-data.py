import pandas as pd
import os

# --- Set paths ---
input_file = './all_burglary.csv'
output_file = './all_burglary_2024.csv'

# --- Load and parse dates ---
df = pd.read_csv(input_file, parse_dates=['Month'], low_memory=False)

# --- Filter rows: Keep only rows from 2020 onwards ---
df_filtered = df[df['Month'] >= pd.Timestamp('2024-01')]

# --- Save filtered data ---
df_filtered.to_csv(output_file, index=False)

print(f"Filtered data saved to: {output_file}")
print(f"Original rows: {len(df)}, After filtering: {len(df_filtered)}")
