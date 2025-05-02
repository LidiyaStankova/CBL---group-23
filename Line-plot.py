#!/usr/bin/env python3
"""
plot_burglary_trend.py

Reads the consolidated burglary CSV, computes monthly counts and
a 3-month rolling average, and plots the time series with
shaded areas showing where the monthly count is above or below
the rolling average.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# —— Load and prepare data —— #
csv_path = Path(
    r"C:\Users\Gabri\Documents\GitHub\CBL---group-23\all_burglary.csv"
)
df = pd.read_csv(csv_path, dtype=str)

# Ensure Month is datetime, and sort
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month")

# Aggregate to get one count per month
monthly = df.groupby("Month").size().rename("Burglaries").to_frame()

# Compute a centered 3-month rolling average
monthly["RollingAvg3"] = (
    monthly["Burglaries"]
    .astype(float)
    .rolling(window=3, center=True)
    .mean()
)

# —— Plot —— #
fig, ax = plt.subplots(figsize=(12, 6))

# Convert dates for matplotlib
x = monthly.index.to_pydatetime()
y = monthly["Burglaries"].values
y_avg = monthly["RollingAvg3"].values

# Masks for shading
mask_above = (y >= y_avg) & ~pd.isna(y_avg)
mask_below = (y <  y_avg) & ~pd.isna(y_avg)

# Plot lines
ax.plot(x, y,      linewidth=1.5, marker="o", markersize=4, label="Monthly Count")
ax.plot(x, y_avg,  linewidth=2.5,            label="3-Month Rolling Avg")

# Shade between
ax.fill_between(x, y, y_avg, where=mask_above, interpolate=True,
                color="#1f77b4", alpha=0.1)
ax.fill_between(x, y, y_avg, where=mask_below, interpolate=True,
                color="#ff7f0e", alpha=0.1)

# Formatting the x-axis as dates
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_minor_locator(mdates.MonthLocator())

# Labels, title, grid, legend
ax.set_title(
    "Monthly Residential Burglaries in London\n"
    "(Metropolitan & City of London Police)",
    fontsize=16,
    pad=15
)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Number of Burglaries", fontsize=12)
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
ax.minorticks_on()
ax.legend()

# Rotate dates for readability
fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.show()
