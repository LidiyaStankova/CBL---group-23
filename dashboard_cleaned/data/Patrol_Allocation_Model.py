### Allocation Model ###

import pandas as pd
import numpy as np
import cvxpy as cp

wla = pd.read_csv("ward_lsoa_with_area.csv")   
ward_preds = pd.read_csv(
    "ward_burglary_predictions_12months.csv",
    parse_dates=["Month"]
)

target_months = [
    pd.Timestamp(2025, 3, 1),
    pd.Timestamp(2025, 4, 1),
    pd.Timestamp(2025, 5, 1),
    pd.Timestamp(2025, 6, 1),
    pd.Timestamp(2025, 7, 1),
    pd.Timestamp(2025, 8, 1),
]

all_results = []

for tm in target_months:
    print(f"→ Processing {tm.strftime('%B %Y')}")
    
    ward_risk = (
        ward_preds
          .loc[ward_preds["Month"] == tm, ["Ward", "Predicted_Burglaries"]]
          .rename(columns={"Predicted_Burglaries": "r_i"})
    )
    if ward_risk.empty:
        print(f"   ⚠️  No data for {tm.strftime('%Y-%m')}, skipping.")
        continue

    data = pd.merge(wla, ward_risk, on="Ward", how="inner")

    
    days = tm.days_in_month                        # e.g. 31 for March
    base_monthly = 2.6 * days                       # 2.6 h/day × days
    min_monthly  = 1.0 * days                       # 1 h/day × days

    
    avg_area = data["area_km2"].mean()
    ratio    = data["area_km2"] / avg_area
    scale    = ratio.clip(lower=1.0, upper=2.0)

    data["x_max"] = base_monthly * scale
    data["alpha"] = 1.0 / np.log1p(data["x_max"])

    month_results = []
    for ward_name, grp in data.groupby("Ward"):
        n        = len(grp)
        r_i      = grp["r_i"].values
        alpha_i  = grp["alpha"].values
        x_max_i  = grp["x_max"].values

        X = cp.Variable(n)
        obj = cp.Maximize(cp.sum(cp.multiply(r_i * alpha_i,
                                             cp.log1p(X))))
        cons = [
            cp.sum(X) <= 3200,          # ward budget cap
            X >= min_monthly,           # minimum 1 h/day
            X <= x_max_i                # area-scaled max
        ]

        prob = cp.Problem(obj, cons)
        prob.solve()

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"   ⚠️  Ward {ward_name}: {prob.status}")
            continue

        X_cont = X.value
        X_rounded = np.ceil(X_cont * 4) / 4.0
        out = pd.DataFrame({
            "Month":      tm.strftime("%Y-%m"),
            "Ward":       ward_name,
            "LSOA_Code":  grp["LSOA_Code"].values,
            "Risk_Score": r_i,
            "Allocation": X_rounded,
        })
        month_results.append(out)

    if month_results:
        all_results.append(pd.concat(month_results, ignore_index=True))

if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df = final_df[["Month", "Ward", "LSOA_Code", "Risk_Score", "Allocation"]]

    out_name = "allocation_halfyear.csv"
    final_df.to_csv(out_name, index=False)
    print(f"\n✓ All done! Saved combined file: {out_name}")
else:
    print("No results to save.")