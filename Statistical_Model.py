##### With Log variabels, same regression with transforming scales #####

import pandas as pd
import numpy as np
import statsmodels.api as sm
from IPython.display import display, FileLink

# Load and clean data
df = pd.read_csv('FinalDataC.csv')
df.columns = df.columns.str.strip()
df.set_index('LSOAcode', inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Remove invalid values for burglary and population
df = df.dropna(subset=['BurglaryCount', 'Totalpopulationmid2015excludingprisoners'])
df = df[(df['BurglaryCount'] > 0) & (df['Totalpopulationmid2015excludingprisoners'] > 0)]

# Create burglary rate per 1000
df['BurglaryRate'] = df['BurglaryCount'] / df['Totalpopulationmid2015excludingprisoners'] * 1000
df['LogBurglaryRate'] = np.log(df['BurglaryRate'] + 1e-5)

# Log-transform selected predictors
df['log_NROfMeters'] = np.log(df['NROfMeters'])
df['log_MeanHousingPrice2022'] = np.log(df['MeanHousingPrice2022'])
df['log_Houseswithoutcentralheatingindicator'] = np.log(df['Houseswithoutcentralheatingindicator'] + 1e-5)

# Define predictors
selected_features = [
    'LivingEnvironmentRankwhere1ismostdeprived',
    'CrimeRankwhere1ismostdeprived',
    'Nitrogendioxidecomponentofairqualityindicator',
    'Entrytohighereducationindicator',
    'log_NROfMeters',
    'log_MeanHousingPrice2022',
    'log_Houseswithoutcentralheatingindicator'
]

# Prepare X and y
X = df[selected_features]
y = df['LogBurglaryRate']

# Add constant and clean
X = sm.add_constant(X)
X = X.loc[:, ~X.columns.duplicated()]
X = X.replace([np.inf, -np.inf], np.nan)

# Drop rows with missing values
data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns='LogBurglaryRate')
y = data['LogBurglaryRate']

# Fit model
model = sm.OLS(y, X).fit()

# Predict
pred = model.predict(X)

# Assign predictions
df = df.loc[data.index].copy()
df['Log_Risk_Score'] = pred
df['Exp_Risk_Score'] = np.exp(pred)

# Save to new output file
output_path = "log_burglary_risk_scores_v2.csv"
df[['Log_Risk_Score', 'Exp_Risk_Score']].to_csv(output_path, index_label='LSOAcode')

# Show summary and download link
display(model.summary())
display(FileLink(output_path))
print(f"File saved to: {output_path}")