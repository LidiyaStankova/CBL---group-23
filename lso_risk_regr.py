import pandas as pd
import statsmodels.api as sm
import numpy as np

crime_df = pd.read_csv('data/combined_dataset.csv', low_memory=False)
burglary_df = crime_df[crime_df['Crime type'] == 'Burglary']
burglary_by_lsoa = burglary_df.groupby('LSOA code').size().reset_index(name='Burglary_Count')
# Load merged structural data
domain_cols = [
    'LSOA code (2011)',
    'Income Decile (where 1 is most deprived 10% of LSOAs)',
    'Education, Skills and Training Decile (where 1 is most deprived 10% of LSOAs)',
    'Health Deprivation and Disability Decile (where 1 is most deprived 10% of LSOAs)',
    'Crime Decile (where 1 is most deprived 10% of LSOAs)',
    'Living Environment Decile (where 1 is most deprived 10% of LSOAs)'
]
domains = pd.read_excel('deprivation_data/File_2_-_IoD2019_Domains_of_Deprivation.xlsx')[domain_cols]
domains.columns = ['LSOA code', 'Income Decile', 'Education Decile', 'Health Decile', 'Crime Decile', 'Living Decile']

# Poverty Change
poverty_df = pd.read_excel('deprivation_data/poverty_2013_update.xls', sheet_name='Map 8', skiprows=3)
poverty_df = poverty_df.rename(columns={
    poverty_df.columns[0]: 'LSOA code',
    poverty_df.columns[2]: 'Poverty_2006',
    poverty_df.columns[3]: 'Poverty_2010'
})
poverty_df['Poverty_2006'] = poverty_df['Poverty_2006'].astype(str).str.rstrip('%').replace('-', np.nan).astype(float)
poverty_df['Poverty_2010'] = poverty_df['Poverty_2010'].astype(str).str.rstrip('%').replace('-', np.nan).astype(float)
poverty_df['Poverty_Change'] = poverty_df['Poverty_2010'] - poverty_df['Poverty_2006']

# %_Single_Person, %_With_Children
household_df = pd.read_excel('deprivation_data/Household composition.xlsx', sheet_name='2021')
household_df['%_Single_Person'] = (
    (household_df['One person Aged 66+'] + household_df['One person Aged up to 65']) /
    household_df['All households'] * 100
)

# %_Arrived_After_18
age_uk = pd.read_excel('deprivation_data/Age on arrival in UK.xlsx', sheet_name='2021')
age_uk['%_Arrived_After_18'] = age_uk[['Aged 18 to 19', 'Aged 20 to 24', 'Aged 25 to 29',
                                       'Aged 30 to 44', 'Aged 45 to 59', 'Aged 60 to 64',
                                       'Aged 65 to 74', 'Aged 75 to 84', 'Aged 85 to 89',
                                       'Aged 90 or over']].sum(axis=1) / age_uk['All usual residents'] * 100

# Looking_After_Home, Sick_Disabled
economic = pd.read_excel('deprivation_data/Economic Activity.xlsx', sheet_name='2021')
economic = economic[['LSOA code',
                     'Economically inactive: Looking after home or family',
                     'Economically inactive: Long-term sick or disabled']]
economic.columns = ['LSOA code', 'Looking_After_Home', 'Sick_Disabled']

population = pd.read_excel('deprivation_data/File_6_-_IoD2019_Population_Denominators.xlsx')
population = population[['LSOA code (2011)', 'Total population: mid 2015 (excluding prisoners)']]
population.columns = ['LSOA code', 'Population']

merged = burglary_by_lsoa \
    .merge(domains, on='LSOA code', how='left') \
    .merge(poverty_df[['LSOA code', 'Poverty_Change']], on='LSOA code', how='left') \
    .merge(household_df[['LSOA code', '%_Single_Person']], on='LSOA code', how='left') \
    .merge(age_uk[['LSOA code', '%_Arrived_After_18']], on='LSOA code', how='left') \
    .merge(economic, on='LSOA code', how='left')\
    .merge(population, on='LSOA code', how='left')

# Calculate burglary rate
merged = merged.dropna(subset=['Population'])
merged['Burglary_Rate_per_1000'] = merged['Burglary_Count'] / merged['Population'] * 1000

# Features and target
X = merged[[
     'Income Decile', 'Education Decile', 'Crime Decile',
    'Looking_After_Home', 'Sick_Disabled',
    '%_Arrived_After_18', '%_Single_Person', 'Poverty_Change'
]]
y = merged['Burglary_Rate_per_1000']

X = sm.add_constant(X)
# Drop rows with any NaNs or Infs in features or target
X = X.replace([np.inf, -np.inf], np.nan)
data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns='Burglary_Rate_per_1000')
y = data['Burglary_Rate_per_1000']

model = sm.OLS(y, X).fit()
merged['Risk_Score'] = model.predict(X)

risk_scores = merged[['LSOA code', 'Risk_Score']]
risk_scores.to_csv("data/lsoa_risk_scores.csv", index=False)