import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import xlrd
import numpy as np

#load burglary data
crime_df = pd.read_csv('data/combined_dataset.csv', low_memory=False)
burglary_df = crime_df[crime_df['Crime type'] == 'Burglary']

#group by lsoa
burglary_by_lsoa = burglary_df.groupby('LSOA code').size().reset_index(name='Burglary_Count')

#load deprivation data
imd = pd.read_excel('deprivation_data/File_1_-_IMD2019_Index_of_Multiple_Deprivation.xlsx')
domains = pd.read_excel('deprivation_data/File_2_-_IoD2019_Domains_of_Deprivation.xlsx')
scores = pd.read_excel('deprivation_data/File_5_-_IoD2019_Scores.xlsx')
population = pd.read_excel('deprivation_data/File_6_-_IoD2019_Population_Denominators.xlsx')
imd.columns = imd.columns.str.strip()

#new data
age_uk = pd.read_excel('deprivation_data/Age on arrival in UK.xlsx', sheet_name='2021')
birth_country = pd.read_excel('deprivation_data/Country of birth.xlsx', sheet_name='2021')
economic = pd.read_excel('deprivation_data/Economic Activity.xlsx', sheet_name='2021')
ethnicity = pd.read_excel('deprivation_data/Ethnic group.xlsx')
household_df = pd.read_excel('deprivation_data/Household composition.xlsx', sheet_name='2021')
household_depr = pd.read_excel('deprivation_data/Household deprivation.xlsx')
poverty_df = pd.read_excel('deprivation_data/poverty_2013_update.xls', sheet_name='Map 8', skiprows=3)
#table_1 = pd.read_excel('deprivation_data/table_2025-05-05_21-03-14.xlsx')
#table_2 = pd.read_excel('deprivation_data/table_2025-05-05_21-09-39.xlsx')




###COLUMN CLEANING




#clean columns, prep for merging
imd = imd[['LSOA code (2011)',
           'Index of Multiple Deprivation (IMD) Rank',
           'Index of Multiple Deprivation (IMD) Decile']]
imd.columns = ['LSOA code', 'IMD Decile', 'IMD Rank']

scores = scores[['LSOA code (2011)', 'Index of Multiple Deprivation (IMD) Score']]
scores.columns = ['LSOA code', 'IMD Score']

population = population[['LSOA code (2011)', 'Total population: mid 2015 (excluding prisoners)']]
population.columns = ['LSOA code', 'Population']

birth_country = birth_country[['LSOA code', 'All Usual residents', 'United Kingdom']]
birth_country.columns = ['LSOA code', 'Total_Residents', 'UK_Born']
birth_country['%_Born_Abroad'] = 100 * (birth_country['Total_Residents'] - birth_country['UK_Born']) / birth_country['Total_Residents']

#select necessary columns
domain_cols = [
    'LSOA code (2011)',
    'Income Decile (where 1 is most deprived 10% of LSOAs)',
    'Education, Skills and Training Decile (where 1 is most deprived 10% of LSOAs)',
    'Health Deprivation and Disability Decile (where 1 is most deprived 10% of LSOAs)',
    'Crime Decile (where 1 is most deprived 10% of LSOAs)'
]
domains = domains[domain_cols]
domains.columns = ['LSOA code', 'Income Decile', 'Education Decile', 'Health Decile', 'Crime Decile']

#econ col
econ_cols = ['LSOA code',
             'Economically active: Unemployed',
             'Economically inactive: Long-term sick or disabled',
             'Economically inactive: Looking after home or family']

economic = economic[econ_cols]
economic.columns = ['LSOA code', 'Unemployed', 'Sick_Disabled', 'Looking_After_Home']
#age col
age_uk = age_uk[['LSOA code', 'All usual residents', 'Aged 0 to 4', 'Aged 5 to 7', 'Aged 8 to 9',
                 'Aged 10 to 14', 'Aged 16 to 17', 'Aged 18 to 19', 'Aged 20 to 24', 'Aged 25 to 29',
                 'Aged 30 to 44', 'Aged 45 to 59', 'Aged 60 to 64', 'Aged 65 to 74', 'Aged 75 to 84',
                 'Aged 85 to 89', 'Aged 90 or over']]

age_uk['%_Arrived_Before_18'] = age_uk[['Aged 0 to 4', 'Aged 5 to 7', 'Aged 8 to 9',
                                        'Aged 10 to 14', 'Aged 16 to 17']].sum(axis=1) / age_uk['All usual residents'] * 100

age_uk['%_Arrived_After_18'] = age_uk[['Aged 18 to 19', 'Aged 20 to 24', 'Aged 25 to 29',
                                       'Aged 30 to 44', 'Aged 45 to 59', 'Aged 60 to 64',
                                       'Aged 65 to 74', 'Aged 75 to 84', 'Aged 85 to 89',
                                       'Aged 90 or over']].sum(axis=1) / age_uk['All usual residents'] * 100

age_new_vars = age_uk[['LSOA code', '%_Arrived_Before_18', '%_Arrived_After_18']]
#poverty
poverty_df = poverty_df.rename(columns={
    poverty_df.columns[0]: 'LSOA code',
    poverty_df.columns[1]: 'Local Authority',
    poverty_df.columns[2]: 'Poverty_2006',
    poverty_df.columns[3]: 'Poverty_2010'
})

# Step 3: Remove % symbols and convert to float
poverty_df['Poverty_2006'] = (
    poverty_df['Poverty_2006']
    .astype(str)
    .str.rstrip('%')
    .replace('-', np.nan)
    .astype(float)
)
poverty_df['Poverty_2010'] = (
    poverty_df['Poverty_2010']
    .astype(str)
    .str.rstrip('%')
    .replace('-', np.nan)
    .astype(float)
)

# Step 4: Create new column for change
poverty_df['Poverty_Change'] = poverty_df['Poverty_2010'] - poverty_df['Poverty_2006']
#household
household_df['%_Lone_Parent'] = (
    (household_df['Lone parent: dependent children'] + household_df['Lone parent: non-dependent children'])
    / household_df['All households'] * 100
)

household_df['%_Single_Person'] = (
    (household_df['One person Aged 66+'] + household_df['One person Aged up to 65'])
    / household_df['All households'] * 100
)

# Calculate % Households With Children
household_df['%_With_Children'] = (
    (
        household_df['Married or civil partnership couple: Dependent children'] +
        household_df['Partnership couple: Dependent children'] +
        household_df['Cohabiting couple: Dependent children'] +
        household_df['Lone parent: dependent children'] +
        household_df['Other with dependent children']
    ) / household_df['All households'] * 100
)

#merge togheter
merged = burglary_by_lsoa \
    .merge(imd, on='LSOA code', how='left') \
    .merge(scores, on='LSOA code', how='left') \
    .merge(domains, on='LSOA code', how='left') \
    .merge(population, on='LSOA code', how='left')

#drop rows whre pop 0
merged = merged.dropna(subset=['Population'])

#calculate per 1000
merged['Burglary_Rate_per_1000'] = merged['Burglary_Count'] / merged['Population'] * 1000
merged = merged.merge(birth_country[['LSOA code', '%_Born_Abroad']], on='LSOA code', how='left')
merged = merged.merge(economic, on='LSOA code', how='left')
merged = merged.merge(household_df, on='LSOA code', how='left')
#cleaned = merged[(merged['Burglary_Rate_per_1000'] < 1250) & (merged['Population'] < 5000)]
merged = merged.merge(age_new_vars, on='LSOA code', how='left')
merged = merged.merge(poverty_df[['LSOA code', 'Poverty_2006', 'Poverty_2010', 'Poverty_Change']],
                      on='LSOA code', how='left')
#Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='%_Single_Person', y='Burglary_Rate_per_1000', data=merged)
plt.title(" Burglary Rate vs %%_Single_Person")
plt.xlabel("%%%_Single_Person")
plt.ylabel("Burglary Rate per 1,000 people")
plt.tight_layout()
plt.show()

#merged['IMD Decile'] = merged['IMD Decile'].astype(int).astype(str)

#filter outliers
#filtered = merged[merged['Burglary_Rate_per_1000'] < 800]

#compute avg
#group_avg = filtered.groupby('IMD Decile')['Burglary_Rate_per_1000'].mean().reset_index()

# Plot
# plt.figure(figsize=(10, 5))
# sns.barplot(x='%_Born_Abroad', y='Burglary_Rate_per_1000', data=merged)
# plt.title('📊 Average Burglary Rate per IMD Decile ')
# plt.xlabel('IMD Decile (1 = Most Deprived)')
# plt.ylabel('Average Burglaries per 1,000 People')
# plt.tight_layout()
# plt.show()
#
corr_cols = ['Burglary_Rate_per_1000', 'IMD Score','%_Single_Person', 'Income Decile','%_Arrived_Before_18','%_Arrived_After_18', 'Education Decile', 'Health Decile', 'Crime Decile', 'Unemployed', 'Looking_After_Home','Sick_Disabled','Poverty_Change']
corr_matrix = merged[corr_cols].corr(method='spearman')
corr_cols_extended = corr_cols + ['%_Born_Abroad']
corr_matrix_ext = merged[corr_cols_extended].corr(method='spearman')
print("📊 Spearman Correlation Matrix:")
print(corr_matrix_ext)

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix_ext, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Spearman Correlation Heatmap')
plt.tight_layout()
plt.show()


mlr_data = merged[['Burglary_Rate_per_1000', 'IMD Score', '%_Single_Person','Income Decile', 'Poverty_Change',
                   'Education Decile', 'Health Decile','%_Arrived_After_18', 'Crime Decile', 'Looking_After_Home','Sick_Disabled']].dropna()


X = mlr_data[['Income Decile','Poverty_Change','%_Single_Person', 'Education Decile','%_Arrived_After_18','Crime Decile','Looking_After_Home','Sick_Disabled']]#indep
y = mlr_data['Burglary_Rate_per_1000']#depend

#add constant for intercept
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

model_summary = model.summary()
print(vif_data)
print(model_summary)