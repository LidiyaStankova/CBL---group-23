import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

crime_df = pd.read_csv('data/combined_dataset.csv', low_memory=False)
burglary_df = crime_df[crime_df['Crime type'] == 'Burglary']
burglary_by_lsoa = burglary_df.groupby('LSOA code').size().reset_index(name='Burglary_Count')

#external data
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

#poverty data
poverty_df = pd.read_excel('deprivation_data/poverty_2013_update.xls', sheet_name='Map 8', skiprows=3)
poverty_df = poverty_df.rename(columns={
    poverty_df.columns[0]: 'LSOA code',
    poverty_df.columns[2]: 'Poverty_2006',
    poverty_df.columns[3]: 'Poverty_2010'
})
poverty_df['Poverty_2006'] = poverty_df['Poverty_2006'].astype(str).str.rstrip('%').replace('-', np.nan).astype(float)
poverty_df['Poverty_2010'] = poverty_df['Poverty_2010'].astype(str).str.rstrip('%').replace('-', np.nan).astype(float)
poverty_df['Poverty_Change'] = poverty_df['Poverty_2010'] - poverty_df['Poverty_2006']

#fam composition
household_df = pd.read_excel('deprivation_data/Household composition.xlsx', sheet_name='2021')
household_df['%_Single_Person'] = (
    (household_df['One person Aged 66+'] + household_df['One person Aged up to 65']) /
    household_df['All households'] * 100
)

#age data
age_uk = pd.read_excel('deprivation_data/Age on arrival in UK.xlsx', sheet_name='2021')
age_uk['%_Arrived_After_18'] = age_uk[['Aged 18 to 19', 'Aged 20 to 24', 'Aged 25 to 29',
                                       'Aged 30 to 44', 'Aged 45 to 59', 'Aged 60 to 64',
                                       'Aged 65 to 74', 'Aged 75 to 84', 'Aged 85 to 89',
                                       'Aged 90 or over']].sum(axis=1) / age_uk['All usual residents'] * 100

#stay at home data
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

#calc per 1000 rate
merged = merged.dropna(subset=['Population'])
merged['Burglary_Rate_per_1000'] = merged['Burglary_Count'] / merged['Population'] * 1000

#feature, target
X = merged[[
     'Income Decile', 'Education Decile', 'Crime Decile',
    'Looking_After_Home', 'Sick_Disabled',
    '%_Arrived_After_18', '%_Single_Person', 'Poverty_Change'
]]
y = merged['Burglary_Rate_per_1000']

X = sm.add_constant(X)
#drop empty
X = X.replace([np.inf, -np.inf], np.nan)
data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns='Burglary_Rate_per_1000')
y = data['Burglary_Rate_per_1000']

model = sm.OLS(y, X).fit()
merged['Risk_Score'] = model.predict(X)

risk_scores = merged[['LSOA code', 'Risk_Score']]
#risk_scores.to_csv("data/lsoa_risk_scores.csv", index=False)
print(model.summary())

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# 1. Coefficient Plot - Visualize the impact of each feature
plt.figure(figsize=(12, 8))
coef_df = pd.DataFrame({
    'Feature': model.params.index[1:],  # Skip the constant
    'Coefficient': model.params.values[1:],
    'Error': model.bse.values[1:],
    'p-value': model.pvalues.values[1:]
})
coef_df = coef_df.sort_values('Coefficient')

# Create coefficient plot - FIXED version
plt.figure(figsize=(12, 8))
colors = ['#1e88e5' if p < 0.05 else '#d32f2f' for p in coef_df['p-value']]

# Plot each point individually to avoid the color error
for i, row in coef_df.iterrows():
    color = '#1e88e5' if row['p-value'] < 0.05 else '#d32f2f'
    plt.errorbar(
        row['Coefficient'],
        row['Feature'],
        xerr=row['Error'],
        fmt='o',
        capsize=5,
        ecolor='black',
        markersize=8,
        linewidth=2,
        color=color
    )

# Add vertical line at x=0
plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)

# Add labels and title
plt.xlabel('Coefficient Value', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Regression Coefficients with 95% Confidence Intervals', fontsize=16)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1e88e5', label='p < 0.05', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d32f2f', label='p ≥ 0.05', markersize=10)
]
plt.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.savefig('coefficient_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Actual vs Predicted Values
plt.figure(figsize=(10, 8))
plt.scatter(y, model.predict(X), alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Burglary Rate', fontsize=14)
plt.ylabel('Predicted Burglary Rate', fontsize=14)
plt.title('Actual vs. Predicted Burglary Rates', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Residual Plot
plt.figure(figsize=(10, 8))
residuals = model.resid
plt.scatter(model.predict(X), residuals, alpha=0.6)
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted Values', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.title('Residual Plot', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('residual_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. QQ Plot for Residuals
plt.figure(figsize=(10, 8))
sm.qqplot(residuals, line='s', fit=True)
plt.title('Q-Q Plot of Residuals', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('qq_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Feature Importance - Based on absolute t-statistics
plt.figure(figsize=(12, 8))
importance_df = pd.DataFrame({
    'Feature': model.params.index[1:],  # Skip the constant
    'Importance': np.abs(model.tvalues.values[1:])
})
importance_df = importance_df.sort_values('Importance', ascending=True)

plt.barh(importance_df['Feature'], importance_df['Importance'], color='#1976d2')
plt.xlabel('Absolute t-statistic', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature Importance Based on t-statistics', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Partial Regression Plots for top features
fig = plt.figure(figsize=(15, 12))
sm.graphics.plot_partregress_grid(model, fig=fig)
plt.tight_layout()
plt.savefig('partial_regression_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Model Performance Metrics
metrics = pd.DataFrame({
    'Metric': ['R-squared', 'Adjusted R-squared', 'F-statistic', 'AIC', 'BIC'],
    'Value': [model.rsquared, model.rsquared_adj, model.fvalue, model.aic, model.bic]
})

plt.figure(figsize=(10, 6))
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
table = plt.table(cellText=metrics.values,
                  colLabels=metrics.columns,
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.4, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.2, 1.2)
plt.title('Model Performance Metrics', fontsize=16)
plt.tight_layout()
plt.savefig('model_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Correlation Heatmap of Features
corr_matrix = X.corr()
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='viridis',
            linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Influence Plot (to identify influential points)
fig, ax = plt.subplots(figsize=(12, 8))
sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
plt.title('Influence Plot - Leverage vs Standardized Residuals', fontsize=16)
plt.tight_layout()
plt.savefig('influence_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. Component+Residual (Partial Residual) Plots - Alternative approach
fig = plt.figure(figsize=(15, 12))
sm.graphics.plot_ccpr_grid(model, fig=fig)
plt.suptitle('Component-Component plus Residual (CCPR) Plots', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('ccpr_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. Residuals Distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel('Residual Value', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Residuals', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('residuals_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. Predicted vs Residuals with LOWESS smoothing to identify patterns
plt.figure(figsize=(10, 8))
predicted = model.predict(X)
sns.scatterplot(x=predicted, y=residuals, alpha=0.6)
sns.regplot(x=predicted, y=residuals, scatter=False, lowess=True,
            line_kws={'color': 'red', 'lw': 2})
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted Values', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.title('Predicted vs Residuals with LOWESS Smoothing', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lowess_residuals.png', dpi=300, bbox_inches='tight')
plt.show()