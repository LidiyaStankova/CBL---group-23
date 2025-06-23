# Addressing real-world crime and security problems with data science
## Multi-Disciplinary CBL 4CBLW00-2024-2025
> Group 23

# ℹ Project Description
Burglary in London represents a critical public-safety challenge driven by socio-economic disparities and seasonal crime patterns. In this report, we present a data-driven framework that integrates a linear risk score with an XGBoost time-series model, leveraging deprivation and demographic data at the LSOA level to forecast monthly burglary risk. Results are interpretable for both technical and non-technical stakeholders by making use of the interactive dashboard. Forecasts are used to feed a constrained optimisation model that distributes police patrol hours among LSOAs and wards while following fairness standards, personnel limitations, and shift constraints. The findings show that by efficiently focusing on high-risk regions and guaranteeing fair distribution, the strategy can improve resource allocation.

> This repository ingests UK burglary data, Census 2021, Index of Deprivation (IoD) 2019, and other external data.

## 🧹 Data Cleaning and Processing
Burglary dataset `burglary-cleaned.csv` and other datasets that are cleaned and processed can be found in directory `explainability` in the `.zip` folder when you download the main directly from GitHub repository link provided.
The original datasets can be cleaned and processed by studying the steps in `../explainability/explainable.ipynb` (adjust the datasets and directory paths at your convenience and purpose). Then, datasets can be correctly joined without data leakage.

- Police.uk. (2025). Police.uk open data crime dataset [Data set]. Retrieved June 13, 2025, from http://police.uk/

- Census 2021, Office for National Statistics (2022)
   -  Office for National Statistics. (2022). Age on arrival in the UK by country of birth (TS018) (Version 3) [Data set]. Retrieved June 13, 2025, from https://www.ons.gov.uk/datasets/TS018/editions/2021/versions/3
   - Office for National Statistics. (2022). Census 2021: Country of birth [Data set]. Retrieved June 13, 2025, from https://www.data.gov.uk/dataset/37a15fe5-53b4-4e42-9e56-fe7912f514ff/census-2021-country-of-birth
   - Office for National Statistics. (2022). Economic activity status, England and Wales: Census 2021 [Data file]. Retrieved June 13, 2025, from https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/bulletins/economicactivitystatusenglandandwales/census2021
   - Office for National Statistics. (2022). Household composition (TS003) (Version 4) [Data set]. Retrieved June 13, 2025, from https://www.ons.gov.uk/datasets/TS003/editions/2021/versions/4
   - (Optional) 
      - Office for National Statistics. (2022). *Households by deprivation dimensions (TS011) (Version 6)* [Data set]. Retrieved June 13, 2025, from https://cy.ons.gov.uk/datasets/TS011/editions/2021/versions/6
      - Office for National Statistics. (2025). *Nature of crime: Burglary* [Data set]. Retrieved June 13, 2025, from https://www.ons.gov.uk/peoplepopulationandcommunity/crimeandjustice/datasets/natureofcrimeburglary

- IoD 2019 (13 files)
   - Ministry of Housing, Communities & Local Government. (2019). English indices of deprivation 2019: Explanation of deprivation data [Data set]. Retrieved June 13, 2025, from https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019
      - Main Findings: https://assets.publishing.service.gov.uk/media/5d8e26f6ed915d5570c6cc55/IoD2019_Statistical_Release.pdf
      - Research report (how to interpret the data): https://assets.publishing.service.gov.uk/media/5d8b364ced915d03709e3cf2/IoD2019_Research_Report.pdf
      - Frequently Asked Questions: https://assets.publishing.service.gov.uk/media/5dfb3d7ce5274a3432700cf3/IoD2019_FAQ_v4.pdf

- HM Revenue & Customs. (2013). Children in Poverty NI116: poverty_2013_update.xls [Data set]. Retrieved June 21, 2025, from https://data.london.gov.uk/dataset/children-poverty-ni116/

- Greater London Authority. (2025). Statistical GIS boundary files for London [Data set]. https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london/
   - `London-wards-2018.zip`
   - `statistical-gis-boundaries-london.zip`


# 🎬 Setting Up
## Environment
> **Note:** The `../config/environment.yml` files assume the environments has conda installed specifying `python=3.11.11+`, which uses conda chaneels. 
They are not directly compatiable with pip-only setups. 
- `conda env create -f environment.yml`
- `conda activate py311`

> Pip-only Setup
`pip install -r requirements.txt`
- `.gitignore` (files to be excluded from version control)


# 🕳️ Project Structure
## OLS Linear Regression (LSAO Level Risk)
Run `python lso_risk_regr.py`.
The risk score for each LSOA in London is computed and full regression summary is printed.
- `../model/lso_risk_regr.py`
### 🔧 Model Pipeline
1. `combined_dataset.csv` (not included in the `.zip` file)
2. Deprivation indicators (IoD) `File_6_-_IoD2019_Population_Denominators.xlsx`: income, education, health, crime, and living environment deciles
3. Poverty `poverty_2013_update.xls`: 2006 vs. 2010
4. Census 2021: `Household composition.xlsx`, `Age on arrival in UK.xlsx`, and `Economic Activity.xlsx`
5. Linear risk score is predicted after fitting an OLS regression of burglary rates per 1,000 residents 

> Recommended to rerun quarterly if new IMD or housing data is available.

## XGBoost Model + Optuna Time-Series Forecast (Ward Level)
Run `python plm.py`.
This model ingests historical burglary counts and static “risk scores". It builds lagged time-series features, tunes an XGBoost regressor with Optuna, and produces out-of-sample forecasts and demand-scores for downstream allocation.
- `../model/plm.py`
The model forecasts residential burglaries in next 12 months.

### 🔧 Model Pipeline
1. **Data Preparation**  
   - **Filter** `combined_dataset.csv` to burglary from Jan 2022 onward
   - Geographical data of LSOA-ward mapping are loaded.
   - **Aggregate** to monthly counts per ward  
   - **Merge** in `lsoa_risk_scores.csv` to compute mean, std, min, and max
   - **Feature Engineering** temporal features (year, month, quarter, seasons, cyclical encodings) through RFECV
   - **Create** lagged features (1, 2, 3, 6, 12), rolling stats. EWMA, trend, volatility, and interaction with risk
   - **Hyperparameter Tuning** Optuna (minimising RMSE over 5 trials)
   - **Drop** any rows with missing lags or risk scores  
2. **Train/Test Split**  
   - Trained the XGBRegressor on selected features (RMSE, MAE, and $R^2$ are reported on both splits)
3. **12-month Forecasts**
   - Monthly recursive forecasting per ward is saved
   - Outputs `ward_burglary_predictions_12months.csv`

Justification and assumptions of the model can be found in the report. Similarly for hyper parameter tuning, final model fit, and prediction are explained to subsequently evaluate the metrics.

> Suggested to retrain the model monthly as soon as prior month's crime figures are available.

## Police Resource Allocation Model
Running `python Patrol_Allocation_Model.py` will produce `../data/allocation_halfyear.csv` which can be ingested into the dashboard.

1. **Load inputs**  
   - `ward_burglary_predictions_12months.csv`
      - pandas DataFrame with `Month`, `Ward`, `Predicted_Burglaries`.  
   - `ward_lsoa_with_area.csv`
      - merges ward - LSOA and area (km²).
2. **Set monthly bounds** 
   - Maximise $\sum_{i}{r_i \alpha_i (1 + X_i)}
   - Total ward has a budget cap of 3,200 h
   - Minimum 1 h per day per LSOA
   ```python
      days = tm.days_in_month
      base_max = 2.6 * days
      base_min = 1.0 * days
      (hours per LSOA)
3. **Output**
   - Allocations are rounded up to the nearest 0.25 h into a combined `csv` file

Detailed computation of area-scaled caps, weights, and optimisation can be found in the report. 
It uses the latest 12 month forecast and produces patrol-hour allocations for next 6 months per ward and LSOA.

- `allocation_halfyear.csv`: Model output (6 months)
- `../model/Patrol_Allocation_Model.py`: Allocation solver
- `../model/Patrol_Allocation_Model.ipynb`
- `../model/data/statistical-gis-boundaries-london.zip`: Borough & ward shapefiles
- `../model/data/ward_burglary_predictions_12months.csv`: XGBoost forecast output
- `../model/data/ward_lsoa_mapping.csv`: Ward - LSOA lookup
- `../model/data/ward_lsoa_with_area.csv`:: LSOA areas ($km^2$)

> Recommended to monthly rerun the model immediately after XGBoost model runs.

## Dashboard
Run `streamlit run Overview.py`
> All heavy I/O and computations are cached.

- `../dashboard_cleaned/data`
   - `../processed`
      - `processed_burglaries.parquet`
      - `ward_burglary.feather`
   - `../raw`
      - `../LB_shp`: London borough/ward shapefiles
   - `../Patrol_Allocation_Model.py`
- `../dashboard_cleaned/other`
   - `constants.py`: File paths and global settings.
- `../dashboard_cleaned/pages`
   - `1_insights.py`: Exploratory analysis of correlation charts & static risk‐score maps
   - `2_allocation.py`: Allocation model interface & outputs (interactive patrol hours)
   - `3_Resources.py`: Detailed resource tables (downloadable)
- `../dashboard_cleaned/Overview.py`: Time‐slider overview of actual vs. predicted

## Explainability Report
### Minimal Predictors
1. Burglary
   - `lsoa_code`
   - `month`
   - `burglary_count`
2. Risk-score / deprivation
   - `imd_score`
   - domains: `income_score`, `employment_score`, `education_score`, `health_score`, `crime_score`, `barriers_score`, `living_environment_score`
   - supplementary: `idaci_score`, `idaopi_score`
3. Demographics
   - `% under-18`, `% over-65`
   - `% economically active` vs. `% inactive` (or just `% unemployed`)
   - `% single-person households` vs. `% families with children`
4. Housing & environment
   - `population_per_hectare` (housing density)
   - `price` or `imd_score` from HMLRPPD
   - child poverty

### Layout
- `../explainability/environment.gpu.yml`
- `../explainability/notebooks/`
   - `explainable.ipynb`: main notebook to study which contains the SHAP plots
   - `burg_cleaning.ipynb`: burglarly data cleaning steps
   - `onspd-ppd-cleaned.ipynb`: housing data preprocessing steps
      > HM Land Registry. (2025). Price Paid Data (PPD) monthly downloadable data [Data set]. Retrieved June 13, 2025, from https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads

## Statistical Model
The `Statistical_Model.py` script fits a straightforward (log–linear) OLS model to estimate a static burglary‐risk score for each LSOA, based on socio-environmental predictors. Running the script will produce `../log_burglary_risk_scores_v2.csv`.

The ingested all-in-one file `FinalDataC.csv` is not included.
- `../model/Statistical_Model.py`
- `../model/Statistical_Model_1.ipynb`
- `../model/Statistical_Model_2.ipynb`
- `../model/lsoa_risk_scores.csv`
- `../model/log_burglarly_risk_scores_v2.csv`: 
   - Log risk score: model prediction in log-rate
   - Exp risk score: back-transformed "burglaries per 1,000" score


# Report
Key transformations (OLS, XGBoost features, and Allocation), feature engineering, performance diagnostics, and hyperparamter ranges through Optuna are all elaborated in the report, therefore, this README.md will only contain the setup providing data collection guide, processing pipeline, and how-to-use instructions of the models.

# Assumptions and Constraints
1. OLS (`lso_risk_regre.py`)
   - Linearity and additive effects
   - No multicollinearity
   - Independe of residuals
2. XGBoost (`plm.py`)
   - Stationary over training
   - No major policy changes
3. Allocation (`Patrol_Allocation_Model.py`)
   - Diminishing returns captured by $ln(1 + X)$
   - Exchangable patrol hours within a ward
   - Min/max bounds ensure safety and duo-patrol operations

# Downstream Integraion of Outputs
- Statistical model's `log_burglary_risk_scores_v2.csv` file is ingested into `1_insights.py` in `dashboard_cleaned` directory.
- Resulted `ward_burglary_predictions_12months.csv` file of XGBoost model is ingested into `Overview.py` in `dashboard_cleaned` directory.
- `data/allocation_halfyear.csv` is ingested into `2_allocation.py` and `3_Resources.py`.

### 🚨 Disclaimer
This guide is created with support of ChatGPT o4-mini model and VSCode's Copilot.