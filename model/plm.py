import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import joblib
from sklearn.model_selection import learning_curve


# load shapes
def load_shape_from_zip(zip_path, inside_path):
    gdf = gpd.read_file(f"zip://{zip_path}!{inside_path}")
    return gdf.to_crs(epsg=4326)


#Load geographical data
print("Loading geographical data...")
GEO_ZIP_PATH = r"C:\Users\tudor\Downloads\statistical-gis-boundaries-london.zip"
WARD_IN_ZIP = "statistical-gis-boundaries-london/ESRI/London_Ward_CityMerged.shp"
LSOA_IN_ZIP = "statistical-gis-boundaries-london/ESRI/LSOA_2011_London_gen_MHW.shp"

#load LSOA and Ward
lsoa_gdf = load_shape_from_zip(GEO_ZIP_PATH, LSOA_IN_ZIP)[["LSOA11CD", "geometry"]]
ward_gdf = load_shape_from_zip(GEO_ZIP_PATH, WARD_IN_ZIP)[["NAME", "geometry"]].rename(columns={"NAME": "Ward"})

ward_gdf = ward_gdf.to_crs(epsg=27700)
lsoa_gdf = lsoa_gdf.to_crs(epsg=27700)

print(f"Loaded {len(lsoa_gdf)} LSOAs and {len(ward_gdf)} Wards")

# create mapping
lsoa_centroids = lsoa_gdf.copy()
lsoa_centroids['geometry'] = lsoa_centroids.geometry.centroid

#find lsoas in wards
lsoa_ward_mapping = gpd.sjoin(lsoa_centroids, ward_gdf, how='left', predicate='within')
lsoa_ward_mapping = lsoa_ward_mapping[['LSOA11CD', 'Ward']].rename(columns={'LSOA11CD': 'LSOA code'})

#load combined datset
df = pd.read_csv('../data/combined_dataset.csv', low_memory=False)
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')

#filter post covid
burglary_df = df[(df['Crime type'] == 'Burglary') & (df['Month'] >= '2022-01-01')]
print(f"Filtered burglary data range: {burglary_df['Month'].min()} to {burglary_df['Month'].max()}")
print(f"Records after filtering: {len(burglary_df):,}")
burglary_df = burglary_df.merge(lsoa_ward_mapping, on='LSOA code', how='left')
burglary_df = burglary_df.dropna(subset=['Ward'])
print(f"Records after Ward mapping: {len(burglary_df):,}")

#group by ward, month
monthly_ward = (
    burglary_df.groupby(['Ward', 'Month'])
    .size()
    .reset_index(name='Burglary_Count')
)
print(f"Ward-level monthly data shape: {monthly_ward.shape}")

#aggregate risk
risk_df = pd.read_csv('../data/lsoa_risk_scores.csv')

ward_risk = (
    lsoa_ward_mapping.merge(risk_df, on='LSOA code', how='left')
    .groupby('Ward')
    .agg({
        'Risk_Score': ['mean', 'std', 'max', 'min', 'count']
    })
    .round(4)
)

ward_risk.columns = ['Risk_Score_Mean', 'Risk_Score_Std', 'Risk_Score_Max', 'Risk_Score_Min', 'LSOA_Count']
ward_risk = ward_risk.reset_index()
ward_risk['Risk_Score'] = ward_risk['Risk_Score_Mean']
ward_risk['Risk_Score_Std'] = ward_risk['Risk_Score_Std'].fillna(0)
monthly_ward = monthly_ward.merge(ward_risk, on='Ward', how='left')


def create_enhanced_features_ward(df):
    """
    Create  lag and seasonality features for Ward data
    """
    df = df.copy()
    df = df.sort_values(['Ward', 'Month']).reset_index(drop=True)

    #Extract time components
    df['Year'] = df['Month'].dt.year
    df['Month_Num'] = df['Month'].dt.month
    df['Quarter'] = df['Month'].dt.quarter

    #Seasonality Features
    df['Winter_Season'] = df['Month_Num'].isin([11, 12, 1, 2]).astype(int)
    df['Holiday_Season'] = df['Month_Num'].isin([12, 1]).astype(int)
    df['Summer_Season'] = df['Month_Num'].isin([6, 7, 8]).astype(int)

    #Cyclical encoding
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month_Num'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month_Num'] / 12)

    #quarter encoding
    df['Quarter_Sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
    df['Quarter_Cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)

    #seasonal features
    df['Days_Since_Winter'] = ((df['Month_Num'] - 1) % 12) * 30
    df['Days_To_Winter'] = ((13 - df['Month_Num']) % 12) * 30

    lag_columns = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_6', 'Lag_12', 'Lag_3_Avg', 'Lag_6_Avg', 'Lag_12_Avg',
                   'Lag_3_Std', 'Lag_6_Std', 'EWMA_3', 'EWMA_6', 'Trend_3', 'Trend_6',
                   'Seasonal_Lag_12', 'Max_3', 'Min_3', 'Max_6', 'Min_6',
                   'Current_to_Avg_3', 'Current_to_Avg_6', 'Volatility_3', 'Volatility_6']

    for col in lag_columns:
        df[col] = np.nan

    # Create ward lag feat
    for ward in df['Ward'].unique():
        ward_mask = df['Ward'] == ward
        ward_data = df.loc[ward_mask, 'Burglary_Count']
        ward_indices = df[ward_mask].index

        df.loc[ward_indices, 'Lag_1'] = ward_data.shift(1)
        df.loc[ward_indices, 'Lag_2'] = ward_data.shift(2)
        df.loc[ward_indices, 'Lag_3'] = ward_data.shift(3)
        df.loc[ward_indices, 'Lag_6'] = ward_data.shift(6)
        df.loc[ward_indices, 'Lag_12'] = ward_data.shift(12)

        #Rolling averages
        df.loc[ward_indices, 'Lag_3_Avg'] = ward_data.rolling(3, min_periods=1).mean().shift(1)
        df.loc[ward_indices, 'Lag_6_Avg'] = ward_data.rolling(6, min_periods=1).mean().shift(1)
        df.loc[ward_indices, 'Lag_12_Avg'] = ward_data.rolling(12, min_periods=1).mean().shift(1)

        # rolling standard deviations
        df.loc[ward_indices, 'Lag_3_Std'] = ward_data.rolling(3, min_periods=1).std().shift(1)
        df.loc[ward_indices, 'Lag_6_Std'] = ward_data.rolling(6, min_periods=1).std().shift(1)

        # Volatility
        rolling_3_mean = ward_data.rolling(3, min_periods=1).mean().shift(1)
        rolling_6_mean = ward_data.rolling(6, min_periods=1).mean().shift(1)
        rolling_3_std = ward_data.rolling(3, min_periods=1).std().shift(1)
        rolling_6_std = ward_data.rolling(6, min_periods=1).std().shift(1)

        df.loc[ward_indices, 'Volatility_3'] = rolling_3_std / (rolling_3_mean + 0.1)
        df.loc[ward_indices, 'Volatility_6'] = rolling_6_std / (rolling_6_mean + 0.1)

        #ewma
        df.loc[ward_indices, 'EWMA_3'] = ward_data.ewm(span=3).mean().shift(1)
        df.loc[ward_indices, 'EWMA_6'] = ward_data.ewm(span=6).mean().shift(1)

        # Trend
        df.loc[ward_indices, 'Trend_3'] = ward_data.shift(1) - ward_data.shift(4)
        df.loc[ward_indices, 'Trend_6'] = ward_data.shift(1) - ward_data.shift(7)

        # Seasonal lag
        df.loc[ward_indices, 'Seasonal_Lag_12'] = ward_data.shift(12)

        #Min/Max on recent periods
        df.loc[ward_indices, 'Max_3'] = ward_data.rolling(3, min_periods=1).max().shift(1)
        df.loc[ward_indices, 'Min_3'] = ward_data.rolling(3, min_periods=1).min().shift(1)
        df.loc[ward_indices, 'Max_6'] = ward_data.rolling(6, min_periods=1).max().shift(1)
        df.loc[ward_indices, 'Min_6'] = ward_data.rolling(6, min_periods=1).min().shift(1)

        #ratios
        df.loc[ward_indices, 'Current_to_Avg_3'] = ward_data.shift(1) / (rolling_3_mean + 0.1)
        df.loc[ward_indices, 'Current_to_Avg_6'] = ward_data.shift(1) / (rolling_6_mean + 0.1)

    std_cols = [col for col in df.columns if '_Std' in col or 'Volatility' in col]
    df[std_cols] = df[std_cols].fillna(0)

    #interaction feat
    interaction_features = ['Lag_1', 'Lag_3_Avg', 'Winter_Season', 'EWMA_3', 'Summer_Season']
    for col in interaction_features:
        if col in df.columns:
            df[f'{col}_x_Risk'] = df[col] * df['Risk_Score']

    risk_variability_features = ['Lag_1', 'Lag_3_Avg', 'EWMA_3']
    for col in risk_variability_features:
        if col in df.columns:
            df[f'{col}_x_RiskStd'] = df[col] * df['Risk_Score_Std']

    return df


monthly_ward_enhanced = create_enhanced_features_ward(monthly_ward)

#define the features creates
feature_cols = [
    # Basic lag features
    'Lag_1','Lag_6', 'Lag_12',

    # Rolling averages
    'Lag_3_Avg', 'Lag_6_Avg', 'Lag_12_Avg',

    # Exponential weighted averages
    'EWMA_3', 'EWMA_6',

    # Trend features
    'Trend_6',

    # Seasonal features
    'Winter_Season',  'Summer_Season'

    # Risk features
    'Risk_Score', 'Risk_Score_Std', 'Risk_Score_Max', 'Risk_Score_Min', 'LSOA_Count',

    # Interaction features
    'Lag_1_x_Risk', 'Lag_3_Avg_x_Risk', 'Winter_Season_x_Risk', 'EWMA_3_x_Risk', 'Summer_Season_x_Risk',
    'Lag_1_x_RiskStd', 'Lag_3_Avg_x_RiskStd', 'EWMA_3_x_RiskStd',

    # Min/Max features
    'Max_6',

    # Seasonal lag
    'Seasonal_Lag_12'
]

available_features = [col for col in feature_cols if col in monthly_ward_enhanced.columns]
print(f"Using {len(available_features)} features for Ward-level modeling")

#rmv rows with missing values
essential_cols = ['Lag_1', 'Risk_Score']
monthly_ward_enhanced = monthly_ward_enhanced.dropna(subset=essential_cols + ['Burglary_Count'])

#feat and target var
X = monthly_ward_enhanced[available_features]
y = monthly_ward_enhanced['Burglary_Count']

#time train-test
split_date = monthly_ward_enhanced['Month'].quantile(0.7)
train_mask = monthly_ward_enhanced['Month'] <= split_date
test_mask = monthly_ward_enhanced['Month'] > split_date

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train period: {split_date}")


monthly_ward_enhanced.to_csv("data/prepared_ward_time_series.csv", index=False)
lsoa_ward_mapping.to_csv("data/lsoa_ward_mapping.csv", index=False)
ward_risk.to_csv("data/ward_risk_scores.csv", index=False)

#feature select
print("FEATURE SELECTION")

# base rfecv model
base_model_for_rfecv = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-2
)

#cv
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# run RFECV
print("Starting RFECV feature selection for Ward-level data...")
rfecv = RFECV(
    estimator=base_model_for_rfecv,
    step=1,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    min_features_to_select=5,
    n_jobs=-1,
    verbose=1
)

rfecv.fit(X_train, y_train)

#Get selected features
selected_features = list(X_train.columns[rfecv.support_])
print(f"\nRFECV selected {len(selected_features)} features:")
print(selected_features)

#Update training data to use only selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

#Optuna

print("OPTUNA")



def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization
    """
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 300, 2000),
        'max_depth': trial.suggest_int("max_depth", 3, 20),
        'learning_rate': trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        'reg_alpha': trial.suggest_float("reg_alpha", 0, 50, log=False),
        'reg_lambda': trial.suggest_float("reg_lambda", 0, 50, log=False),
        'gamma': trial.suggest_float("gamma", 0, 20),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        'colsample_bynode': trial.suggest_float("colsample_bynode", 0.5, 1.0),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 30),
        'max_delta_step': trial.suggest_int("max_delta_step", 0, 10),
        'scale_pos_weight': trial.suggest_float("scale_pos_weight", 0.5, 2.0),
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }

    #Create and train model
    model = XGBRegressor(**params)
    model.fit(X_train_selected, y_train)

    # make predictions and calculate RMSE
    preds = model.predict(X_test_selected)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return rmse


#Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

print(f"Best RMSE: {study.best_value:.4f}")
print(f"Best parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

#train final model
print("TRAIN FINAL WARD MODEL")
final_ward_model = XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)

final_ward_model.fit(X_train_selected, y_train)

# Make predictions
train_preds = final_ward_model.predict(X_train_selected)
test_preds = final_ward_model.predict(X_test_selected)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
train_mae = mean_absolute_error(y_train, train_preds)
test_mae = mean_absolute_error(y_test, test_preds)
train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)

print("performance")
print(f"Selected Features: {len(selected_features)}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Training MAE: {train_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Overfitting Check: {test_rmse / train_rmse:.4f}")

#save model and redults
ward_results = {
    'selected_features': selected_features,
    'best_params': study.best_params,
    'performance_metrics': {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2
    },
    'ward_count': monthly_ward_enhanced['Ward'].nunique(),
    'lsoa_ward_mapping_available': True
}

joblib.dump(final_ward_model, '../data/final_ward_burglary_model.pkl')
joblib.dump(ward_results, '../data/ward_model_results.pkl')

print("TRAINING COMPLETE!")
print("1. data/final_ward_burglary_model.pkl - Trained Ward-level model")
print("2. data/ward_model_results.pkl - Results and parameters")
print("3. data/prepared_ward_time_series.csv - Prepared Ward-level data")
print("4. data/lsoa_ward_mapping.csv - LSOA to Ward mapping")
print("5. data/ward_risk_scores.csv - Ward-level risk scores")

#Generate PREDICTIONS

print("GENERATE 12 MONTH PREDICTIONS")

def generate_future_predictions(model, data, selected_features, n_months=12):
    """
    Generate predictions for the next n_months per Ward
    """
    #Get  latest date in the data
    latest_date = data['Month'].max()
    print(f"Latest date in data: {latest_date}")

    #Create future dates
    future_dates = pd.date_range(
        start=latest_date + pd.DateOffset(months=1),
        periods=n_months,
        freq='MS'  # Month Start
    )

    print(f"Generating predictions for: {future_dates[0]} to {future_dates[-1]}")

    wards = data['Ward'].unique()
    predictions_list = []
    extended_data = data.copy()

    for month_idx, future_date in enumerate(future_dates):
        print(f"Predicting month {month_idx + 1}/{n_months}: {future_date.strftime('%Y-%m')}")

        month_predictions = []

        for ward in wards:
            # get historical data
            ward_data = extended_data[extended_data['Ward'] == ward].copy()

            if len(ward_data) == 0:
                continue

            #create new row for this Ward and future month
            new_row = ward_data.iloc[-1:].copy()
            new_row['Month'] = future_date
            new_row['Year'] = future_date.year
            new_row['Month_Num'] = future_date.month
            new_row['Quarter'] = future_date.quarter

            # update seasonal features
            new_row['Winter_Season'] = int(future_date.month in [11, 12, 1, 2])
            new_row['Holiday_Season'] = int(future_date.month in [12, 1])
            new_row['Summer_Season'] = int(future_date.month in [6, 7, 8])

            # Cyclical encoding
            new_row['Month_Sin'] = np.sin(2 * np.pi * future_date.month / 12)
            new_row['Month_Cos'] = np.cos(2 * np.pi * future_date.month / 12)
            new_row['Quarter_Sin'] = np.sin(2 * np.pi * future_date.quarter / 4)
            new_row['Quarter_Cos'] = np.cos(2 * np.pi * future_date.quarter / 4)

            # Advanced seasonal features
            new_row['Days_Since_Winter'] = ((future_date.month - 1) % 12) * 30
            new_row['Days_To_Winter'] = ((13 - future_date.month) % 12) * 30

            # get ward tinmeries
            ward_extended = extended_data[extended_data['Ward'] == ward].copy()
            ward_series = ward_extended['Burglary_Count']

            #recalc lags
            if len(ward_series) >= 1:
                new_row['Lag_1'] = ward_series.iloc[-1]
            if len(ward_series) >= 2:
                new_row['Lag_2'] = ward_series.iloc[-2]
            if len(ward_series) >= 3:
                new_row['Lag_3'] = ward_series.iloc[-3]
            if len(ward_series) >= 6:
                new_row['Lag_6'] = ward_series.iloc[-6]
            if len(ward_series) >= 12:
                new_row['Lag_12'] = ward_series.iloc[-12]
                new_row['Seasonal_Lag_12'] = ward_series.iloc[-12]

            # Rolling averages
            if len(ward_series) >= 3:
                new_row['Lag_3_Avg'] = ward_series.tail(3).mean()
                new_row['Lag_3_Std'] = ward_series.tail(3).std() if len(ward_series) >= 3 else 0
                new_row['Max_3'] = ward_series.tail(3).max()
                new_row['Min_3'] = ward_series.tail(3).min()
                new_row['Volatility_3'] = (ward_series.tail(3).std() / (ward_series.tail(3).mean() + 0.1))

            if len(ward_series) >= 6:
                new_row['Lag_6_Avg'] = ward_series.tail(6).mean()
                new_row['Lag_6_Std'] = ward_series.tail(6).std() if len(ward_series) >= 6 else 0
                new_row['Max_6'] = ward_series.tail(6).max()
                new_row['Min_6'] = ward_series.tail(6).min()
                new_row['Volatility_6'] = (ward_series.tail(6).std() / (ward_series.tail(6).mean() + 0.1))

            if len(ward_series) >= 12:
                new_row['Lag_12_Avg'] = ward_series.tail(12).mean()

            # EWMA features
            if len(ward_series) >= 3:
                new_row['EWMA_3'] = ward_series.ewm(span=3).mean().iloc[-1]
            if len(ward_series) >= 6:
                new_row['EWMA_6'] = ward_series.ewm(span=6).mean().iloc[-1]

            # Trend features
            if len(ward_series) >= 4:
                new_row['Trend_3'] = ward_series.iloc[-1] - ward_series.iloc[-4]
            if len(ward_series) >= 7:
                new_row['Trend_6'] = ward_series.iloc[-1] - ward_series.iloc[-7]

            # Ratio features
            if len(ward_series) >= 3:
                recent_avg_3 = ward_series.tail(4).head(3).mean()  # Average of 3 months before last
                new_row['Current_to_Avg_3'] = ward_series.iloc[-1] / (recent_avg_3 + 0.1)

            if len(ward_series) >= 6:
                recent_avg_6 = ward_series.tail(7).head(6).mean()  # Average of 6 months before last
                new_row['Current_to_Avg_6'] = ward_series.iloc[-1] / (recent_avg_6 + 0.1)

            numeric_cols = new_row.select_dtypes(include=[np.number]).columns
            new_row[numeric_cols] = new_row[numeric_cols].fillna(0)

            # Update interaction features
            interaction_features = ['Lag_1', 'Lag_3_Avg', 'Winter_Season', 'EWMA_3', 'Summer_Season']
            for col in interaction_features:
                if col in new_row.columns and 'Risk_Score' in new_row.columns:
                    new_row[f'{col}_x_Risk'] = new_row[col] * new_row['Risk_Score']

            risk_variability_features = ['Lag_1', 'Lag_3_Avg', 'EWMA_3']
            for col in risk_variability_features:
                if col in new_row.columns and 'Risk_Score_Std' in new_row.columns:
                    new_row[f'{col}_x_RiskStd'] = new_row[col] * new_row['Risk_Score_Std']

            # Make prediction using selected features only
            try:
                features_for_prediction = new_row[selected_features].fillna(0)
                prediction = model.predict(features_for_prediction.values.reshape(1, -1))[0]

                # Ensure prediction is not negative
                prediction = max(0, prediction)

                # Store prediction
                new_row['Burglary_Count'] = prediction
                new_row['Prediction_Date'] = future_date
                new_row['Is_Prediction'] = True

                month_predictions.append({
                    'Ward': ward,
                    'Month': future_date,
                    'Predicted_Burglaries': round(prediction, 2),
                    'Month_Name': future_date.strftime('%B %Y'),
                    'Year': future_date.year,
                    'Month_Num': future_date.month,
                    'Quarter': future_date.quarter,
                    'Winter_Season': int(future_date.month in [11, 12, 1, 2]),
                    'Holiday_Season': int(future_date.month in [12, 1]),
                    'Summer_Season': int(future_date.month in [6, 7, 8])
                })

            except Exception as e:
                print(f"Error predicting for Ward {ward} in {future_date}: {e}")
                continue

        #add this months predictions to the extended data
        if month_predictions:
            month_df = pd.DataFrame(month_predictions)
            #convert format
            for _, pred_row in month_df.iterrows():
                new_extended_row = extended_data[extended_data['Ward'] == pred_row['Ward']].iloc[-1:].copy()
                new_extended_row['Month'] = pred_row['Month']
                new_extended_row['Burglary_Count'] = pred_row['Predicted_Burglaries']
                extended_data = pd.concat([extended_data, new_extended_row], ignore_index=True)

        predictions_list.extend(month_predictions)

    return pd.DataFrame(predictions_list)


# Generate 12-month predictions
print("Generating 12-month predictions")
future_predictions = generate_future_predictions(
    model=final_ward_model,
    data=monthly_ward_enhanced,
    selected_features=selected_features,
    n_months=12
)

print(f"Generated predictions for {len(future_predictions)} Ward-month combinations")

#summary stats
print("PREDICTION SUMMARY")
# Monthly totals
monthly_totals = future_predictions.groupby('Month_Name')['Predicted_Burglaries'].agg(['sum', 'mean', 'std']).round(2)
print("\nMonthly Totals All Wards:")
print(monthly_totals)

#ward totals for next 12 months
ward_totals = future_predictions.groupby('Ward')['Predicted_Burglaries'].agg(['sum', 'mean', 'std']).round(2)
ward_totals = ward_totals.sort_values('sum', ascending=False)
print(f"\nTop 10 Wards by Total Predicted Burglaries (Next 12 Months):")
print(ward_totals.head(10))

# Seasonal analysis
seasonal_totals = future_predictions.groupby(['Winter_Season', 'Summer_Season']).agg({
    'Predicted_Burglaries': ['sum', 'mean', 'count']
}).round(2)
print(f"\nSeasonal Statistics:")
print(seasonal_totals)

#Save predictions
predictions_filename = '../data/ward_burglary_predictions_12months.csv'
future_predictions.to_csv(predictions_filename, index=False)

#pivot for analysis
pivot_predictions = future_predictions.pivot(index='Ward', columns='Month_Name', values='Predicted_Burglaries').fillna(
    0)
pivot_filename = '../data/ward_burglary_predictions_pivot.csv'
pivot_predictions.to_csv(pivot_filename)

#Create summary report
summary_report = pd.DataFrame({
    'Ward': ward_totals.index,
    'Total_12_Months': ward_totals['sum'],
    'Average_Monthly': ward_totals['mean'],
    'Std_Monthly': ward_totals['std'],
    'Risk_Level': pd.cut(ward_totals['sum'],
                         bins=[0, 50, 100, 200, float('inf')],
                         labels=['Low', 'Medium', 'High', 'Very High'])
})

summary_filename = '../data/ward_risk_summary_12months.csv'
summary_report.to_csv(summary_filename, index=False)

print("Prediction files saved:")
print(f"1. {predictions_filename} - Detailed monthly predictions")
print(f"2. {pivot_filename} - Pivot table format")
print(f"3. {summary_filename} - Ward risk summary")
print(f"\nTotal predictions generated: {len(future_predictions):,}")
print(f"Date range: {future_predictions['Month'].min()} to {future_predictions['Month'].max()}")
print(f"Average monthly burglaries across all wards: {future_predictions['Predicted_Burglaries'].mean():.2f}")
print(f"Total predicted burglaries (next 12 months): {future_predictions['Predicted_Burglaries'].sum():.0f}")

monthly_totals = future_predictions.groupby('Month_Name')['Predicted_Burglaries'].sum().reindex(
    future_predictions['Month_Name'].unique(), fill_value=0)

