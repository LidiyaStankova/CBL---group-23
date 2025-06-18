import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import plot_tree

#load combined dataset
df = pd.read_csv('data/combined_dataset.csv', low_memory=False)
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')

#filter to burglaries
burglary_df = df[df['Crime type'] == 'Burglary']

#grouping
monthly = (
    burglary_df.groupby(['LSOA code', 'Month'])
    .size()
    .reset_index(name='Burglary_Count')
)

#load risk scores
risk_df = pd.read_csv('data/lsoa_risk_scores.csv')

# Merge risk score into monthly data
monthly = monthly.merge(risk_df, on='LSOA code', how='left')

#create ag features
monthly = monthly.sort_values(['LSOA code', 'Month'])
monthly['Lag_1'] = monthly.groupby('LSOA code')['Burglary_Count'].shift(1)
monthly['Lag_3_Avg'] = (
    monthly.groupby('LSOA code')['Burglary_Count']
    .rolling(3, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
)

# Drop rows with NA lags
monthly = monthly.dropna(subset=['Lag_1', 'Lag_3_Avg', 'Risk_Score'])

#save preped data
monthly.to_csv("data/prepared_time_series.csv", index=False)

full_ts = pd.read_csv("data/prepared_time_series.csv")

X = full_ts[['Lag_1', 'Lag_3_Avg', 'Risk_Score']]
y = full_ts['Burglary_Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Define objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 100, 1000),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'reg_alpha': trial.suggest_float("reg_alpha", 0, 10),
        'reg_lambda': trial.suggest_float("reg_lambda", 0, 10),
    }

    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

# Run study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_params)

#train
best_model = XGBRegressor(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)
final_preds = best_model.predict(X_test)

# Save predictions
X_test_copy = X_test.copy()
X_test_copy['Predicted_Burglary'] = final_preds
X_test_copy['Actual_Burglary'] = y_test.values
X_test_copy['Demand_Score'] = final_preds * X_test_copy['Risk_Score']
X_test_copy.to_csv("data/demand_scores_optuna.csv")

# Visualizations
residuals = y_test - final_preds

# Plot 1: Actual vs Predicted
plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label="Actual", alpha=0.6)
plt.plot(final_preds, label="Predicted", alpha=0.7)
plt.title("Actual vs Predicted Burglary Counts")
plt.xlabel("Time Index")
plt.ylabel("Burglary Count")
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Residuals
plt.figure(figsize=(12, 5))
sns.histplot(residuals, bins=50, kde=True, color='salmon')
plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Plot 3: Feature Importance
importances = best_model.feature_importances_
features = X.columns
plt.figure(figsize=(8, 4))
sns.barplot(x=importances, y=features)
plt.title("XGBoost Feature Importance (Optuna-tuned)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


def plot_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100  # Adding 1 to avoid division by zero

    metrics = ['RMSE', 'MAE', 'MAPE (%)', 'R²']
    values = [rmse, mae, mape, r2]

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=metrics, y=values, palette='viridis')
    plt.title('Model Performance Metrics', fontsize=15)
    plt.ylabel('Value')

    # Add value labels on top of bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()


# 2. Prediction Error Analysis
def plot_error_analysis(y_true, y_pred):
    # Create dataframe for analysis
    error_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Error': y_true - y_pred,
        'Abs_Error': np.abs(y_true - y_pred),
        'Percentage_Error': np.abs((y_true - y_pred) / (y_true + 1)) * 100  # Adding 1 to avoid division by zero
    })

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Error vs Actual plot
    axes[0, 0].scatter(error_df['Actual'], error_df['Error'], alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[0, 0].set_title('Error vs Actual Value', fontsize=14)
    axes[0, 0].set_xlabel('Actual Burglary Count')
    axes[0, 0].set_ylabel('Prediction Error')

    # Absolute Error Distribution
    sns.histplot(error_df['Abs_Error'], bins=30, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Absolute Error Distribution', fontsize=14)
    axes[0, 1].set_xlabel('Absolute Error')

    # Scatter plot with color-coded errors
    scatter = axes[1, 0].scatter(error_df['Actual'], error_df['Predicted'],
                                 c=error_df['Abs_Error'], cmap='YlOrRd', alpha=0.7)
    lims = [0, max(error_df['Actual'].max(), error_df['Predicted'].max()) * 1.1]
    axes[1, 0].plot(lims, lims, 'k-', alpha=0.4)
    axes[1, 0].set_title('Actual vs Predicted with Error Intensity', fontsize=14)
    axes[1, 0].set_xlabel('Actual Burglary Count')
    axes[1, 0].set_ylabel('Predicted Burglary Count')
    plt.colorbar(scatter, ax=axes[1, 0], label='Absolute Error')

    # QQ plot of errors to check normality
    from scipy import stats
    stats.probplot(error_df['Error'], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Errors', fontsize=14)

    plt.tight_layout()
    plt.show()

    return error_df


# 3. Permutation Feature Importance (more reliable than default feature importance)
def plot_permutation_importance(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    perm_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=perm_importance_df, palette='viridis')
    plt.title('Permutation Feature Importance', fontsize=15)
    plt.xlabel('Mean Decrease in RMSE')
    plt.tight_layout()
    plt.show()

    return perm_importance_df


# 4. SHAP Values for Model Interpretation
def plot_shap_analysis(model, X):
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance', fontsize=15)
    plt.tight_layout()
    plt.show()

    # Detailed SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title('SHAP Value Distribution by Feature', fontsize=15)
    plt.tight_layout()
    plt.show()

    # SHAP dependence plots for each feature
    for feature in X.columns:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X, show=False)
        plt.title(f'SHAP Dependence for {feature}', fontsize=15)
        plt.tight_layout()
        plt.show()


# 5. Learning Curves and Prediction Confidence
def plot_learning_curve(model, X_train, y_train, X_test, y_test):
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training RMSE")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation RMSE")
    plt.xlabel("Training examples")
    plt.ylabel("RMSE")
    plt.title("Learning Curves", fontsize=15)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 6. Visualize spatial distribution of errors (if geographic data is available)
def plot_geographical_errors(error_df, geo_df):
    # This assumes you have a GeoDataFrame with LSOA boundaries
    # Merge errors with geographic data
    if 'LSOA code' in error_df.columns and geo_df is not None:
        geo_error_df = geo_df.merge(error_df, on='LSOA code')

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        # Plot Mean Absolute Error by area
        geo_error_df.plot(column='Abs_Error', cmap='Reds',
                          legend=True, ax=ax[0])
        ax[0].set_title('Spatial Distribution of Absolute Error', fontsize=15)
        ax[0].set_axis_off()

        # Plot Error Direction (over/under prediction)
        geo_error_df.plot(column='Error', cmap='RdBu_r',
                          legend=True, ax=ax[1])
        ax[1].set_title('Spatial Distribution of Error Direction', fontsize=15)
        ax[1].set_axis_off()

        plt.tight_layout()
        plt.show()


# 7. Visualize XGBoost Trees
def plot_xgb_trees(model, num_trees=3):
    # Plot the first few trees
    for i in range(min(num_trees, model.n_estimators)):
        plt.figure(figsize=(20, 15))
        plot_tree(model, num_trees=i)
        plt.title(f'XGBoost Tree #{i}', fontsize=15)
        plt.tight_layout()
        plt.show()


# 8. Risk-Demand Relationship Analysis
def plot_risk_demand_relationship(X_test_copy):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Risk_Score', y='Predicted_Burglary',
                    size='Actual_Burglary', sizes=(20, 200),
                    alpha=0.7, data=X_test_copy)
    plt.title('Risk Score vs Predicted Burglary Count', fontsize=15)
    plt.xlabel('Risk Score')
    plt.ylabel('Predicted Burglary Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Risk score bins analysis
    plt.figure(figsize=(12, 6))
    X_test_copy['Risk_Bin'] = pd.qcut(X_test_copy['Risk_Score'], 5)
    sns.boxplot(x='Risk_Bin', y='Actual_Burglary', data=X_test_copy)
    plt.title('Actual Burglary by Risk Score Quintile', fontsize=15)
    plt.xlabel('Risk Score Quintile')
    plt.ylabel('Actual Burglary Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Execute the visualizations
# Uncomment these lines to run all visualizations

# 1. Plot performance metrics
plot_metrics(y_test, final_preds)

# 2. Error analysis
error_df = plot_error_analysis(y_test, final_preds)

# 3. Permutation importance
perm_importance = plot_permutation_importance(best_model, X_test, y_test)

# 4. SHAP analysis for model interpretability
try:
    plot_shap_analysis(best_model, X_test)
except Exception as e:
    print(f"SHAP analysis error: {e}")
    print("Install SHAP library if not available: pip install shap")

# 5. Learning curves
plot_learning_curve(best_model, X_train, y_train, X_test, y_test)

# 6. Geographical analysis - uncomment if geo data is available
# import geopandas as gpd
# geo_df = gpd.read_file("path_to_lsoa_shapefile.shp")
# plot_geographical_errors(X_test_copy, geo_df)

# 7. XGBoost tree visualization
plot_xgb_trees(best_model, num_trees=2)

# 8. Risk-Demand relationship
plot_risk_demand_relationship(X_test_copy)


# 9. Prediction Confidence Intervals using bootstrapping
def plot_prediction_confidence(model, X_test, y_test, n_bootstrap=100):
    bootstrap_preds = []
    indices = np.arange(len(X_train))

    for _ in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_idx = np.random.choice(indices, size=len(indices), replace=True)
        bootstrap_X = X_train.iloc[bootstrap_idx]
        bootstrap_y = y_train.iloc[bootstrap_idx]

        # Train model on bootstrap sample
        bootstrap_model = XGBRegressor(**study.best_params, random_state=42)
        bootstrap_model.fit(bootstrap_X, bootstrap_y)

        # Predict on test set
        bootstrap_preds.append(bootstrap_model.predict(X_test))

    # Convert to array
    bootstrap_preds = np.array(bootstrap_preds)

    # Calculate confidence intervals
    lower_bound = np.percentile(bootstrap_preds, 5, axis=0)
    upper_bound = np.percentile(bootstrap_preds, 95, axis=0)
    mean_pred = np.mean(bootstrap_preds, axis=0)

    # Plot with confidence intervals
    plt.figure(figsize=(12, 6))

    # Sort by actual values for better visualization
    sort_idx = np.argsort(y_test.values)
    actual_sorted = y_test.values[sort_idx]
    mean_sorted = mean_pred[sort_idx]
    lower_sorted = lower_bound[sort_idx]
    upper_sorted = upper_bound[sort_idx]

    plt.fill_between(range(len(actual_sorted)), lower_sorted, upper_sorted,
                     alpha=0.3, label='90% Confidence Interval')
    plt.plot(actual_sorted, 'o-', label='Actual', alpha=0.7)
    plt.plot(mean_sorted, 'o-', label='Mean Prediction', alpha=0.7)

    plt.title('Predictions with Confidence Intervals (Bootstrapped)', fontsize=15)
    plt.xlabel('Sorted Sample Index')
    plt.ylabel('Burglary Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()