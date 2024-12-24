import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from utils import load_data, preprocess_data, set_seed

file_paths = [
    ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNSBE"),
    ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD")
]
target_columns = ["EOC", "SOC", "WOC", "TC", "OM"]
model_names = ["Random Forest", "Gradient Boosting", "XGBoost", "PLSR", "SVM"]

results = []

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, target_column):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = round(r2_score(y_train, y_train_pred), 4)
    train_rmse = round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4)
    train_rpd = round(np.std(y_train) / train_rmse, 4)
    
    test_r2 = round(r2_score(y_test, y_test_pred), 4)
    test_rmse = round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4)
    test_rpd = round(np.std(y_test) / test_rmse, 4)
    
    results.append({
        "Dataset": dataset_name,
        "Target": target_column,
        "Model": model_name,
        "Train R²": train_r2,
        "Train RMSE": train_rmse,
        "Train RPD": train_rpd,
        "Test R²": test_r2,
        "Test RMSE": test_rmse,
        "Test RPD": test_rpd
    })
    # plot_results(y_test, y_pred, f"{model_name} - {target_column}", model_name, target_column)
    # shap_analysis(model, X_test, feature_names, target_column, model_name, None, dataset_name)
    # lime_analysis(model, X_test, y_test, feature_names, target_column, model_name, None, dataset_name)

# 设置随机种子
set_seed()

for file_path, dataset_name in file_paths:
    X, y_dict, feature_names, band_columns = load_data(file_path, target_columns)
    X = preprocess_data(X)
    
    for target_column in target_columns:
        y = y_dict[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, verbosity=0),
            "PLSR": PLSRegression(n_components=5),
            "SVM": SVR(C=1.0, kernel='linear', epsilon=0.1)
        }
        
        for model_name, model in models.items():
            train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, target_column)

# 保存结果到 result_summary.xlsx
results_df = pd.DataFrame(results)
results_df.to_excel("result_summary.xlsx", index=False, header=["Dataset", "Target", "Model", "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD"])