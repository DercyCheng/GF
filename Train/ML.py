import numpy as np
import pandas as pd
import os
import re
import random
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
from optuna import Trial, create_study
from optuna.samplers import TPESampler


file_paths = [
    ("../datasets/data_spectral_bands_sgd_dr.xlsx", "SBSD"),
    ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNSBE"),
    ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD"),
    ("../datasets/data_soil_nutrients_spectral_bands_sae.xlsx", "SNSBSAE"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment_sae.xlsx", "SNSBESAE"),
    ("../datasets/data_spectral_bands_sae.xlsx", "SBSAE")
]
target_columns = ["SOC", "EOC", "WOC", "TC", "OM"]
model_names = ["Random Forest", "XGBoost", "PLSR", "SVM", "LightGBM", "ElasticNet", "AdaBoost"]

results = []

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def sanitize_filename(filename):
    # 去除文件名中的非法字符
    return re.sub(r'[\/:*?"<>|]', '-', filename)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def augment_data(X, y):
    # Example data augmentation: adding Gaussian noise
    noise = np.random.normal(0, 0.01, X.shape)
    X_augmented = X + noise
    return np.concatenate([X, X_augmented]), np.concatenate([y, y])

def load_data(file_path, target_columns):
    try:
        data = pd.read_excel(file_path)
        print(f"Data loaded successfully from {file_path}!")
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        exit()

    data.columns = data.columns.map(str)
    band_columns = [col for col in data.columns if col.isdigit() and 400 <= int(col) <= 2400]

    if not band_columns:
        print("No band columns found! Please check the column names format.")
        exit()

    feature_columns = [col for col in data.columns if col not in target_columns]
    data = data.dropna(subset=target_columns + band_columns)
    X = data[feature_columns].select_dtypes(include=[np.number]).values
    y_dict = {target_column: data[target_column].values for target_column in target_columns}

    print(f"Number of features: {len(feature_columns)}, Number of samples: {X.shape[0]}")
    return X, y_dict, feature_columns, band_columns

def preprocess_data(X):
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    return scaler.fit_transform(X)


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

def optimize_hyperparameters(model_name, X, y):
    def objective(trial: Trial):
        if model_name == "Random Forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "random_state": 42
            }
            model = RandomForestRegressor(**params)
        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "verbosity": 0,
                "random_state": 42
            }
            model = XGBRegressor(**params)
        elif model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "random_state": 42
            }
            model = LGBMRegressor(**params)
        elif model_name == "ElasticNet":
            params = {
                "alpha": trial.suggest_loguniform("alpha", 1e-4, 1e1),
                "l1_ratio": trial.suggest_uniform("l1_ratio", 0.0, 1.0),
                "random_state": 42
            }
            model = ElasticNet(**params)
        elif model_name == "AdaBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
                "random_state": 42
            }
            model = AdaBoostRegressor(**params)
        else:
            return 0  # For models without tuning

        score = cross_val_score(model, X, y, cv=3, scoring='r2').mean()
        return score

    study = create_study(direction="maximize", sampler=TPESampler())
    study.optimize(objective, n_trials=50)
    return study.best_params

# 设置随机种子
set_seed(42)

for file_path, dataset_name in file_paths:
    X, y_dict, feature_names, band_columns = load_data(file_path, target_columns)
    X = preprocess_data(X)
    
    for target_column in target_columns:
        y = y_dict[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=2, random_state=42, verbosity=0),
            "PLSR": PLSRegression(n_components=3),
            "SVM": SVR(C=0.5, kernel='linear', epsilon=0.1),
            "LightGBM": LGBMRegressor(n_estimators=50, learning_rate=0.05, max_depth=2, random_state=42),
            "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
            "AdaBoost": AdaBoostRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
        }
        
        for model_name, model in models.items():
            best_params = optimize_hyperparameters(model_name, X_train, y_train)
            model.set_params(**best_params)
            train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, target_column)

# 保存结果到 result_summary.xlsx
results_df = pd.DataFrame(results)
results_df.to_excel("result_summary.xlsx", index=False, header=["Dataset", "Target", "Model", "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD"])