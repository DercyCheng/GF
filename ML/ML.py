import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import shap
from tabulate import tabulate
import xgboost as xgb
from sklearn.linear_model import Ridge, Lasso  # 添加导入
import lime
import lime.lime_tabular  # 添加导入
import csv

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

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
    X_scaled = scaler.fit_transform(X)
    pt = PowerTransformer(method='yeo-johnson')
    X_transformed = pt.fit_transform(X_scaled)
    return X_transformed

def perform_pca(X, n_components=10):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"PCA completed, retained {n_components} components")
    return X_pca

def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rpd = np.std(y_true) / rmse
    return r2, rmse, rpd

def train_model(X, y, model_name):
    if model_name == "Random Forest":
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        model = RandomForestRegressor(random_state=42)
    elif model_name == "Gradient Boosting":
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        model = GradientBoostingRegressor(random_state=42)
    elif model_name == "XGBoost":
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        }
        model = xgb.XGBRegressor(random_state=42, tree_method='gpu_hist', gpu_id=0)
    elif model_name == "PLSR":
        param_grid = {'n_components': [2, 5, 10, 15, 20, 25, 30]}
        model = PLSRegression()
    elif model_name == "SVM":
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'epsilon': [0.01, 0.1, 0.2, 0.3],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }
        model = SVR()
    elif model_name == "Ridge":
        param_grid = {'alpha': [0.1, 1, 10, 100, 1000]}
        model = Ridge()
    elif model_name == "Lasso":
        param_grid = {'alpha': [0.1, 1, 10, 100, 1000]}
        model = Lasso()

    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='r2', n_jobs=-1)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scores_train, rmse_scores_train, rpd_scores_train = [], [], []
    r2_scores_test, rmse_scores_test, rpd_scores_test = [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        best_model.fit(X_train, y_train)
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        r2_train, rmse_train, rpd_train = evaluate_model(y_train, y_pred_train)
        r2_test, rmse_test, rpd_test = evaluate_model(y_test, y_pred_test)

        r2_scores_train.append(r2_train)
        rmse_scores_train.append(rmse_train)
        rpd_scores_train.append(rpd_train)
        r2_scores_test.append(r2_test)
        rmse_scores_test.append(rmse_test)
        rpd_scores_test.append(rpd_test)

    print(f"Cross-validated results for {model_name}:")
    print(f"Train - Mean R²: {np.mean(r2_scores_train)}, Mean RMSE: {np.mean(rmse_scores_train)}, Mean RPD: {np.mean(rpd_scores_train)}")
    print(f"Test - Mean R²: {np.mean(r2_scores_test)}, Mean RMSE: {np.mean(rmse_scores_test)}, Mean RPD: {np.mean(rpd_scores_test)}")

    best_model.fit(X, y)
    return best_model, (np.mean(r2_scores_train), np.mean(rmse_scores_train), np.mean(rpd_scores_train)), (np.mean(r2_scores_test), np.mean(rmse_scores_test), np.mean(rpd_scores_test))

def shap_analysis(model, X, feature_columns, target_column, dataset_name, model_name):
    explainer = shap.TreeExplainer(model) if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor, xgb.XGBRegressor)) else shap.KernelExplainer(model.predict, X)
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X)
    shap.initjs()
    plt.figure()
    plt.title(f"SHAP - {target_column} ({model_name}) - {dataset_name}")
    shap.summary_plot(shap_values, X, feature_names=feature_columns)
    plt.show()
    plt.close()

def lime_analysis(model, X, feature_columns, target_column, dataset_name, model_name):
    explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names=feature_columns, class_names=[target_column], verbose=True, mode='regression')
    i = np.random.randint(0, X.shape[0])
    exp = explainer.explain_instance(X[i], model.predict, num_features=10)
    print(f"Calculating LIME values for {target_column} ({model_name}) - {dataset_name}")
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME - {target_column} ({model_name}) - {dataset_name}")
    plt.show()
    # fig.savefig(f"lime_scatter_{dataset_name}_{target_column}_{model_name}.png")
    plt.close()

def main():
    file_paths = [
        ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
        ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNSBE"),
        ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
        ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD")
    ]
    target_columns = ["易氧化有机碳(mg/g)", "有机碳含量(g/kg)","水溶性有机碳(mg/g)","全碳(g/kg)","有机质(g/kg)"]
    model_names = ["Random Forest", "Gradient Boosting", "XGBoost", "PLSR", "SVM", "Ridge", "Lasso"]
    results = []

    for file_path, dataset_name in file_paths:
        X, y_dict, feature_columns, band_columns = load_data(file_path, target_columns)
        X = preprocess_data(X)

        for target_column, y in y_dict.items():
            for model_name in model_names:
                print(f"Processing {target_column} from {dataset_name} using {model_name}")
                X_pca = perform_pca(X)
                model, train_metrics, test_metrics = train_model(X_pca, y, model_name)
                # shap_analysis(model, X_pca, feature_columns, target_column, dataset_name, model_name)
                # lime_analysis(model, X_pca, feature_columns, target_column, dataset_name, model_name)  # 添加 LIME 分析
                results.append((dataset_name, target_column, model_name, train_metrics, test_metrics))

    headers = ["Dataset", "Target", "Model", "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD"]
    table = [[dataset_name, target_column, model_name, f"{train_metrics[0]:.4f}", f"{train_metrics[1]:.4f}", f"{train_metrics[2]:.4f}", f"{test_metrics[0]:.4f}", f"{test_metrics[1]:.4f}", f"{test_metrics[2]:.4f}"] for dataset_name, target_column, model_name, train_metrics, test_metrics in results]

    print("\nResults Summary:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

    # 将结果导出为 xlsx 文件
    results_df = pd.DataFrame(table, columns=headers)
    results_df.to_excel('results_summary.xlsx', index=False)
    
if __name__ == "__main__":
    main()