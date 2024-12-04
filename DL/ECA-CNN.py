import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import torch
from torch import device
from torch.utils.data import DataLoader, TensorDataset
import shap
from tabulate import tabulate

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ECANet(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECANet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.eca1 = ECANet(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.eca2 = ECANet(32)
        self.fc1 = nn.Linear(32 * input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.eca1(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.eca2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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

def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rpd = np.std(y_true) / rmse
    return r2, rmse, rpd

def train_model(X, y, input_dim):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CNN(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores_train, rmse_scores_train, rpd_scores_train = [], [], []
    r2_scores_test, rmse_scores_test, rpd_scores_test = [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        for epoch in range(50):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred_train = model(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)).cpu().numpy().squeeze()
            y_pred_test = model(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)).cpu().numpy().squeeze()

        r2_train, rmse_train, rpd_train = evaluate_model(y_train, y_pred_train)
        r2_test, rmse_test, rpd_test = evaluate_model(y_test, y_pred_test)

        r2_scores_train.append(r2_train)
        rmse_scores_train.append(rmse_train)
        rpd_scores_train.append(rpd_train)
        r2_scores_test.append(r2_test)
        rmse_scores_test.append(rmse_test)
        rpd_scores_test.append(rpd_test)

    print(f"Cross-validated results for CNN:")
    print(f"Train - Mean R²: {np.mean(r2_scores_train)}, Mean RMSE: {np.mean(rmse_scores_train)}, Mean RPD: {np.mean(rpd_scores_train)}")
    print(f"Test - Mean R²: {np.mean(r2_scores_test)}, Mean RMSE: {np.mean(rmse_scores_test)}, Mean RPD: {np.mean(rpd_scores_test)}")

    model.train()
    train_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for epoch in range(50):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    return model, (np.mean(r2_scores_train), np.mean(rmse_scores_train), np.mean(rpd_scores_train)), (np.mean(r2_scores_test), np.mean(rmse_scores_test), np.mean(rpd_scores_test))

def shap_analysis(model, X, feature_columns, target_column, dataset_name, model_name):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    explainer = shap.DeepExplainer(model, torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device))
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device))
    shap.initjs()
    plt.figure()
    plt.title(f"SHAP - {target_column} ({model_name}) - {dataset_name}")
    shap.summary_plot(shap_values, X, feature_names=feature_columns)
    plt.show()
    plt.close()

def main():
    file_paths = [
        ("./datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
        ("./datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNDBE"),
        ("./datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
        ("./datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD")
    ]
    target_columns = ["易氧化有机碳(mg/g)", "有机碳含量(g/kg)", "水溶性有机碳(mg/g)"]
    results = []

    for file_path, dataset_name in file_paths:
        X, y_dict, feature_columns, band_columns = load_data(file_path, target_columns)
        X = preprocess_data(X)

        for target_column, y in y_dict.items():
            print(f"Processing {target_column} from {dataset_name} using CNN")
            model, train_metrics, test_metrics = train_model(X, y, X.shape[1])
            shap_analysis(model, X, feature_columns, target_column, dataset_name, "CNN")
            results.append((dataset_name, target_column, "CNN", train_metrics, test_metrics))

    headers = ["Dataset", "Target", "Model", "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD"]
    table = [[dataset_name, target_column, model_name, f"{train_metrics[0]:.4f}", f"{train_metrics[1]:.4f}", f"{train_metrics[2]:.4f}", f"{test_metrics[0]:.4f}", f"{test_metrics[1]:.4f}", f"{test_metrics[2]:.4f}"] for dataset_name, target_column, model_name, train_metrics, test_metrics in results]

    print("\nResults Summary:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
    def plot_results(y_true, y_pred, target_column, dataset_name, model_name, metrics):
        r2, rmse, rpd = metrics
        plt.figure(figsize=(20, 15))
        plt.scatter(y_true, y_pred, label='Predicted vs Actual', alpha=0.6)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='1:1 Line')
        plt.plot(np.unique(y_true), np.poly1d(np.polyfit(y_true, y_pred, 1))(np.unique(y_true)), label='Fit Line')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{target_column} - {dataset_name} ({model_name})')
        plt.legend()
        plt.text(0.05, 0.95, f'R²: {r2:.4f}\nRMSE: {rmse:.4f}\nRPD: {rpd:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.grid(True)
        plt.show()

    def main():
        file_paths = [
            ("./datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
            ("./datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNDBE"),
            ("./datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
            ("./datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD")
        ]
        target_columns = ["易氧化有机碳(mg/g)", "有机碳含量(g/kg)", "水溶性有机碳(mg/g)"]
        results = []

        for file_path, dataset_name in file_paths:
            X, y_dict, feature_columns, band_columns = load_data(file_path, target_columns)
            X = preprocess_data(X)

            for target_column, y in y_dict.items():
                print(f"Processing {target_column} from {dataset_name} using CNN")
                model, train_metrics, test_metrics = train_model(X, y, X.shape[1])
                shap_analysis(model, X, feature_columns, target_column, dataset_name, "CNN")
                results.append((dataset_name, target_column, "CNN", train_metrics, test_metrics))

                # Plot results
                y_pred = model(torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)).cpu().detach().numpy().squeeze()
                plot_results(y, y_pred, target_column, dataset_name, "CNN", test_metrics)

        headers = ["Dataset", "Target", "Model", "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD"]
        table = [[dataset_name, target_column, model_name, f"{train_metrics[0]:.4f}", f"{train_metrics[1]:.4f}", f"{train_metrics[2]:.4f}", f"{test_metrics[0]:.4f}", f"{test_metrics[1]:.4f}", f"{test_metrics[2]:.4f}"] for dataset_name, target_column, model_name, train_metrics, test_metrics in results]

        print("\nResults Summary:")
        print(tabulate(table, headers=headers, tablefmt="grid"))

    if __name__ == "__main__":
        main()