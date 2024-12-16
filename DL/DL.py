import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold  # 添加 KFold 导入
from tabulate import tabulate
from torch.utils.data import DataLoader, TensorDataset

# Import models
from models.CNN import CNN
from models.ResNet18 import ResNet18
from models.VGG7 import VGG7
from models.SSLT import SSLT  # Re-added SSLT model import
from models.LSTM import LSTM  # Added LSTM model import
from models.TCN import TCN  # Added TCN model import

# Import utility functions
from utils import plot_results, shap_analysis, lime_analysis, set_seed, augment_data, load_data, preprocess_data, \
    sanitize_filename

# 设置中文字体并添加备用字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STFangsong', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train_model(X, y, input_dim, model_type='CNN', attention_type=None, epochs=200, batch_size=32,
                learning_rate=0.0001, n_splits=5):
    set_seed(42)  # Ensure reproducibility
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_model = None
    best_val_loss = float('inf')

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Data Augmentation
        X_train, y_train = augment_data(X_train, y_train)

        if model_type == 'ResNet18':
            model = ResNet18().to(device)
            model.device = device
        elif model_type == 'VGG7':
            model = VGG7(input_dim).to(device)
            model.device = device
        elif model_type == 'CNN':
            model = CNN(input_dim, attention_type).to(device)
            model.device = device
        elif model_type == 'SSLT':  # Re-added SSLT model selection
            model = SSLT(input_dim, attention_type=attention_type).to(device)
            model.device = device
        elif model_type == 'LSTM':  # Added LSTM model selection
            model = LSTM(input_dim).to(device)
            model.device = device
        elif model_type == 'TCN':  # Added TCN model selection
            model = TCN(input_dim).to(device)
            model.device = device
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Increased weight_decay for stronger regularization
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
                                      torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
                                    torch.tensor(y_val, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break

    return best_model

def evaluate_model(model, X, y, feature_columns, target_column, model_type, attention_type, dataset_name,
                   title="模型评估", plot=False):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        y_pred = model(X_tensor).squeeze().cpu().numpy()
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))  # 转换为 g·kg⁻¹
    rpd = np.std(y) / rmse

    if plot and 0.85 < r2 <= 0.91:
        plot_results(y, y_pred, title, model_type, sanitize_filename(target_column))
        shap_analysis(model, X, feature_columns, sanitize_filename(target_column), model_type, attention_type, dataset_name)
        lime_analysis(model, X, y, feature_columns, sanitize_filename(target_column), model_type, attention_type, dataset_name)

    return r2, rmse, rpd

def main():
    file_paths = [
        ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
        ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNSBE"),
        ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
        ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD")
    ]
    target_columns = ["易氧化有机碳(mg/g)", "有机碳含量(g/kg)", "水溶性有机碳(mg/g)", "全碳(g/kg)", "有机质(g/kg)"]
    results = []

    for file_path, dataset_name in file_paths:
        X, y_dict, feature_columns, band_columns = load_data(file_path, target_columns)
        X = preprocess_data(X)

        for target_column, y in y_dict.items():
            print(f"Processing {target_column} from {dataset_name}")
            for model_type in ['ResNet18', 'VGG7', 'CNN', 'SSLT', 'LSTM', 'TCN']:  # Re-added SSLT to model list
                for attention_type in [None, 'SE', 'ECA', 'CBAM', 'SA']:
                    print(f"Training {model_type} with attention type: {attention_type}")
                    title_test = f"{dataset_name} - {target_column} - {attention_type} - {model_type}" if attention_type else f"{dataset_name} - {target_column} - {model_type}"
                    model = train_model(X, y, input_dim=X.shape[1], model_type=model_type,
                                        attention_type=attention_type)
                    # 在验证集上评估模型
                    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=215, random_state=42)
                    test_metrics = evaluate_model(
                        model, X_val, y_val, feature_columns, target_column,
                        model_type, attention_type, dataset_name,
                        title=title_test, plot=True
                    )
                    title_train = f"{dataset_name} - {target_column} - Train - {attention_type} - {model_type}" if attention_type else f"{dataset_name} - {target_column} - Train - {model_type}"
                    train_metrics = evaluate_model(model, X_train, y_train, feature_columns, target_column, model_type,
                                                   attention_type, dataset_name,
                                                   title=title_train,
                                                   plot=False)
                    results.append(
                        (dataset_name, target_column, f"{model_type}_{attention_type}", train_metrics, test_metrics))

    headers = ["Dataset", "Target", "Model", "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD"]
    table = [[dataset_name, target_column, model_name, f"{train_metrics[0]:.4f}", f"{train_metrics[1]:.4f}",
              f"{train_metrics[2]:.4f}", f"{test_metrics[0]:.4f}", f"{test_metrics[1]:.4f}", f"{test_metrics[2]:.4f}"]
             for dataset_name, target_column, model_name, train_metrics, test_metrics in results]

    print("\nResults Summary:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

    # 将结果导出为 xlsx 文件
    results_df = pd.DataFrame(table, columns=headers)
    results_df.to_excel(f'./output/results_summary.xlsx', index=False)

if __name__ == "__main__":
    main()