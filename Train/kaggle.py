import os
import re

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold  # 添加 KFold 导入
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18
from utils import ensure_dir, sanitize_filename, plot_results, plot_loss_curve, shap_analysis, lime_analysis

# 设置中文字体并添加备用字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STFangsong', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_data(file_path, target_columns):
    try:
        data = pd.read_excel(file_path)
        print(f"Data loaded successfully from {file_path}!")
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        exit()

    data.columns = data.columns.map(str)
    band_columns = [col for col in data.columns if col.isdigit() and 350 <= int(col) <= 2500]

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


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=False),  # 修改 inplace 参数
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ECABlock(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16, k_size=3):
        super(CBAMBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=False),  # 修改 inplace 参数
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv = nn.Conv1d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1)
        channel_out = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_out.expand_as(x)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.conv(spatial_out)
        spatial_out = self.sigmoid_spatial(spatial_out)
        return x * spatial_out.expand_as(x)

class DCNN(nn.Module):
    def __init__(self, input_dim, attention_type=None):
        super(DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1, dilation=1)
        self.conv3_dilated = nn.Conv1d(128, 256, kernel_size=3, padding=2, dilation=2)  # Added dilated convolution
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(1024)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.attention_type = attention_type
        if attention_type == 'SE':
            self.attention1 = SEBlock(64)
            self.attention2 = SEBlock(128)
            self.attention3 = SEBlock(256)
            self.attention4 = SEBlock(512)
            self.attention5 = SEBlock(1024)
        elif attention_type == 'ECA':
            self.attention1 = ECABlock(64)
            self.attention2 = ECABlock(128)
            self.attention3 = ECABlock(256)
            self.attention4 = ECABlock(512)
            self.attention5 = ECABlock(1024)
        elif attention_type == 'CBAM':
            self.attention1 = CBAMBlock(64)
            self.attention2 = CBAMBlock(128)
            self.attention3 = CBAMBlock(256)
            self.attention4 = CBAMBlock(512)
            self.attention5 = CBAMBlock(1024)
        else:
            self.attention1 = self.attention2 = self.attention3 = self.attention4 = self.attention5 = None

        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 1)
        self.relu = nn.ReLU(inplace=False)
        self.leaky_relu = nn.LeakyReLU(inplace=False)
        self.elu = nn.ELU(inplace=False)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        if self.attention1:
            x = self.attention1(x)
        x = self.pool(x)
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        if self.attention2:
            x = self.attention2(x)
        x = self.pool(x)
        # Multi-scale feature fusion
        out1 = self.leaky_relu(self.bn3(self.conv3(x)))
        out2 = self.leaky_relu(self.bn3(self.conv3_dilated(x)))
        x = torch.cat([out1, out2], dim=1)  # Fuse features
        if self.attention3:
            x = self.attention3(x)
        x = self.pool(x)
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        if self.attention4:
            x = self.attention4(x)
        x = self.pool(x)
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        if self.attention5:
            x = self.attention5(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.elu(self.fc4(x))
        x = self.fc5(x)
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(weights=None)  # 修改为使用 weights 参数
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加这一行以确保输入有正确的通道数
        x = self.resnet(x)
        return x



class VGG7(nn.Module):
    def __init__(self, input_dim):
        super(VGG7, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * (input_dim // 8), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_model(X, y, input_dim, model_type='DCNN', attention_type=None, epochs=50, batch_size=32,
                learning_rate=0.0001, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_model = None
    best_val_loss = float('inf')

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        if model_type == 'ResNet18':
            model = ResNet18().to(device)
        elif model_type == 'VGG7':
            model = VGG7(input_dim).to(device)
        else:
            model = DCNN(input_dim, attention_type).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2正则化
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

    if plot and 0.85 <= r2 < 0.99:
        plot_results(y, y_pred, title)
        shap_analysis(model, X, feature_columns, target_column, model_type, attention_type, dataset_name)
        lime_analysis(model, X, y, feature_columns, target_column, model_type, attention_type, dataset_name)

    return r2, rmse, rpd

def main():
    file_paths = [
        ("../datasets/data_spectral_bands_sgd_dr.xlsx", "SBSD"),
        ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
        ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNSBE"),
        ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
        ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD")
    ]
    target_columns = ["EOC", "SOC", "WOC", "TC", "OM"]
    attention_types = [None, 'SE', 'ECA', 'CBAM']
    model_types = ['ResNet18', 'VGG7','DCNN']
    results = []

    for file_path, dataset_name in file_paths:
        X, y_dict, feature_columns, band_columns = load_data(file_path, target_columns)
        X = preprocess_data(X)

        for target_column, y in y_dict.items():
            print(f"Processing {target_column} from {dataset_name}")
            for model_type in model_types:
                for attention_type in attention_types :
                    print(f"Training {model_type} with attention type: {attention_type}")
                    model = train_model(X, y, input_dim=X.shape[1], model_type=model_type,
                                        attention_type=attention_type)
                    # 在验证集上评估模型
                    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=215, random_state=42)
                    test_metrics = evaluate_model(
                        model, X_val, y_val, feature_columns, target_column,
                        model_type, attention_type, dataset_name,
                        title=f"{dataset_name} - {target_column} - {attention_type} - {model_type}", plot=True
                    )
                    train_metrics = evaluate_model(model, X_train, y_train, feature_columns, target_column, model_type,
                                                   attention_type, dataset_name,
                                                   title=f"{dataset_name} - {target_column} - Train - {attention_type} - {model_type}",
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
    results_df.to_excel('./output/results_summary.xlsx', index=False)

if __name__ == "__main__":
    main()