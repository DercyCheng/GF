import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from utils import load_data, preprocess_data, set_seed, ensure_dir, sanitize_filename
import torch.nn as nn
from utils import plot_results, shap_analysis, lime_analysis  # Add these imports

class SABlock(nn.Module):
    def __init__(self, channels):
        super(SABlock, self).__init__()
        # Define layers for scaled attention
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights


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
    def __init__(self, channel, reduction=16, k_size=7):
        super(CBAMBlock, self).__init__()
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv1d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        avg_out = self.channel_attention(x)
        channel_out = x * avg_out
        # 空间注意力
        avg_out = torch.mean(channel_out, dim=1, keepdim=True)
        max_out, _ = torch.max(channel_out, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.spatial_attention(spatial_out)
        return channel_out * spatial_out

class DCNN(nn.Module):
    def __init__(self, input_dim, attention_type=None):
        super(DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)  # Revert to 64 filters
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)  # Revert to 128 filters
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)  # Revert to 256 filters
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)  # Revert to 512 filters
        self.bn4 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Adjust dropout rate
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        attention_blocks = {
            'SE': SEBlock,
            'ECA': ECABlock,
            'CBAM': CBAMBlock,
            'SA': SABlock
        }
        self.attentions = nn.ModuleList([
            attention_blocks[attention_type](64) if attention_type else None,
            attention_blocks[attention_type](128) if attention_type else None,
            attention_blocks[attention_type](256) if attention_type else None,
            attention_blocks[attention_type](512) if attention_type else None
        ])
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),  # Change activation to ReLU
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),  # Change activation to ReLU
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),  # Change activation to ReLU
            nn.Linear(64, 1)
        )
        self.relu = nn.ReLU(inplace=True)  # Change activation to ReLU

    def forward(self, x):
        for i in range(4):
            x = self.relu(getattr(self, f'bn{i+1}')(getattr(self, f'conv{i+1}')(x)))  # Use ReLU
            if self.attentions[i]:
                x = self.attentions[i](x)
            x = self.pool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

file_paths = [
    ("../datasets/data_spectral_bands_sgd_dr.xlsx", "SBSD"),
    ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNSBE"),
    ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD")
]
target_columns = ["SOC", "EOC", "WOC", "TC", "OM"]
model_names = ["DCNN", "SE_DCNN", "ECA_DCNN", "CBAM_DCNN", "SA_DCNN"]  # Update model names

results = []

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, target_column):
    criterion = torch.nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Change optimizer to RMSprop with weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Adjust learning rate scheduler
    num_epochs = 500  # Adjust number of epochs
    early_stopping_patience = 50  # Adjust early stopping patience

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

    best_test_r2 = float('-inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test).cpu().numpy()
            test_r2 = r2_score(y_test.cpu().numpy(), y_test_pred)

        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train).cpu().numpy()
        y_test_pred = model(X_test).cpu().numpy()

    train_r2 = round(r2_score(y_train.cpu().numpy(), y_train_pred), 4)
    train_rmse = round(np.sqrt(mean_squared_error(y_train.cpu().numpy(), y_train_pred)), 4)
    train_rpd = round(np.std(y_train.cpu().numpy()) / train_rmse, 4)

    test_r2 = round(r2_score(y_test.cpu().numpy(), y_test_pred), 4)
    test_rmse = round(np.sqrt(mean_squared_error(y_test.cpu().numpy(), y_test_pred)), 4)
    test_rpd = round(np.std(y_test.cpu().numpy()) / test_rmse, 4)

    print(f"Train R²: {train_r2}, Test R²: {test_r2}")

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

    if 0.85 <= test_r2 < 0.99:
        plot_results(y_test.cpu().numpy(), y_test_pred, f"{dataset_name} - {target_column}", model_name, target_column)
        shap_analysis(model, X_test.cpu().numpy(), feature_names, target_column, model_name, None, dataset_name)
        lime_analysis(model, X_test.cpu().numpy(), y_test.cpu().numpy(), feature_names, target_column, model_name, None, dataset_name)

# 设置随机种子
set_seed()

for file_path, dataset_name in file_paths:
    X, y_dict, feature_names, band_columns = load_data(file_path, target_columns)
    X = preprocess_data(X)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    
    for target_column in target_columns:
        y = y_dict[target_column]
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        input_dim = X_train.shape[2]
        for model_name in model_names:
            attention_type = model_name.split('_')[0] if '_' in model_name else None
            model = DCNN(input_dim, attention_type).to('cuda' if torch.cuda.is_available() else 'cpu')
            train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, target_column)

# 保存结果到 result_summary.xlsx
results_df = pd.DataFrame(results)
results_df.to_excel("result_summary.xlsx", index=False, header=["Dataset", "Target", "Model", "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD"])