import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
from tabulate import tabulate
from torchvision.models import resnet18
from torch.nn.utils.parametrizations import weight_norm
from sklearn.decomposition import PCA
import shap
import lime
import lime.lime_tabular

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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


class CNNWithAttention(nn.Module):
    def __init__(self, input_dim, attention_type=None):
        super(CNNWithAttention, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.conv6 = nn.Conv1d(512, 1024, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(1024)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.attention_type = attention_type
        if (attention_type == 'SE'):
            self.attention1 = SEBlock(32)
            self.attention2 = SEBlock(64)
            self.attention3 = SEBlock(128)
            self.attention4 = SEBlock(256)
            self.attention5 = SEBlock(512)
            self.attention6 = SEBlock(1024)  # 新增注意力层
        elif (attention_type == 'ECA'):
            self.attention1 = ECABlock(32)
            self.attention2 = ECABlock(64)
            self.attention3 = ECABlock(128)
            self.attention4 = ECABlock(256)
            self.attention5 = ECABlock(512)
            self.attention6 = ECABlock(1024)  # 新增注意力层
        elif (attention_type == 'CBAM'):
            self.attention1 = CBAMBlock(32)
            self.attention2 = CBAMBlock(64)
            self.attention3 = CBAMBlock(128)
            self.attention4 = CBAMBlock(256)
            self.attention5 = CBAMBlock(512)
            self.attention6 = CBAMBlock(1024)  # 新增注意力层
        else:
            self.attention1 = self.attention2 = self.attention3 = self.attention4 = self.attention5 = self.attention6 = None

        self.fc1 = nn.Linear(1024, 1024)  # 更新全连接层
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)  # 新增全连接层
        self.relu = nn.ReLU(inplace=False)          # 修改 inplace 参数
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
        x = self.leaky_relu(self.bn3(self.conv3(x)))
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
        x = self.pool(x)
        x = self.leaky_relu(self.bn6(self.conv6(x)))  # 新增前向传播
        if self.attention6:
            x = self.attention6(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.elu(self.fc4(x))
        x = self.fc5(x)  # 新增前向传播
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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU(inplace=False)  # 修改 inplace 参数
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU(inplace=False)  # 修改 inplace 参数
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU(inplace=False)   # 修改 inplace 参数
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(1, num_channels, kernel_size, dropout)  # 修改输入通道数为1
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        y1 = self.tcn(x)
        o = self.linear(y1[:, :, -1])
        return o


def train_model(X_train, y_train, X_val, y_val, input_dim, model_type='CNN', attention_type=None, epochs=100, batch_size=32, learning_rate=0.0001):
    if model_type == 'ResNet18':
        model = ResNet18().to(device)
    elif model_type == 'TCN':
        model = TCN(input_dim, [64, 128, 256, 512]).to(device)
    else:
        model = CNNWithAttention(input_dim, attention_type).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
                                torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
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
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    return model


def plot_results(y_true, y_pred, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, label='Predicted vs Actual', alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='1:1 Line')
    
    # 拟合线
    m, b = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, m * y_true + b, 'b-', label=f'Fit Line: y={m:.2f}x+{b:.2f}')
    
    # 计算 R², RMSE, RPD
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))   # 转换为 g·kg⁻¹
    rpd = np.std(y_true) / rmse
    
    # 左上角显示 R², RMSE, RPD
    plt.text(0.05, 0.95, f'R²: {r2:.4f}\nRMSE: {rmse:.4f} g·kg⁻¹\nRPD: {rpd:.4f}', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def shap_analysis(model, X, feature_names, target_column, model_type, attention_type, dataset_name):
    model.eval()
    explainer = shap.GradientExplainer(model, torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device))
    shap_values = explainer.shap_values(torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device))
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim > 2:
        shap_values = shap_values.squeeze()
    # 将 X 转换为带有特征名称的 DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    plt.figure()
    plt.title(f"SHAP - {target_column} ({attention_type}-{model_type}) - {dataset_name}")
    shap.summary_plot(shap_values, X_df)
    plt.show()
    plt.close()

def lime_analysis(model, X, y, feature_names, target_column, model_type, attention_type, dataset_name):
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)
    # 使用主成分名称作为特征名称
    pca_feature_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_pca, mode='regression', feature_names=pca_feature_names, verbose=True
    )
    i = np.random.randint(0, X_pca.shape[0])
    model.eval()
    with torch.no_grad():
        # 定义预测函数，先逆变换回原始空间，再输入模型
        def predict_fn(x):
            x_original = pca.inverse_transform(x)
            x_tensor = torch.tensor(x_original, dtype=torch.float32).unsqueeze(1).to(device)
            return model(x_tensor).cpu().numpy().flatten()
        exp = explainer.explain_instance(
            X_pca[i], predict_fn, num_features=10
        )
    # 生成并显示散点图
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME - {target_column} ({attention_type}-{model_type}) - {dataset_name}")
    plt.show()

def evaluate_model(model, X, y, feature_columns, target_column, model_type, attention_type, dataset_name, title="模型评估"):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        y_pred = model(X_tensor).squeeze().cpu().numpy()
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))   # 转换为 g·kg⁻¹
    rpd =np.std(y) / rmse
    
    # 当 R² 大于 0.7 时，生成散点图
    if 0.9 < r2 <= 0.99 :
        plot_results(y, y_pred, title)
        # 添加 SHAP 和 LIME 分析
        shap_analysis(model, X, feature_columns, target_column, model_type, attention_type, dataset_name)
        # lime_analysis(model, X, y, feature_columns, target_column, model_type, model_name, dataset_name)
    
    return r2, rmse, rpd

def main():
    file_paths = [
        ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
        ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNSBE"),
        ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
        ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD")
    ]
    target_columns = ["易氧化有机碳(mg/g)", "有机碳含量(g/kg)", "水溶性有机碳(mg/g)"]
    results = []

    for file_path, dataset_name in file_paths:
        X, y_dict, feature_columns, band_columns = load_data(file_path, target_columns)
        X = preprocess_data(X)

        for target_column, y in y_dict.items():
            print(f"Processing {target_column} from {dataset_name}")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            for model_type in ['CNN', 'ResNet18', 'TCN']:
                for attention_type in ["Origin", 'SE', 'ECA', 'CBAM']:
                    print(f"Training {model_type} with attention type: {attention_type}")
                    model = train_model(X_train, y_train, X_val, y_val, input_dim=X.shape[1], model_type=model_type, attention_type=attention_type)
                    test_metrics = evaluate_model(model, X_val, y_val, feature_columns, target_column, model_type, attention_type, dataset_name, title=f"{dataset_name} - {target_column} - Test - {attention_type} - {model_type}")
                    results.append((dataset_name, target_column, f"{model_type}_{attention_type}", test_metrics))
    
    headers = ["Dataset", "Target", "Model", "Test R²", "Test RMSE", "Test RPD"]
    table = [[dataset_name, target_column, model_name, f"{test_metrics[0]:.4f}", f"{test_metrics[1]:.4f}", f"{test_metrics[2]:.4f}"]
             for dataset_name, target_column, model_name, test_metrics in results]
                    # train_metrics = evaluate_model(model, X_train, y_train, title=f"{dataset_name} - {target_column} - Train - {attention_type}")
                    # results.append((dataset_name, target_column, f"{model_type}_{attention_type}", train_metrics, test_metrics))
    # headers = ["Dataset", "Target", "Model", "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD"]
    # table = [[dataset_name, target_column, model_name, f"{train_metrics[0]:.4f}", f"{train_metrics[1]:.4f}",
    #           f"{train_metrics[2]::.4f}", f"{test_metrics[0]:.4f}", f"{test_metrics[1]:.4f}", f"{test_metrics[2]:.4f}"]
    #          for dataset_name, target_column, model_name, train_metrics, test_metrics in results]

    print("\nResults Summary:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()