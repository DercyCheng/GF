import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from tabulate import tabulate
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

# Import models
from models.DCNN import DCNN
from models.ResNet18 import ResNet18
from models.VGG7 import VGG7
from models.SSLT import SSLT
from models.LSTM import LSTM
from models.TCN import TCN
from models.CNN_LSTM import CNN_LSTM

# Import utility functions
from utils import plot_results, shap_analysis, lime_analysis, set_seed, augment_data, load_data, preprocess_data, \
    sanitize_filename, plot_loss_curve, compute_regularization  # 添加 plot_loss_curve 和 compute_regularization

# 设置中文字体并添加备用字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STFangsong', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def initialize_model(model_type, input_dim, attention_type=None):
    model_classes = {
        'ResNet18': ResNet18,
        'VGG7': VGG7,
        'DCNN': DCNN,
        'SSLT': SSLT,
        'LSTM': LSTM,
        'TCN': TCN,
        'CNN_LSTM': CNN_LSTM
    }
    if model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_type}")
    if model_type == 'TCN':
        return model_classes[model_type](input_dim, output_size=1, num_channels=[25]*8, kernel_size=7, dropout=0.2)
    elif model_type in ['DCNN', 'SSLT']:
        return model_classes[model_type](input_dim, attention_type=attention_type)
    else:
        return model_classes[model_type](input_dim)

def prepare_dataset(X_train, y_train, X_val, y_val, model_type):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
                                torch.tensor(y_val, dtype=torch.float32))
    return train_dataset, val_dataset


from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, train_loader, optimizer, criterion, device, model_type, l1_coeff, l2_coeff):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        if model_type in ['SSLT', 'TCN']:
            X_batch = X_batch.permute(0, 2, 1).contiguous()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        reg_loss = compute_regularization(model, l1_coeff, l2_coeff)  # 添加正则化损失
        total_loss = loss + reg_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 降低梯度裁剪的 max_norm
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    return train_loss / len(train_loader.dataset)

def train_model(X, y, input_dim, model_type, attention_type, epochs, batch_size, learning_rate, n_splits, seed, patience, dataset_name, target_column, l1_coeff=0.0, l2_coeff=1e-4):
    set_seed(seed)  # 使用传入的随机种子
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_model = None
    best_val_loss = float('inf')

    all_train_losses = []  # 添加用于存储所有折叠训练损失的列表
    all_val_losses = []    # 添加用于存储所有折叠验证损失的列表

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Data Augmentation
        X_train, y_train = augment_data(X_train, y_train)

        model = initialize_model(model_type, input_dim, attention_type).to(device)
        model.device = device
        criterion = nn.MSELoss()

        train_dataset, val_dataset = prepare_dataset(X_train, y_train, X_val, y_val, model_type)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_coeff)  # 使用 AdamW
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  # 使用 CosineAnnealingLR

        patience_counter = 0

        train_losses = []  # 初始化训练损失列表
        val_losses = []    # 初始化验证损失列表

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, model_type, l1_coeff, l2_coeff)
            train_losses.append(train_loss)  # 记录训练损失

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    if model_type in ['SSLT', 'TCN']:
                        X_batch = X_batch.permute(0, 2, 1).contiguous()
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    reg_loss = compute_regularization(model, l1_coeff, l2_coeff)
                    total_loss = loss + reg_loss
                    val_loss += total_loss.item() * X_batch.size(0)

            val_loss /= len(val_loader)
            val_losses.append(val_loss)  # 记录验证损失
            scheduler.step()  # 更新学习率
            current_lr = scheduler.get_last_lr()[0]  # 获取当前学习率

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")  # 添加学习率信息

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break

        # 在每个折叠结束后，存储训练和验证损失
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

    # 所有折叠完成后，统一绘制损失曲线
    plot_loss_curve(all_train_losses, all_val_losses, target_column, model_type, dataset_name, attention_type)  # 修改为传递 dataset_name 和 attention_type

    return best_model

def evaluate_model(model, X, y, feature_columns, target_column, model_type, attention_type, dataset_name,
                   title="模型评估", plot=False):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        if model_type in ['SSLT', 'TCN']:
            X_tensor = X_tensor.squeeze(1)
        torch.tensor(y, dtype=torch.float32).to(device)
        y_pred = model(X_tensor).squeeze().cpu().numpy()
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    rpd = np.std(y) / rmse

    if plot and 0.85 <= r2 < 0.99:#
        plot_results(y, y_pred, title, model_type, sanitize_filename(target_column))
        shap_analysis(model, X, feature_columns, sanitize_filename(target_column), model_type, attention_type, dataset_name)
        lime_analysis(model, X, y, feature_columns, sanitize_filename(target_column), model_type, attention_type, dataset_name)

    return r2, rmse, rpd

def train_and_evaluate(X, y, input_dim, model_type, attention_type, epochs, batch_size, learning_rate, n_splits, seed, patience, feature_columns, target_column, dataset_name):
    model = train_model(
        X, y, input_dim=input_dim,
        model_type=model_type,
        attention_type=attention_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_splits=n_splits,
        seed=seed,
        patience=patience,  # 传入 patience
        dataset_name=dataset_name,  # 添加 dataset_name 参数
        target_column=target_column,  # 添加 target_column 参数
        l1_coeff=1e-5,  # 设置 L1 正则化系数
        l2_coeff=1e-4  # 设置 L2 正则化系数
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=215, random_state=seed)
    test_metrics = evaluate_model(
        model, X_val, y_val, feature_columns, target_column,
        model_type, attention_type, dataset_name,
        title=f"{dataset_name} - {target_column} - {attention_type} - {model_type}" if attention_type else f"{dataset_name} - {target_column} - {model_type}",
        plot=True
    )
    train_metrics = evaluate_model(
        model, X_train, y_train, feature_columns, target_column, model_type,
        attention_type, dataset_name,
        title=f"{dataset_name} - {target_column} - Train - {attention_type} - {model_type}" if attention_type else f"{dataset_name} - {target_column} - {model_type}",
        plot=False
    )
    return train_metrics, test_metrics

def process_dataset(file_path, dataset_name, target_columns, EPOCHS, BATCH_SIZE, LEARNING_RATE, N_SPLITS, SEED, model_types, attention_types, patience, results, l1_coeff, l2_coeff):
    X, y_dict, feature_columns = load_data(file_path, target_columns)
    X = preprocess_data(X)
    pca = PCA(n_components=30)  # 将 n_components 从 50 减少到 30
    
    X = pca.fit_transform(X)
    
    for target_column, y in y_dict.items():
        print(f"Processing {target_column} from {dataset_name}")
        for model_type in model_types:
            for attention_type in attention_types:
                print(f"Training {model_type} with attention type: {attention_type}")
                train_metrics, test_metrics = train_and_evaluate(
                    X, y, input_dim=X.shape[1],
                    model_type=model_type,
                    attention_type=attention_type,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    learning_rate=LEARNING_RATE,
                    n_splits=N_SPLITS,
                    seed=SEED,
                    patience=patience,  # 传入 patience
                    feature_columns=feature_columns,
                    target_column=target_column,  # 传递 target_column
                    dataset_name=dataset_name
                )
                results.append(
                    (dataset_name, target_column, model_type, attention_type, train_metrics, test_metrics)
                )

def main():
    file_paths = [
        ("../datasets/data_spectral_bands_sgd_dr.xlsx", "SBSD"),
        ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
        ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNSBE"),
        ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
        ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD")
    ]
    target_columns = ["EOC", "SOC", "WOC", "TC", "OM"]
    model_types = ['DCNN'] #'TCN', 'SSLT','ResNet18', 'VGG7','DCNN','LSTM', 'CNN_LSTM'
    attention_types = [ None, 'SE','ECA', 'CBAM', 'SA']#
    results = []

    SEED = 42
    EPOCHS = 300
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    N_SPLITS = 5
    PATIENCE = 20  # 定义 patience
    L1_COEFF = 1e-4  # 将 L1 正则化系数从 1e-5 增加到 1e-4
    L2_COEFF = 1e-3  # 将 L2 正则化系数从 1e-4 增加到 1e-3

    for file_path, dataset_name in file_paths:
        process_dataset(
            file_path, dataset_name, target_columns,
            EPOCHS, BATCH_SIZE, LEARNING_RATE, N_SPLITS, SEED,
            model_types, attention_types, PATIENCE, results,
            l1_coeff=L1_COEFF, l2_coeff=L2_COEFF  # 传入正则化系数
        )

    headers = ["Dataset", "Target", "Model", "Attention", "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD"]
    table = [
        [dataset_name, target_column, model_type, attention_type, 
         f"{train_metrics[0]:.4f}", f"{train_metrics[1]:.4f}", f"{train_metrics[2]:.4f}", 
         f"{test_metrics[0]:.4f}", f"{test_metrics[1]:.4f}", f"{test_metrics[2]::.4f}"]
        for dataset_name, target_column, model_type, attention_type, train_metrics, test_metrics in results
    ]

    print("\nResults Summary:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

    results_df = pd.DataFrame(table, columns=headers)
    results_df.to_excel(f'./output/results_summary.xlsx', index=False)

if __name__ == "__main__":
    main()
