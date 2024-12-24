import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from tabulate import tabulate
from torch.utils.data import DataLoader, TensorDataset
import os  # 添加导入

# Import models
from models.DCNN import DCNN
from models.ResNet18 import ResNet18
from models.VGG7 import VGG7
# from models.SSLT import SSLT  # 移除导入
from models.LSTM import LSTM
from models.TCN import TCN
from models.CNN_LSTM import CNN_LSTM

# Import utility functions
from utils import average_loss_curves, plot_results, shap_analysis, lime_analysis, set_seed, augment_data, load_data, preprocess_data, \
    sanitize_filename, reduce_dimensionality, plot_loss_curve  # 添加 plot_loss_curve
from utils import apply_regularization, increase_data, grid_search_regularization, advanced_augment_data, prune_model, SklearnWrapper  # 添加导入

# 设置中文字体并添加备用字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STFangsong', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def initialize_model(model_type, input_dim, attention_type=None):
    model_classes = {
        'ResNet18': ResNet18,
        'VGG7': VGG7,
        'DCNN': DCNN,
        'LSTM': LSTM,
        'TCN': TCN,
        'CNN_LSTM': CNN_LSTM
    }
    if model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_type}")
    if model_type == 'TCN':
        return model_classes[model_type](input_dim, output_size=1, num_channels=[25]*8, kernel_size=7, dropout=0.2)
    elif model_type in ['DCNN']:  # 移除 SSLT
        return model_classes[model_type](input_dim, attention_type=attention_type)
    else:
        return model_classes[model_type](input_dim)

def prepare_dataset(X_train, y_train, X_val, y_val, model_type):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
                                torch.tensor(y_val, dtype=torch.float32))
    return train_dataset, val_dataset

def train_one_epoch(model, train_loader, optimizer, criterion, device, model_type, reg_type='L2', reg_weight=1e-4):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        if model_type in ['TCN']:
            X_batch = X_batch.permute(0, 2, 1).contiguous()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        apply_regularization(model, reg_type, reg_weight)  # 应用正则化
        train_loss += loss.item() * X_batch.size(0)
    return train_loss / len(train_loader.dataset)

<<<<<<< HEAD
def train_model(X, y, input_dim, model_type, attention_type, epochs, batch_size, learning_rate, n_splits, seed, patience, target_column, dataset_name, reg_type='L2', reg_weight=1e-4):
    set_seed(seed)
=======
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_loss_curves(train_losses, val_losses, n_splits, target_name, model_name):
    # 取所有 fold 的最短训练长度，避免早停带来的长度不一致
    min_len = min(len(tl) for tl in train_losses)
    trimmed_train_losses = [tl[:min_len] for tl in train_losses]
    trimmed_val_losses = [vl[:min_len] for vl in val_losses]
    
    # 计算平均训练损失和平均验证损失
    avg_train_loss = np.mean(trimmed_train_losses, axis=0)
    avg_val_loss = np.mean(trimmed_val_losses, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(moving_average(avg_train_loss), label='Train Loss', alpha=0.7)
    plt.plot(moving_average(avg_val_loss), label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss Curves - {target_name} - {model_name}')
    output_dir = os.path.join('output', target_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

def train_model(X, y, input_dim, model_type, attention_type, epochs, batch_size, learning_rate, n_splits, seed, patience, target_name, model_name):
    set_seed(seed)  # 使用传入的随机种子
>>>>>>> b4e72cfbe2a520924c25a74b0da34ae5130df585
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_model = None
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    all_val_losses = []  # 用于存储所有折的验证损失

    # 初始化损失记录为每个fold的列表
    train_losses = []
    val_losses = []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # 增加数据量
        X_train, y_train = advanced_augment_data(X_train, y_train)

        model = initialize_model(model_type, input_dim, attention_type).to(device)
        model.device = device
        criterion = nn.MSELoss()

        train_dataset, val_dataset = prepare_dataset(X_train, y_train, X_val, y_val, model_type)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

<<<<<<< HEAD
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
=======
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)
>>>>>>> b4e72cfbe2a520924c25a74b0da34ae5130df585

        patience_counter = 0
        fold_val_losses = []  # 用于存储当前折的验证损失

        # 初始化当前fold的损失记录
        fold_train_losses = []
        fold_val_losses = []

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, model_type, reg_type, reg_weight)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    if model_type in ['TCN']:
                        X_batch = X_batch.permute(0, 2, 1).contiguous()
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

<<<<<<< HEAD
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            fold_val_losses.append(val_loss)  # 记录当前折的验证损失
=======
            val_loss /= len(val_loader)
            scheduler.step(val_loss)  # 更新调度器
            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)
>>>>>>> b4e72cfbe2a520924c25a74b0da34ae5130df585

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

<<<<<<< HEAD
        all_val_losses.append(fold_val_losses)  # 将当前折的验证损失添加到所有折的验证损失中

    avg_val_losses = average_loss_curves(all_val_losses)  # 计算所有折的验证损失曲线的平均值

    # 在结束前绘制损失曲线
    title = f"Loss Curve - {target_column} ({attention_type}-{model_type}) - {dataset_name}" if attention_type else f"Loss Curve - {target_column} ({model_type}) - {dataset_name}"
    plot_loss_curve(train_losses, avg_val_losses, title, target_column, model_type, attention_type, dataset_name)
=======
        # 保存当前fold的损失
        train_losses.append(fold_train_losses)
        val_losses.append(fold_val_losses)

    # 调用新函数进行绘图
    plot_loss_curves(train_losses, val_losses, n_splits, target_name, model_name)

>>>>>>> b4e72cfbe2a520924c25a74b0da34ae5130df585
    return best_model

def evaluate_model(model, X, y, feature_columns, target_column, model_type, attention_type, dataset_name,
                   title="模型评估", plot=False, overfitting=False):  # 添加参数 overfitting
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        if model_type in ['TCN']:
            X_tensor = X_tensor.squeeze(1)
        torch.tensor(y, dtype=torch.float32).to(device)
        y_pred = model(X_tensor).squeeze().cpu().numpy()
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    rpd = np.std(y) / rmse

    if plot and 0.85 <= r2 < 0.99 and not overfitting:
        plot_results(y, y_pred, title, model_type, sanitize_filename(target_column))
        shap_analysis(model, X, feature_columns, sanitize_filename(target_column), model_type, attention_type, dataset_name)
        lime_analysis(model, X, y, feature_columns, sanitize_filename(target_column), model_type, attention_type, dataset_name)

    return r2, rmse, rpd

<<<<<<< HEAD
def train_and_evaluate(X, y, input_dim, model_type, attention_type, epochs, batch_size, learning_rate, n_splits, seed, patience, feature_columns, target_column, dataset_name):
    # 使用网格搜索选择最佳正则化参数
    param_grid = {'reg_type': ['L1', 'L2'], 'reg_weight': [1e-4, 1e-3, 1e-2]}
    sklearn_model = SklearnWrapper(DCNN, input_dim, attention_type)  # 使用 DCNN 作为模型类
    best_params = grid_search_regularization(sklearn_model, X, y, param_grid)

=======
def train_and_evaluate(X, y, input_dim, model_type, attention_type, epochs, batch_size,
                       learning_rate, n_splits, seed, patience, feature_columns,
                       target_column, dataset_name):
>>>>>>> b4e72cfbe2a520924c25a74b0da34ae5130df585
    model = train_model(
        X, y, input_dim=input_dim,
        model_type=model_type,
        attention_type=attention_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_splits=n_splits,
        seed=seed,
<<<<<<< HEAD
        patience=patience,
        target_column=target_column,
        dataset_name=dataset_name,
        reg_type=best_params['reg_type'],  # 使用最佳正则化参数
        reg_weight=best_params['reg_weight']  # 使用最佳正则化参数
=======
        patience=patience,  # 传入 patience
        target_name=target_column,  # 添加参数
        model_name=model_type         # 添加参数
>>>>>>> b4e72cfbe2a520924c25a74b0da34ae5130df585
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=215, random_state=seed)
    # 先分别计算训练集和测试集指标，并禁止绘图
    train_metrics = evaluate_model(
        model, X_train, y_train, feature_columns, target_column,
        model_type, attention_type, dataset_name,
        plot=False
    )
    test_metrics = evaluate_model(
        model, X_val, y_val, feature_columns, target_column,
        model_type, attention_type, dataset_name,
        plot=False
    )
    # 判断是否过拟合
    overfitting_bool = (train_metrics[0] - test_metrics[0] > 0.1)
    # 如果不过拟合，则允许在测试集上绘图
    evaluate_model(
        model, X_val, y_val, feature_columns, target_column,
        model_type, attention_type, dataset_name,
        title=f"{dataset_name} - {target_column} - {attention_type} - {model_type}" if attention_type else f"{dataset_name} - {target_column} - {model_type}",
        plot=True,
        overfitting=overfitting_bool
    )
    return train_metrics, test_metrics

def process_dataset(file_path, dataset_name, target_columns, EPOCHS, BATCH_SIZE, LEARNING_RATE, N_SPLITS, SEED, model_types, attention_types, patience, results):
    X, y_dict, feature_columns = load_data(file_path, target_columns)
    X = preprocess_data(X)
    X = reduce_dimensionality(X, n_components=100)
    
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
                    target_column=target_column,
                    dataset_name=dataset_name
                )
                results.append(
                    (dataset_name, target_column, model_type, attention_type, train_metrics, test_metrics)
                )

# 新增函数: 获取配置参数
def configs():
    file_paths = [
        ("../datasets/data_spectral_bands_sgd_dr.xlsx", "SBSD"),
        ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
        ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNSBE"),
        ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
        ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD")
    ]
    target_columns = ["EOC", "SOC", "WOC", "TC", "OM"]
<<<<<<< HEAD
    model_types = ['DCNN']  # 'TCN','ResNet18','VGG7','DCNN','LSTM','CNN_LSTM'
    attention_types = [None, 'SE', 'ECA', 'CBAM', 'SA']  # 移除 SSLT
=======
    model_types = ['DCNN']  #'TCN','ResNet18','VGG7',,'LSTM','CNN_LSTM' 
    attention_types = [ None, 'SE','ECA', 'CBAM', 'SA']  # 移除 SSLT
    results = []

>>>>>>> b4e72cfbe2a520924c25a74b0da34ae5130df585
    SEED = 42
    EPOCHS = 300
    BATCH_SIZE = 64  # 将 batch size 从32增加到64
    LEARNING_RATE = 0.0001
    N_SPLITS = 5
    PATIENCE = 20  # 定义 patience
    return file_paths, target_columns, model_types, attention_types, SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE, N_SPLITS, PATIENCE

<<<<<<< HEAD
# 新增函数: 处理结果并保存
def handle_results(results):
    headers = ["Dataset", "Target", "Model", "Attention", "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD"]
    table = [
        [dataset_name, target_column, model_type, attention_type, 
         f"{train_metrics[0]:.4f}", f"{train_metrics[1]:.4f}", f"{train_metrics[2]::.4f}", 
         f"{test_metrics[0]:.4f}", f"{test_metrics[1]:.4f}", f"{test_metrics[2]:.4f}"]
=======
    for file_path, dataset_name in file_paths:
        process_dataset(
            file_path, dataset_name, target_columns,
            EPOCHS, BATCH_SIZE, LEARNING_RATE, N_SPLITS, SEED,
            model_types, attention_types, PATIENCE, results  # 传入 patience
        )

    table = [
        [
            dataset_name, 
            target_column, 
            model_type, 
            attention_type, 
            f"{train_metrics[0]:.4f}", 
            f"{train_metrics[1]:.4f}", 
            f"{train_metrics[2]:.4f}", 
            f"{test_metrics[0]:.4f}", 
            f"{test_metrics[1]:.4f}", 
            f"{test_metrics[2]:.4f}",
            "Yes" if (train_metrics[0] - test_metrics[0] > 0.1) else "No"  # Determine overfitting
        ]
>>>>>>> b4e72cfbe2a520924c25a74b0da34ae5130df585
        for dataset_name, target_column, model_type, attention_type, train_metrics, test_metrics in results
    ]

    headers = [
        "Dataset", "Target", "Model", "Attention", 
        "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD",
        "Overfitting"  # New column header
    ]

    print("\nResults Summary:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

    results_df = pd.DataFrame(table, columns=headers)
    results_df.to_excel(f'./output/results_summary.xlsx', index=False)

# 修改 main 函数
def main():
    file_paths, target_columns, model_types, attention_types, SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE, N_SPLITS, PATIENCE = configs()
    results = []

    for file_path, dataset_name in file_paths:
        process_dataset(
            file_path, dataset_name, target_columns,
            EPOCHS, BATCH_SIZE, LEARNING_RATE, N_SPLITS, SEED,
            model_types, attention_types, PATIENCE, results  # 传入 patience
        )

    handle_results(results)

if __name__ == "__main__":
    main()
