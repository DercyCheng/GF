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
from sklearn.decomposition import PCA

# Import models
from models.DCNN import DCNN
from models.ResNet18 import ResNet18
from models.VGG7 import VGG7

# Import utility functions
from utils import plot_results, shap_analysis, lime_analysis, set_seed, augment_data, load_data, preprocess_data, \
    sanitize_filename

# 设置中文字体并添加备用字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STFangsong', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def initialize_model(model_type, input_dim, attention_type=None):
    model_classes = {
        'ResNet18': ResNet18,
        'VGG7': VGG7,
        'DCNN': DCNN
    }
    if model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_type}")
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

def train_one_epoch(model, train_loader, optimizer, criterion, device, model_type):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ...existing code...
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    return train_loss / len(train_loader.dataset)

def train_model(X, y, input_dim, model_type, attention_type, epochs, batch_size, learning_rate, n_splits, seed, patience):
    set_seed(seed)  # 使用传入的随机种子
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_model = None
    best_val_loss = float('inf')

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{n_splits}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Enhance data augmentation
        X_train, y_train = augment_data(X_train, y_train)  # Ensure augment_data provides sufficient augmentation

        model = initialize_model(model_type, input_dim, attention_type).to(device)
        model.device = device
        criterion = nn.MSELoss()

        train_dataset, val_dataset = prepare_dataset(X_train, y_train, X_val, y_val, model_type)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Changed optimizer to Adam with higher learning rate
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)  # More aggressive factor

        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, model_type)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

            val_loss /= len(val_loader)
            scheduler.step(val_loss)  # 更新调度器

            # Calculate R² for train and validation sets
            train_r2 = r2_score(y_train, model(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)).squeeze().cpu().detach().numpy())
            val_r2 = r2_score(y_val, model(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)).squeeze().cpu().detach().numpy())

            # Modify early stopping to monitor validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    return best_model

def evaluate_model(model, X, y, feature_columns, target_column, model_type, attention_type, dataset_name,
                   title="模型评估", plot=False):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
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
        patience=patience  # 传入 patience
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=seed)
    test_metrics = evaluate_model(
        model, X_val, y_val, feature_columns, target_column,
        model_type, attention_type, dataset_name,
        title=f"{dataset_name} - {target_column} - {attention_type} - {model_type}" if attention_type else f"{dataset_name} - {target_column} - {model_type}",
        plot=True
    )
    train_metrics = evaluate_model(
        model, X_train, y_train, feature_columns, target_column, model_type,
        attention_type, dataset_name,
        title=f"{dataset_name} - {target_column} - Train - {attention_type} - {model_type}" if attention_type else f"{dataset_name} - {target_column} - Train - {model_type}",
        plot=False
    )
    return train_metrics, test_metrics

def process_dataset(file_path, dataset_name, target_columns, EPOCHS, BATCH_SIZE, LEARNING_RATE, N_SPLITS, SEED, model_types, attention_types, patience, results):
    X, y_dict, feature_columns = load_data(file_path, target_columns)
    X = preprocess_data(X)
    pca = PCA(n_components=50)
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
                    target_column=target_column,
                    dataset_name=dataset_name
                )
                results.append(
                    (dataset_name, target_column, model_type, attention_type, train_metrics, test_metrics)
                )
                print(f"Dataset: {dataset_name}, Target: {target_column}, Model: {model_type}, Attention: {attention_type}, Train R²: {train_metrics[0]}, Train Loss: {train_metrics[1]}, Val R²: {test_metrics[0]}, Val Loss: {test_metrics[1]}")

def main():
    file_paths = [
        # ("../datasets/data_spectral_bands_sgd_dr.xlsx", "SBSD"),
        # ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
        # ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNSBE"),
        # ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
        ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD")
    ]
    target_columns = ["SOC", "EOC", "WOC", "TC", "OM"]
    model_types = ['DCNN','ResNet18','VGG7']
    attention_types = [ None, 'SE','ECA', 'CBAM', 'SA']
    results = []

    SEED = 42
    EPOCHS = 1000  # Increased from 50 to allow more training epochs
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    N_SPLITS = 5
    PATIENCE = 500  # Increased from 20 to allow more patience for early stopping

    for file_path, dataset_name in file_paths:
        process_dataset(
            file_path, dataset_name, target_columns,
            EPOCHS, BATCH_SIZE, LEARNING_RATE, N_SPLITS, SEED,
            model_types, attention_types, PATIENCE, results  # 传入 patience
        )

    headers = ["Dataset", "Target", "Model", "Attention", "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD"]
    table = [
        [dataset_name, target_column, model_type, attention_type, 
         f"{train_metrics[0]:.4f}", f"{train_metrics[1]:.4f}", f"{train_metrics[2]:.4f}", 
         f"{test_metrics[0]:.4f}", f"{test_metrics[1]:.4f}", f"{test_metrics[2]:.4f}"]
        for dataset_name, target_column, model_type, attention_type, train_metrics, test_metrics in results
    ]

    print("\nResults Summary:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

    results_df = pd.DataFrame(table, columns=headers)
    results_df.to_excel(f'./output/results_summary.xlsx', index=False)

if __name__ == "__main__":
    main()
