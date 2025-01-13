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
import optuna

# Import models
from models.DCNN import DCNN
from models.ResNet18 import ResNet18
from models.VGG7 import VGG7

# Import utility functions
from utils import plot_results, shap_analysis, lime_analysis, set_seed, augment_data, load_data, preprocess_data, \
    sanitize_filename, plot_accuracy_and_loss

# 设置中文字体并添加备用字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STFangsong', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

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
model_types = ['DCNN', 'ResNet18', 'VGG7']
attention_types = [None, 'SE', 'ECA', 'CBAM', 'SA']

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

def train_model(X, y, input_dim, model_type, attention_type, device, dataset_name,
                target_column, epochs, batch_size, learning_rate, patience):
    set_seed(42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_val_loss = float('inf')

    all_train_losses = []
    all_val_losses = []
    all_r2_scores = []
    all_rmse_values = []
    all_rpd_values = []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/5")
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

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)  # More aggressive factor

        patience_counter = 0

        train_losses = []  # Initialize list for this fold's training losses
        val_losses = []    # Initialize list for this fold's validation losses
        r2_scores_fold = []   # Initialize list for this fold's R² scores
        rmse_values_fold = [] # Initialize list for this fold's RMSE values
        rpd_values_fold = []  # Initialize list for this fold's RPD values

        for epoch in range(epochs):  # Minimal dummy loop
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

            # Calculate R², RMSE, and RPD for validation set
            y_val_pred = model(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)).squeeze().cpu().detach().numpy()
            val_r2 = r2_score(y_val, y_val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            rpd = np.std(y_val) / rmse

            # Append losses for this epoch
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            r2_scores_fold.append(val_r2)
            rmse_values_fold.append(rmse)
            rpd_values_fold.append(rpd)

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

        # After each fold, append the fold's losses and metrics to the main lists
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_r2_scores.append(r2_scores_fold)
        all_rmse_values.append(rmse_values_fold)
        all_rpd_values.append(rpd_values_fold)

    # Compute average metrics per epoch across all folds
    avg_r2_scores = np.mean(all_r2_scores, axis=0)
    avg_rmse_values = np.mean(all_rmse_values, axis=0)
    avg_rpd_values = np.mean(all_rpd_values, axis=0)

    # After training
    plot_accuracy_and_loss(
        epochs=range(1, epochs + 1),
        train_losses=all_train_losses,  # List of lists
        val_losses=all_val_losses,      # List of lists
        r2_scores=avg_r2_scores,        # Averaged per epoch
        rmse_values=avg_rmse_values,    # Averaged per epoch
        rpd_values=avg_rpd_values,      # Averaged per epoch
        title=f"{dataset_name} - {target_column} - {model_type}",
        target_column=target_column
    )

    return best_model, best_val_loss

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

def train_and_evaluate(X, y, input_dim, model_type, attention_type, device, feature_columns, target_column, dataset_name, hyperparams):
    model, best_val_loss = train_model(
        X, y, input_dim=input_dim,
        model_type=model_type,
        attention_type=attention_type,
        device=device,
        dataset_name=dataset_name,
        target_column=target_column,          # Added target_column
        epochs=hyperparams['epochs'],        # Updated to use hyperparams
        batch_size=hyperparams['batch_size'],
        learning_rate=hyperparams['learning_rate'],
        patience=hyperparams['patience']
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)
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
    return train_metrics, test_metrics, best_val_loss

def process_dataset(X, y_dict, feature_columns, dataset_name, device, model_types, attention_types, results):
    for target_column, y in y_dict.items():
        print(f"Processing {target_column} from {dataset_name}")
        for model_type in model_types:
            for attention_type in attention_types:
                print(f"Training {model_type} with attention type: {attention_type}")
                hyperparams = {                            # Added hyperparams dictionary
                    'epochs': 100,
                    'batch_size': 32,
                    'learning_rate': 1e-3,
                    'patience': 100
                }
                train_metrics, test_metrics, _ = train_and_evaluate(
                    X, y, input_dim=X.shape[1],
                    model_type=model_type,
                    attention_type=attention_type,
                    device=device,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    dataset_name=dataset_name,
                    hyperparams=hyperparams                   # Passed hyperparams
                )
                results.append(
                    (dataset_name, target_column, model_type, attention_type, train_metrics, test_metrics)
                )
                print(f"Dataset: {dataset_name}, Target: {target_column}, Model: {model_type}, Attention: {attention_type}, Train R²: {train_metrics[0]}, Train Loss: {train_metrics[1]}, Val R²: {test_metrics[0]}, Val Loss: {test_metrics[1]}")

def objective(trial, X, y, model_type, feature_columns, target_column, dataset_name):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    epochs = trial.suggest_int('epochs', 10, 200)
    patience = trial.suggest_int('patience', 10, 300)
    hyperparams = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'patience': patience
    }
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)
    test_metrics = evaluate_model(
        train_model(
            X_train, y_train, input_dim=X.shape[1],
            model_type=model_type,  # Use the model_type from the outer scope
            attention_type=None,
            device=device,
            dataset_name=dataset_name,               # Added dataset_name
            target_column=target_column,             # Added target_column
            epochs=hyperparams['epochs'],
            batch_size=hyperparams['batch_size'],
            learning_rate=hyperparams['learning_rate'],
            patience=hyperparams['patience']
        )[0],  # Get the best model from train_model
        X_val, y_val,
        feature_columns=feature_columns,  # Provided feature_columns
        target_column=target_column,      # Provided target_column
        model_type=model_type,
        attention_type=None,
        dataset_name=dataset_name,        # Provided dataset_name
        plot=False
    )
    return test_metrics[1]  # Assuming val_loss is at index 1

def main():
    results = []
    global device

    for file_path, dataset_name in file_paths:
        X, y_dict, feature_columns = load_data(file_path, target_columns)
        X = preprocess_data(X)
        pca = PCA(n_components=50)
        X = pca.fit_transform(X)

        process_dataset(
            X, y_dict, feature_columns, dataset_name,
            device,
            model_types, attention_types, results
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

    study = optuna.create_study(direction='minimize')
    for model_type in model_types:
        for target_column, y in y_dict.items():
            study.optimize(lambda trial: objective(trial, X, y, model_type, feature_columns, target_column, dataset_name), n_trials=20)
    best_params = study.best_params
    print("Best hyperparameters found by Optuna:", best_params)

    # Optionally retrain using best hyperparameters
    final_model = train_model(
        X, y,
        input_dim=X.shape[1],
        model_type='DCNN', 
        attention_type=None,
        device=device,
        dataset_name=dataset_name,               # Added dataset_name
        target_column=target_column,             # Added target_column
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        learning_rate=best_params['learning_rate'],
        patience=best_params['patience']
    )
    torch.save(final_model.state_dict(), './output/model.pth')  # Save the best model to output folder

if __name__ == "__main__":
    main()
