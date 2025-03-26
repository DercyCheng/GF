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
# Add ML models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Ridge
# Add PLSR
from sklearn.cross_decomposition import PLSRegression

# Import models
from models.DCNN import DCNN
from models.ResNet18 import ResNet18
from models.VGG7 import VGG7
# from models.MultiModalNet import MultiModalNet

# Import utility functions
from utils import (plot_results, shap_analysis, lime_analysis, set_seed, augment_data, load_data, preprocess_data,
                  sanitize_filename, plot_accuracy_and_loss, plot_spectral_curves, plot_correlation_matrix,
                  plot_regression_diagnostics, plot_feature_importance)

# 设置中文字体并添加备用字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STFangsong', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

file_paths = [
    ("../datasets/data_spectral_bands_dwt.xlsx", "SBDWT"),
    ("../datasets/data_spectral_bands_msc.xlsx", "SBMSC"),
    ("../datasets/data_spectral_bands_msc_dwt.xlsx", "SBMSC-DWT"),
    ("../datasets/data_spectral_bands_sgd_dr.xlsx", "SBSD"),
    ("../datasets/data_spectral_bands_snv.xlsx", "SBSNV"),
    ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
    ("../datasets/data_soil_nutrients_spectral_bands_dwt.xlsx", "SNSBDWT"),
    ("../datasets/data_soil_nutrients_spectral_bands_msc.xlsx", "SNSBMSC"),
    ("../datasets/data_soil_nutrients_spectral_bands_msc_dwt.xlsx", "SNSBMSC-DWT"),
    ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
    ("../datasets/data_soil_nutrients_spectral_bands_snv.xlsx", "SNSBSNV"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNSBE"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment_dwt.xlsx", "SNSBEDWT"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment_msc.xlsx", "SNSBEMSC"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment_msc_dwt.xlsx", "SNSBEMSC-DWT"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment_snv.xlsx", "SNSBESNV"),
]

target_columns = ["SOC", "EOC", "WOC", "TC", "OM"]
model_types = [
    'DCNN',
    'ResNet18', 'VGG7', 'ECA-DCNN', 'CBAM-DCNN', 'SE-DCNN',
    # Add ML models
    'RandomForest', 'GradientBoosting', 'SVR', 'ElasticNet', 'Ridge', 'PLSR'  # Add PLSR model
]

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def initialize_model(model_type, input_dim, attention_type=None):
    # Parse attention type from model name if it contains a hyphen
    if '-' in model_type:
        attention_type, base_model = model_type.split('-')
    else:
        base_model = model_type
        attention_type = None

    model_classes = {
        'ResNet18': ResNet18,
        'VGG7': VGG7,
        'DCNN': DCNN
    }
    
    if base_model not in model_classes:
        raise ValueError(f"Unsupported model type: {base_model}")
    elif base_model in ['DCNN']:
        return model_classes[base_model](input_dim, attention_type=attention_type)
    # elif 'MultiModal' in model_type:
    #     spectral_dim = input_dim
    #     env_dim = len(environment_info)
    #     nutrient_dim = len(soil_nutrients)
    #     return MultiModalNet(spectral_dim, env_dim, nutrient_dim, attention_type)
    else:
        return model_classes[base_model](input_dim)

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
        
        # Remove dropout during training to increase model capacity
        if hasattr(model, 'apply_dropout'):
            model.apply_dropout(False)
            
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        
        # Reduce L2 regularization
        l2_lambda = 0.001  # Decreased from 0.01
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_norm
        
        loss.backward()
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

        # Replace custom augmentation with the augment_data function
        X_train, y_train = augment_data(X_train, y_train)
        
        model = initialize_model(model_type, input_dim, attention_type).to(device)
        model.device = device
        criterion = nn.MSELoss()

        train_dataset, val_dataset = prepare_dataset(X_train, y_train, X_val, y_val, model_type)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Use Adam instead of AdamW to reduce regularization
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Less aggressive learning rate scheduling
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

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
        if hasattr(model, 'apply_dropout'):
            model.apply_dropout(False)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        torch.tensor(y, dtype=torch.float32).to(device)
        y_pred = model(X_tensor).squeeze().cpu().numpy()
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    rpd = np.std(y) / rmse

    if plot and 0.85 <= r2 < 0.99:
        plot_results(y, y_pred, title, model_type, sanitize_filename(target_column))
        shap_analysis(model, X, feature_columns, sanitize_filename(target_column), model_type, attention_type, dataset_name)
        lime_analysis(model, X, y, feature_columns, sanitize_filename(target_column), model_type, attention_type, dataset_name)
        
        # Add new visualization methods
        plot_spectral_curves(X, feature_columns, target_column, model_type, dataset_name)
        plot_correlation_matrix(X, feature_columns, target_column, model_type, dataset_name)
        plot_regression_diagnostics(y, y_pred, target_column, model_type, dataset_name)
        
        # Calculate feature importance based on correlation with predictions
        # For deep learning models, this is a simple approximation
        importance_scores = np.abs(np.corrcoef(X.T, y_pred)[:-1, -1])
        plot_feature_importance(importance_scores, feature_columns, target_column, model_type, dataset_name)
        
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

def train_and_evaluate_ml_model(model, X, y, feature_columns, target_column, dataset_name, model_type, plot=False):
    """Train and evaluate machine learning models"""
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # Train the model
    if isinstance(model, PLSRegression):
        # For PLSR, reshape y if needed
        if len(y_train.shape) == 1:
            y_train_reshaped = y_train.reshape(-1, 1)
        else:
            y_train_reshaped = y_train
        model.fit(X_train, y_train_reshaped)
    else:
        model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # For PLSR, predictions might need reshaping
    if isinstance(model, PLSRegression):
        if len(y_train_pred.shape) > 1 and y_train_pred.shape[1] == 1:
            y_train_pred = y_train_pred.flatten()
        if len(y_val_pred.shape) > 1 and y_val_pred.shape[1] == 1:
            y_val_pred = y_val_pred.flatten()
    
    # Calculate metrics for training set
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_rpd = np.std(y_train) / train_rmse
    
    # Calculate metrics for validation set
    val_r2 = r2_score(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_rpd = np.std(y_val) / val_rmse
    
    # We're now disabling plotting for ML models as requested
    # But keeping feature importance analysis for records (just not plotting)
    return (train_r2, train_rmse, train_rpd), (val_r2, val_rmse, val_rpd)

def process_dataset(X, y_dict, feature_columns, dataset_name, device, model_types, results):
    for target_column, y in y_dict.items():
        print(f"Processing {target_column} from {dataset_name}")
        for model_type in model_types:
            print(f"Training {model_type}")
            
            # Handle machine learning models
            if model_type in ['RandomForest', 'GradientBoosting', 'SVR', 'ElasticNet', 'Ridge', 'PLSR']:
                if model_type == 'RandomForest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_type == 'GradientBoosting':
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                elif model_type == 'SVR':
                    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                elif model_type == 'ElasticNet':
                    model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
                elif model_type == 'Ridge':
                    model = Ridge(alpha=1.0, random_state=42)
                elif model_type == 'PLSR':
                    # Start with n_components = 10, will be optimized with Optuna
                    model = PLSRegression(n_components=10, scale=True)
                    
                train_metrics, test_metrics = train_and_evaluate_ml_model(
                    model, X, y, feature_columns, target_column, dataset_name, model_type, plot=False  # Set plot=False for ML models
                )
            else:
                # Deep learning models
                hyperparams = {
                    'epochs': 150,  # Increased from 100
                    'batch_size': 16,  # Decreased from 32 for better fitting
                    'learning_rate': 2e-3,  # Increased from 1e-3
                    'patience': 150  # Increased from 100
                }
                train_metrics, test_metrics, _ = train_and_evaluate(
                    X, y, input_dim=X.shape[1],
                    model_type=model_type,
                    attention_type=None,
                    device=device,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    dataset_name=dataset_name,
                    hyperparams=hyperparams
                )
                
            results.append(
                (dataset_name, target_column, model_type, train_metrics, test_metrics)
            )
            print(f"Dataset: {dataset_name}, Target: {target_column}, Model: {model_type}, "
                  f"Train R²: {train_metrics[0]:.4f}, Train RMSE: {train_metrics[1]:.4f}, "
                  f"Val R²: {test_metrics[0]:.4f}, Val RMSE: {test_metrics[1]:.4f}")

def objective(trial, X, y, model_type, feature_columns, target_column, dataset_name):
    # For ML models
    if model_type == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        train_metrics, test_metrics = train_and_evaluate_ml_model(
            model, X, y, feature_columns, target_column, dataset_name, model_type, plot=False
        )
    elif model_type == 'GradientBoosting':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        train_metrics, test_metrics = train_and_evaluate_ml_model(
            model, X, y, feature_columns, target_column, dataset_name, model_type, plot=False
        )
    elif model_type == 'SVR':
        C = trial.suggest_loguniform('C', 1e-2, 1e2)
        epsilon = trial.suggest_loguniform('epsilon', 1e-3, 1)
        model = SVR(kernel='rbf', C=C, epsilon=epsilon)
        train_metrics, test_metrics = train_and_evaluate_ml_model(
            model, X, y, feature_columns, target_column, dataset_name, model_type, plot=False
        )
    elif model_type == 'ElasticNet':
        alpha = trial.suggest_loguniform('alpha', 1e-4, 1)
        l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        train_metrics, test_metrics = train_and_evaluate_ml_model(
            model, X, y, feature_columns, target_column, dataset_name, model_type, plot=False
        )
    elif model_type == 'Ridge':
        alpha = trial.suggest_loguniform('alpha', 1e-4, 1)
        model = Ridge(alpha=alpha, random_state=42)
        train_metrics, test_metrics = train_and_evaluate_ml_model(
            model, X, y, feature_columns, target_column, dataset_name, model_type, plot=False
        )
    elif model_type == 'PLSR':
        n_components = trial.suggest_int('n_components', 2, min(30, X.shape[1]-1))  # Number of components to keep
        model = PLSRegression(n_components=n_components, scale=True)
        train_metrics, test_metrics = train_and_evaluate_ml_model(
            model, X, y, feature_columns, target_column, dataset_name, model_type, plot=False
        )
    else:
        # For DL models, use the existing code
        # ...existing code for DL models...
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
                model_type=model_type,
                attention_type=None,
                device=device,
                dataset_name=dataset_name,
                target_column=target_column,
                epochs=hyperparams['epochs'],
                batch_size=hyperparams['batch_size'],
                learning_rate=hyperparams['learning_rate'],
                patience=hyperparams['patience']
            )[0],
            X_val, y_val,
            feature_columns=feature_columns,
            target_column=target_column,
            model_type=model_type,
            attention_type=None,
            dataset_name=dataset_name,
            plot=False
        )
        
    return test_metrics[1]  # RMSE - lower is better

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
            device, model_types, results
        )

    headers = ["Dataset", "Target", "Model", "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD"]
    table = [
        [dataset_name, target_column, model_type,
         f"{train_metrics[0]:.4f}", f"{train_metrics[1]:.4f}", f"{train_metrics[2]:.4f}", 
         f"{test_metrics[0]:.4f}", f"{test_metrics[1]:.4f}", f"{test_metrics[2]:.4f}"]
        for dataset_name, target_column, model_type, train_metrics, test_metrics in results
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

    # Save final model - different handling based on model type
    if 'DCNN' in best_params:  # Assume the best model is a DL model if it has DCNN in params
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
        )[0]
        # torch.save(final_model.state_dict(), './output/model.pth')
    else:
        # For ML models, use joblib for saving
        import joblib
        if 'n_estimators' in best_params:
            if 'learning_rate' in best_params:  # GradientBoosting
                final_model = GradientBoostingRegressor(
                    n_estimators=best_params['n_estimators'],
                    learning_rate=best_params['learning_rate'],
                    random_state=42
                )
            else:  # RandomForest
                final_model = RandomForestRegressor(
                    n_estimators=best_params['n_estimators'],
                    max_depth=best_params.get('max_depth', None),
                    random_state=42
                )
        elif 'C' in best_params:  # SVR
            final_model = SVR(
                kernel='rbf', 
                C=best_params['C'], 
                epsilon=best_params['epsilon']
            )
        elif 'alpha' in best_params:
            if 'l1_ratio' in best_params:  # ElasticNet
                final_model = ElasticNet(
                    alpha=best_params['alpha'],
                    l1_ratio=best_params['l1_ratio'],
                    random_state=42
                )
            else:  # Ridge
                final_model = Ridge(
                    alpha=best_params['alpha'],
                    random_state=42
                )
        elif 'n_components' in best_params:  # PLSR
            final_model = PLSRegression(
                n_components=best_params['n_components'],
                scale=True
            )
        final_model.fit(X, y)
        # joblib.dump(final_model, './output/ml_model.joblib')

if __name__ == "__main__":
    main()
