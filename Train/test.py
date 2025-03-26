import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from tabulate import tabulate
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import optuna
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.cross_decomposition import PLSRegression
import joblib

# Import models
from models.DCNN import DCNN
from models.ResNet18 import ResNet18
from models.VGG7 import VGG7

# Import utility functions
from utils import (plot_results, shap_analysis, lime_analysis, set_seed, augment_data, preprocess_data,
                  sanitize_filename, plot_accuracy_and_loss, plot_spectral_curves, plot_correlation_matrix,
                  plot_regression_diagnostics, plot_feature_importance)

# Configure plot settings
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STFangsong', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Global configuration
TARGET_COLUMNS = ["TC", "EOC", "SOC", "WOC", "OM"]
MODEL_TYPES = [
    'DCNN', 'ResNet18', 'VGG7', 'ECA-DCNN', 'CBAM-DCNN', 'SE-DCNN',
    'RandomForest', 'GradientBoosting', 'SVR', 'ElasticNet', 'Ridge', 'PLSR'
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
VISUALIZATION_THRESHOLD = 0.85

# Dataset paths
FILE_PATHS = [
    # Raw spectral bands with different preprocessing techniques
    ("../datasets/train/01_raw_spectral_bands_sgd_dr.csv", "Train-SBSD"),
    ("../datasets/test/01_raw_spectral_bands_sgd_dr.csv", "Test-SBSD"),
    ("../datasets/train/02_raw_spectral_bands_snv.csv", "Train-SBSNV"),
    ("../datasets/test/02_raw_spectral_bands_snv.csv", "Test-SBSNV"),
    ("../datasets/train/03_raw_spectral_bands_msc.csv", "Train-SBMSC"),
    ("../datasets/test/03_raw_spectral_bands_msc.csv", "Test-SBMSC"),
    ("../datasets/train/04_raw_spectral_bands_dwt.csv", "Train-SBDWT"),
    ("../datasets/test/04_raw_spectral_bands_dwt.csv", "Test-SBDWT"),
    
    # Soil nutrients + spectral bands
    ("../datasets/train/05_soil_nutrients_spectral_bands_raw.csv", "Train-SNSB"),
    ("../datasets/test/05_soil_nutrients_spectral_bands_raw.csv", "Test-SNSB"),
    ("../datasets/train/06_soil_nutrients_spectral_bands_sgd_dr.csv", "Train-SNSBSD"),
    ("../datasets/test/06_soil_nutrients_spectral_bands_sgd_dr.csv", "Test-SNSBSD"),
    ("../datasets/train/07_soil_nutrients_spectral_bands_snv.csv", "Train-SNSBSNV"),
    ("../datasets/test/07_soil_nutrients_spectral_bands_snv.csv", "Test-SNSBSNV"),
    ("../datasets/train/08_soil_nutrients_spectral_bands_msc.csv", "Train-SNSBMSC"),
    ("../datasets/test/08_soil_nutrients_spectral_bands_msc.csv", "Test-SNSBMSC"),
    ("../datasets/train/09_soil_nutrients_spectral_bands_dwt.csv", "Train-SNSBDWT"),
    ("../datasets/test/09_soil_nutrients_spectral_bands_dwt.csv", "Test-SNSBDWT"),
    ("../datasets/train/10_soil_nutrients_spectral_bands_msc_dwt.csv", "Train-SNSBMSC-DWT"),
    ("../datasets/test/10_soil_nutrients_spectral_bands_msc_dwt.csv", "Test-SNSBMSC-DWT"),
    
    # Soil nutrients + spectral bands + environmental info
    ("../datasets/train/11_soil_nutrients_spectral_bands_environment_raw.csv", "Train-SNSBE"),
    ("../datasets/test/11_soil_nutrients_spectral_bands_environment_raw.csv", "Test-SNSBE"),
    ("../datasets/train/12_soil_nutrients_spectral_bands_environment_sgd_dr.csv", "Train-SNSBESD"),
    ("../datasets/test/12_soil_nutrients_spectral_bands_environment_sgd_dr.csv", "Test-SNSBESD"),
    ("../datasets/train/13_soil_nutrients_spectral_bands_environment_snv.csv", "Train-SNSBESNV"),
    ("../datasets/test/13_soil_nutrients_spectral_bands_environment_snv.csv", "Test-SNSBESNV"),
    ("../datasets/train/14_soil_nutrients_spectral_bands_environment_msc.csv", "Train-SNSBEMSC"),
    ("../datasets/test/14_soil_nutrients_spectral_bands_environment_msc.csv", "Test-SNSBEMSC"),
    ("../datasets/train/15_soil_nutrients_spectral_bands_environment_dwt.csv", "Train-SNSBEDWT"),
    ("../datasets/test/15_soil_nutrients_spectral_bands_environment_dwt.csv", "Test-SNSBEDWT"),
    ("../datasets/train/16_soil_nutrients_spectral_bands_environment_msc_dwt.csv", "Train-SNSBEMSC-DWT"),
    ("../datasets/test/16_soil_nutrients_spectral_bands_environment_msc_dwt.csv", "Test-SNSBEMSC-DWT"),
]

def load_data(file_path, target_columns):
    """Load and preprocess data from CSV or Excel files."""
    file_extension = file_path.split('.')[-1].lower()
    
    try:
        if (file_extension == 'xlsx' or file_extension == 'xls'):
            data = pd.read_excel(file_path)
            print(f"Excel data loaded successfully from {file_path}!")
        elif file_extension == 'csv':
            data = pd.read_csv(file_path)
            print(f"CSV data loaded successfully from {file_path}!")
        else:
            print(f"Unsupported file extension: {file_extension}")
            return None, None, None
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        return None, None, None

    data.columns = data.columns.map(str)
    
    # Extract available target columns
    available_targets = [col for col in target_columns if col in data.columns]
    if not available_targets:
        print(f"Warning: None of the target columns {target_columns} found in {file_path}")
        return None, None, None
    
    # Clean data and extract features/targets
    data = data.dropna(subset=available_targets)
    y_dict = {target_column: data[target_column].values for target_column in available_targets}
    X_data = data.drop(columns=available_targets, errors='ignore')
    feature_columns = X_data.select_dtypes(include=[np.number]).columns.tolist()
    x = X_data[feature_columns].values

    print(f"Number of features: {len(feature_columns)}, Number of samples: {x.shape[0]}")
    return x, y_dict, feature_columns

def preprocess_data(X, scaler=None):
    """
    Enhanced preprocessing function that prevents data leakage.
    If scaler is None, fit a new scaler on the data and return it.
    If scaler is provided, use it to transform the data.
    """
    X_processed = X.copy()
    
    # Handle outliers using winsorization
    if scaler is None:
        # Only fit the winsorization parameters on training data
        quantile_ranges = []
        for col in range(X_processed.shape[1]):
            q1, q3 = np.percentile(X_processed[:, col], [1, 99])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            quantile_ranges.append((lower_bound, upper_bound))
            X_processed[:, col] = np.clip(X_processed[:, col], lower_bound, upper_bound)
        
        # Fit standardization parameters on training data
        mean_std_values = []
        for col in range(X_processed.shape[1]):
            mean = np.mean(X_processed[:, col])
            std = np.std(X_processed[:, col])
            if std > 0:
                X_processed[:, col] = (X_processed[:, col] - mean) / std
            mean_std_values.append((mean, std))
        
        return X_processed, {'quantile_ranges': quantile_ranges, 'mean_std_values': mean_std_values}
    else:
        # Use pre-computed parameters to transform test data
        quantile_ranges = scaler['quantile_ranges']
        mean_std_values = scaler['mean_std_values']
        
        # Apply same winsorization to test data
        for col in range(X_processed.shape[1]):
            lower_bound, upper_bound = quantile_ranges[col]
            X_processed[:, col] = np.clip(X_processed[:, col], lower_bound, upper_bound)
        
        # Apply same standardization to test data
        for col in range(X_processed.shape[1]):
            mean, std = mean_std_values[col]
            if std > 0:
                X_processed[:, col] = (X_processed[:, col] - mean) / std
        
        return X_processed

def select_features(X_train, y_train, X_test, n_features=None, selector=None):
    """
    Select most important features while preventing data leakage.
    If selector is None, fit a new selector on the training data.
    If selector is provided, use it to transform both train and test data.
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    
    if n_features is None:
        n_features = min(50, X_train.shape[1])
    
    if selector is None:
        # Fit feature selector only on training data
        selector = SelectKBest(mutual_info_regression, k=n_features)
        selector.fit(X_train, y_train)
    
    # Apply the same transformation to both datasets
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    return X_train_selected, X_test_selected, selector

def initialize_model(model_type, input_dim, attention_type=None):
    """Initialize the appropriate model based on model_type."""
    # Parse attention type if present
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
    else:
        return model_classes[base_model](input_dim)

def prepare_dataset(X_train, y_train, X_val, y_val):
    """Create PyTorch datasets from numpy arrays."""
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
                                torch.tensor(y_val, dtype=torch.float32))
    return train_dataset, val_dataset

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train a model for one epoch with enhanced training techniques."""
    # Add mixup data augmentation during training
    use_mixup = True
    alpha = 0.2
    
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        if use_mixup:
            # Apply mixup augmentation
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
            index = torch.randperm(X_batch.size(0)).to(device)
            mixed_X = lam * X_batch + (1 - lam) * X_batch[index]
            X_batch = mixed_X
            mixed_y = lam * y_batch + (1 - lam) * y_batch[index]
            y_batch = mixed_y
            
        optimizer.zero_grad()
        
        # Apply dropout more selectively
        if hasattr(model, 'apply_dropout'):
            model.apply_dropout(True)  # Enable dropout during training
            
        outputs = model(X_batch).squeeze()
        
        # Add label smoothing
        smoothing = 0.05
        if smoothing > 0:
            y_batch = y_batch * (1 - smoothing) + smoothing * y_batch.mean()
            
        loss = criterion(outputs, y_batch)
        
        # Add stronger L2 regularization for models prone to overfitting
        l2_lambda = 0.0005
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_norm
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    
    return train_loss / len(train_loader.dataset)

def train_deep_model(X, y, input_dim, model_type, attention_type, device, dataset_name, target_column, hyperparams):
    """Train a deep learning model using k-fold cross-validation."""
    set_seed(42)
    epochs = hyperparams['epochs']
    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    patience = hyperparams['patience']
    
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

        # Data augmentation
        X_train, y_train = augment_data(X_train, y_train)
        
        model = initialize_model(model_type, input_dim, attention_type).to(device)
        model.device = device
        criterion = nn.MSELoss()

        train_dataset, val_dataset = prepare_dataset(X_train, y_train, X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        patience_counter = 0
        fold_metrics = {'train_losses': [], 'val_losses': [], 'r2_scores': [], 'rmse_values': [], 'rpd_values': []}

        for epoch in range(epochs):
            # Training phase
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

            # Validation phase
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

            # Calculate metrics
            y_val_pred = model(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)).squeeze().cpu().detach().numpy()
            val_r2 = r2_score(y_val, y_val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            rpd = np.std(y_val) / rmse

            # Store metrics
            fold_metrics['train_losses'].append(train_loss)
            fold_metrics['val_losses'].append(val_loss)
            fold_metrics['r2_scores'].append(val_r2)
            fold_metrics['rmse_values'].append(rmse)
            fold_metrics['rpd_values'].append(rpd)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Store fold metrics
        all_train_losses.append(fold_metrics['train_losses'])
        all_val_losses.append(fold_metrics['val_losses'])
        all_r2_scores.append(fold_metrics['r2_scores'])
        all_rmse_values.append(fold_metrics['rmse_values'])
        all_rpd_values.append(fold_metrics['rpd_values'])

    # Calculate average metrics
    avg_r2_scores = np.mean(all_r2_scores, axis=0)
    avg_rmse_values = np.mean(all_rmse_values, axis=0)
    avg_rpd_values = np.mean(all_rpd_values, axis=0)

    # Plot training results
    plot_accuracy_and_loss(
        epochs=range(1, epochs + 1),
        train_losses=all_train_losses,
        val_losses=all_val_losses,
        r2_scores=avg_r2_scores,
        rmse_values=avg_rmse_values,
        rpd_values=avg_rpd_values,
        title=f"{dataset_name} - {target_column} - {model_type}",
        target_column=target_column
    )

    return best_model, best_val_loss

def evaluate_deep_model(model, X, y, feature_columns, target_column, model_type, attention_type, dataset_name, title="Model Evaluation", plot=False):
    """Evaluate a trained deep learning model."""
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'apply_dropout'):
            model.apply_dropout(False)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        y_pred = model(X_tensor).squeeze().cpu().numpy()
        
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    rpd = np.std(y) / rmse

    # Generate visualizations if R² is good
    if plot and r2 >= VISUALIZATION_THRESHOLD:
        plot_results(y, y_pred, title, model_type, sanitize_filename(target_column))
        shap_analysis(model, X, feature_columns, sanitize_filename(target_column), model_type, attention_type, dataset_name)
        lime_analysis(model, X, y, feature_columns, sanitize_filename(target_column), model_type, attention_type, dataset_name)
        plot_spectral_curves(X, feature_columns, target_column, model_type, dataset_name)
        plot_correlation_matrix(X, feature_columns, target_column, model_type, dataset_name)
        plot_regression_diagnostics(y, y_pred, target_column, model_type, dataset_name)
        
        # Feature importance for deep learning models (approximation)
        importance_scores = np.abs(np.corrcoef(X.T, y_pred)[:-1, -1])
        plot_feature_importance(importance_scores, feature_columns, target_column, model_type, dataset_name)
        
    return r2, rmse, rpd

def train_and_evaluate_ml_model(model_type, X_train, y_train, X_test, y_test, feature_columns, target_column, dataset_name):
    """Train and evaluate a machine learning model."""
    # Initialize model
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
        model = PLSRegression(n_components=10, scale=True)
    else:
        raise ValueError(f"Unsupported ML model type: {model_type}")
    
    # Train model
    if isinstance(model, PLSRegression):
        y_train_reshaped = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
        model.fit(X_train, y_train_reshaped)
    else:
        model.fit(X_train, y_train)
    
    # Predict and flatten if needed
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    if isinstance(model, PLSRegression):
        if len(y_train_pred.shape) > 1 and y_train_pred.shape[1] == 1:
            y_train_pred = y_train_pred.flatten()
        if len(y_test_pred.shape) > 1 and y_test_pred.shape[1] == 1:
            y_test_pred = y_test_pred.flatten()
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_rpd = np.std(y_train) / train_rmse
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_rpd = np.std(y_test) / test_rmse
    
    # Generate visualizations if R² is good
    if test_r2 >= VISUALIZATION_THRESHOLD:
        plot_results(y_test, y_test_pred, 
                    f"{dataset_name} - {target_column} - {model_type}",
                    model_type, sanitize_filename(target_column))
        
        # Plot feature importance for tree-based models
        if model_type in ['RandomForest', 'GradientBoosting']:
            feature_importances = model.feature_importances_
            plot_feature_importance(feature_importances, 
                                   feature_columns, 
                                   target_column, model_type, dataset_name)
    
    return (train_r2, train_rmse, train_rpd), (test_r2, test_rmse, test_rpd), model

def objective(trial, X_train, y_train, X_test, y_test, model_type, feature_columns, target_column, dataset_name):
    """Objective function for Optuna hyperparameter optimization."""
    if model_type == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_type == 'GradientBoosting':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    elif model_type == 'SVR':
        C = trial.suggest_float('C', 1e-2, 1e2, log=True)
        epsilon = trial.suggest_float('epsilon', 1e-3, 1, log=True)
        model = SVR(kernel='rbf', C=C, epsilon=epsilon)
    elif model_type == 'ElasticNet':
        alpha = trial.suggest_float('alpha', 1e-4, 1, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    elif model_type == 'Ridge':
        alpha = trial.suggest_float('alpha', 1e-4, 1, log=True)
        model = Ridge(alpha=alpha, random_state=42)
    elif model_type == 'PLSR':
        n_components = trial.suggest_int('n_components', 2, min(30, X_train.shape[1]-1))
        model = PLSRegression(n_components=n_components, scale=True)
    elif 'DCNN' in model_type or model_type in ['ResNet18', 'VGG7']:
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 10, 200)
        patience = trial.suggest_int('patience', 10, 300)
        hyperparams = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'patience': patience
        }
        
        model, _ = train_deep_model(
            X_train, y_train,
            input_dim=X_train.shape[1],
            model_type=model_type,
            attention_type=None,
            device=DEVICE,
            dataset_name=dataset_name,
            target_column=target_column,
            hyperparams=hyperparams
        )
        
        _, test_metrics, _ = evaluate_deep_model(
            model, X_test, y_test,
            feature_columns=feature_columns,
            target_column=target_column,
            model_type=model_type,
            attention_type=None,
            dataset_name=dataset_name,
            plot=False
        )
        
        return test_metrics[1]  # Return RMSE
    else:
        raise ValueError(f"Unsupported model type for optimization: {model_type}")
    
    # For ML models, train and evaluate
    if isinstance(model, PLSRegression):
        y_train_reshaped = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
        model.fit(X_train, y_train_reshaped)
    else:
        model.fit(X_train, y_train)
        
    y_test_pred = model.predict(X_test)
    if len(y_test_pred.shape) > 1 and y_test_pred.shape[1] == 1:
        y_test_pred = y_test_pred.flatten()
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    return test_rmse

def main():
    """Main function to process all datasets and evaluate models, with protection against data leakage."""
    results = []
    train_test_pairs = {}
    
    # Step 1: Load and organize train/test dataset pairs
    for file_path, dataset_name in FILE_PATHS:
        X, y_dict, feature_columns = load_data(file_path, TARGET_COLUMNS)
        if X is None or not y_dict:
            print(f"Skipping {dataset_name}: Failed to load data")
            continue
        
        # Do NOT preprocess here - we'll do it properly after train/test separation
        
        # Organize datasets into train/test pairs
        if "Train-" in dataset_name:
            base_name = dataset_name.replace("Train-", "")
            if base_name not in train_test_pairs:
                train_test_pairs[base_name] = {"train": (X, y_dict, feature_columns, dataset_name)}
        elif "Test-" in dataset_name:
            base_name = dataset_name.replace("Test-", "")
            if base_name not in train_test_pairs:
                train_test_pairs[base_name] = {"test": (X, y_dict, feature_columns, dataset_name)}
            elif "train" in train_test_pairs[base_name]:
                train_test_pairs[base_name]["test"] = (X, y_dict, feature_columns, dataset_name)
    
    # Step 2: Process each dataset pair and evaluate all models
    for base_name, pair_data in train_test_pairs.items():
        if "train" in pair_data and "test" in pair_data:
            X_train_raw, y_dict_train, feature_columns_train, train_name = pair_data["train"]
            X_test_raw, y_dict_test, feature_columns_test, test_name = pair_data["test"]
            
            # Process each target column (e.g., SOC, TC, etc.)
            for target in TARGET_COLUMNS:
                if target in y_dict_train and target in y_dict_test:
                    print(f"\n=== Processing {base_name} for target {target} ===")
                    y_train = y_dict_train[target]
                    y_test = y_dict_test[target]
                    
                    # Preprocess training data and get scaler
                    X_train, scaler = preprocess_data(X_train_raw)
                    
                    # Apply same preprocessing to test data
                    X_test = preprocess_data(X_test_raw, scaler=scaler)
                    
                    # Apply PCA fitting only on training data
                    pca = PCA(n_components=min(50, X_train.shape[1]))
                    pca.fit(X_train)  # Fit only on training data
                    X_train_pca = pca.transform(X_train)
                    X_test_pca = pca.transform(X_test)  # Apply the same transformation to test data
                    
                    # Apply feature selection only on training data
                    X_train_selected, X_test_selected, selector = select_features(
                        X_train_pca, y_train, X_test_pca
                    )
                    
                    # Evaluate each model type
                    for model_type in MODEL_TYPES:
                        print(f"Training {model_type} on dataset {base_name}")
                        
                        if model_type in ['RandomForest', 'GradientBoosting', 'SVR', 'ElasticNet', 'Ridge', 'PLSR']:
                            # Train and evaluate machine learning models
                            train_metrics, test_metrics, _ = train_and_evaluate_ml_model(
                                model_type, X_train_selected, y_train, X_test_selected, y_test, 
                                feature_columns_test, target, base_name
                            )
                        else:
                            # Train and evaluate deep learning models
                            hyperparams = {
                                'epochs': 150,
                                'batch_size': 16,
                                'learning_rate': 2e-3,
                                'patience': 150
                            }
                            
                            # Train the model on selected features
                            model, _ = train_deep_model(
                                X_train_selected, y_train, 
                                input_dim=X_train_selected.shape[1],
                                model_type=model_type,
                                attention_type=None,
                                device=DEVICE,
                                dataset_name=base_name,
                                target_column=target,
                                hyperparams=hyperparams
                            )
                            
                            # Evaluate on train set
                            train_metrics = evaluate_deep_model(
                                model, X_train_selected, y_train, 
                                feature_columns_train, target,
                                model_type, None, base_name,
                                title=f"{base_name} - {target} - Train - {model_type}",
                                plot=False
                            )
                            
                            # Evaluate on test set
                            test_metrics = evaluate_deep_model(
                                model, X_test_selected, y_test, 
                                feature_columns_test, target,
                                model_type, None, base_name,
                                title=f"{base_name} - {target} - Test - {model_type}",
                                plot=True
                            )
                        
                        # Store results
                        results.append(
                            (base_name, target, model_type, 
                             train_metrics, test_metrics)
                        )
                        
                        # Print results
                        train_r2, train_rmse, train_rpd = train_metrics
                        test_r2, test_rmse, test_rpd = test_metrics
                        print(f"Dataset: {base_name}, Target: {target}, Model: {model_type}")
                        print(f"Train R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, RPD: {train_rpd:.4f}")
                        print(f"Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, RPD: {test_rpd:.4f}")

    # Step 3: Generate and save results summary
    print("\n=== Generating Results Summary ===")
    headers = ["Dataset", "Target", "Model", "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD"]
    table = [
        [dataset_name, target_column, model_type,
         f"{train_metrics[0]:.4f}", f"{train_metrics[1]:.4f}", f"{train_metrics[2]:.4f}", 
         f"{test_metrics[0]:.4f}", f"{test_metrics[1]:.4f}", f"{test_metrics[2]:.4f}"]
        for dataset_name, target_column, model_type, train_metrics, test_metrics in results
    ]

    print("\nResults Summary:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

    # Save results to Excel
    os.makedirs('./output', exist_ok=True)
    results_df = pd.DataFrame(table, columns=headers)
    results_df.to_excel(f'./output/results_summary.xlsx', index=False)

    # Step 4: Optional - Hyperparameter optimization with Optuna
    print("\n=== Running Hyperparameter Optimization ===")
    # Select a subset of models and targets for optimization to save time
    optimize_models = ['RandomForest', 'DCNN']
    optimize_targets = TARGET_COLUMNS[:2]  # Just use the first two targets
    
    for base_name, pair_data in list(train_test_pairs.items())[:1]:  # Just use the first dataset
        if "train" in pair_data and "test" in pair_data:
            X_train_raw, y_dict_train, feature_columns_train, _ = pair_data["train"]
            X_test_raw, y_dict_test, feature_columns_test, _ = pair_data["test"]
            
            for target in optimize_targets:
                if target in y_dict_train and target in y_dict_test:
                    y_train = y_dict_train[target]
                    y_test = y_dict_test[target]
                    
                    # Preprocess with proper train/test separation
                    X_train, scaler = preprocess_data(X_train_raw)
                    X_test = preprocess_data(X_test_raw, scaler=scaler)
                    
                    # Apply PCA with proper train/test separation
                    pca = PCA(n_components=min(50, X_train.shape[1]))
                    pca.fit(X_train)
                    X_train_pca = pca.transform(X_train)
                    X_test_pca = pca.transform(X_test)
                    
                    # Apply feature selection with proper train/test separation
                    X_train_selected, X_test_selected, selector = select_features(
                        X_train_pca, y_train, X_test_pca
                    )
                    
                    for model_type in optimize_models:
                        print(f"Optimizing {model_type} for {target} on {base_name}")
                        study = optuna.create_study(direction='minimize')
                        study.optimize(
                            lambda trial: objective(
                                trial, X_train_selected, y_train, X_test_selected, y_test,
                                model_type, feature_columns_test, target, base_name
                            ),
                            n_trials=10  # Reduced from 20 for faster execution
                        )
                        
                        print(f"Best parameters for {model_type} on {target}: {study.best_params}")
                        print(f"Best RMSE: {study.best_value:.4f}")

if __name__ == "__main__":
    main()
