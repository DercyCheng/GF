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

# Import preprocessing tools
from scipy.signal import savgol_filter
import pywt
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Import models
from models.DCNN import DCNN
from models.ResNet18 import ResNet18
from models.VGG7 import VGG7

# Import utility functions
from utils import (plot_results, shap_analysis, lime_analysis, set_seed, augment_data, load_data, preprocess_data,
                  sanitize_filename, plot_accuracy_and_loss, plot_spectral_curves, plot_correlation_matrix,
                  plot_regression_diagnostics, plot_feature_importance)

# 设置中文字体并添加备用字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'STFangsong', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

file_paths = [
    ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNSBE"),
]

target_columns = ["SOC", "EOC", "WOC", "TC", "OM"]
model_types = [
    'DCNN',
    'ResNet18', 'VGG7', 'ECA-DCNN', 'CBAM-DCNN', 'SE-DCNN',
    # Add ML models
    'RandomForest', 'GradientBoosting', 'SVR', 'ElasticNet', 'Ridge', 'PLSR'  # Add PLSR model
]

# Define preprocessing combinations
denoising_methods = ['RAW', 'SG', 'DWT', 'MSC']
math_transforms = ['NONE', 'FIRST_DERIVATIVE', 'SECOND_DERIVATIVE', 'DETREND']

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Generate spectral bands from 350 to 2500nm
spectral_bands = [f'{i}' for i in range(350, 2501)]

# Define preprocessing functions
def apply_sg(X, window_length=15, polyorder=2):
    """Apply Savitzky-Golay filter for smoothing"""
    # Ensure window_length is odd and smaller than data length
    if (window_length >= X.shape[1]):
        window_length = min(X.shape[1] - 1 if X.shape[1] % 2 == 0 else X.shape[1] - 2, 15)
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(window_length, 5)  # Minimum window length
    polyorder = min(polyorder, window_length - 1)  # polyorder must be less than window_length
    
    result = np.apply_along_axis(lambda x: savgol_filter(x, window_length, polyorder), 1, X)
    # Replace any NaN values with original values
    nan_mask = np.isnan(result)
    if np.any(nan_mask):
        result[nan_mask] = X[nan_mask]
    return result

def apply_dwt(X, wavelet='db4', level=2):
    """Apply Discrete Wavelet Transform denoising"""
    denoised = np.zeros_like(X)
    
    # Calculate maximum decomposition level based on data length
    max_level = pywt.dwt_max_level(X.shape[1], pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level)
    
    for i in range(X.shape[0]):
        try:
            # Decompose
            coeffs = pywt.wavedec(X[i], wavelet, level=level)
            # Threshold detail coefficients (keep approximation coefficients as is)
            threshold = np.std(X[i]) * np.sqrt(2 * np.log(len(X[i])))
            for j in range(1, len(coeffs)):
                coeffs[j] = pywt.threshold(coeffs[j], threshold, mode='soft')
            # Reconstruct
            denoised[i] = pywt.waverec(coeffs, wavelet)
            
            # Handle potential length mismatch
            if len(denoised[i]) != len(X[i]):
                denoised[i] = denoised[i][:len(X[i])]
        except Exception as e:
            print(f"Warning in DWT processing: {e}")
            denoised[i] = X[i]  # Use original data if error
    
    # Replace NaN values with original values
    nan_mask = np.isnan(denoised)
    if np.any(nan_mask):
        denoised[nan_mask] = X[nan_mask]
    return denoised

def apply_msc(X):
    """Apply Multiplicative Scatter Correction"""
    # Calculate mean spectrum
    mean_spectrum = np.mean(X, axis=0)
    n_samples, n_features = X.shape
    corrected = np.zeros_like(X)
    
    for i in range(n_samples):
        try:
            # Linear regression of spectrum against mean spectrum
            slope, intercept, _, _, _ = stats.linregress(mean_spectrum, X[i])
            # Apply correction
            corrected[i] = (X[i] - intercept) / slope
        except Exception as e:
            print(f"Warning in MSC processing for sample {i}: {e}")
            corrected[i] = X[i]  # Keep original if error
    
    # Replace NaN or inf values with original values
    invalid_mask = np.isnan(corrected) | np.isinf(corrected)
    if np.any(invalid_mask):
        corrected[invalid_mask] = X[invalid_mask]
    return corrected

def apply_first_derivative(X):
    """Apply first derivative"""
    # Calculate derivative
    deriv = np.diff(X, n=1, axis=1)
    
    # Pad to keep original dimensions - use edge values for padding
    padding = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        padding[i, 0] = deriv[i, 0]  # Repeat first derivative value
    
    result = np.hstack((deriv, padding))
    
    # Replace any NaN values
    nan_mask = np.isnan(result)
    if np.any(nan_mask):
        # Replace NaNs with zeros
        result[nan_mask] = 0
    return result

def apply_second_derivative(X):
    """Apply second derivative"""
    # Calculate derivative
    deriv = np.diff(X, n=2, axis=1)
    
    # Pad to keep original dimensions - use edge values for padding
    padding = np.zeros((X.shape[0], 2))
    for i in range(X.shape[0]):
        padding[i, 0] = deriv[i, 0]  # Repeat first derivative value
        padding[i, 1] = deriv[i, 0]  # Repeat first derivative value
    
    result = np.hstack((deriv, padding))
    
    # Replace any NaN values
    nan_mask = np.isnan(result)
    if np.any(nan_mask):
        # Replace NaNs with zeros
        result[nan_mask] = 0
    return result

def apply_detrend(X):
    """Remove linear trend from data"""
    detrended = np.zeros_like(X)
    for i in range(X.shape[0]):
        try:
            detrended[i] = stats.detrend(X[i])
        except Exception as e:
            print(f"Warning in detrending sample {i}: {e}")
            detrended[i] = X[i]  # Use original if error
    
    # Replace any NaN values
    nan_mask = np.isnan(detrended)
    if np.any(nan_mask):
        detrended[nan_mask] = X[nan_mask]
    return detrended

def apply_normalization(X):
    """Apply normalization to make data follow normal distribution"""
    # Handle NaN or Inf values first
    X_clean = np.copy(X)
    invalid_mask = np.isnan(X_clean) | np.isinf(X_clean)
    if np.any(invalid_mask):
        # Replace with column means
        col_means = np.nanmean(X_clean, axis=0)
        for i in range(X_clean.shape[1]):
            col_invalid = invalid_mask[:, i]
            if np.any(col_invalid):
                X_clean[col_invalid, i] = col_means[i]
    
    try:
        scaler = StandardScaler()
        result = scaler.fit_transform(X_clean)
        
        # Check for any remaining NaNs or Infs
        remaining_invalid = np.isnan(result) | np.isinf(result)
        if np.any(remaining_invalid):
            result[remaining_invalid] = 0  # Replace with zeros
    except Exception as e:
        print(f"Warning in normalization: {e}")
        # Fall back to simple normalization if StandardScaler fails
        result = (X_clean - np.nanmean(X_clean, axis=0)) / (np.nanstd(X_clean, axis=0) + 1e-10)
        # Replace any remaining NaNs or Infs
        remaining_invalid = np.isnan(result) | np.isinf(result)
        if np.any(remaining_invalid):
            result[remaining_invalid] = 0
    
    return result

def apply_preprocessing(X, denoising, transform):
    """Apply preprocessing combinations with NaN handling"""
    # Make a copy to avoid modifying original
    X = np.copy(X)
    
    # Handle NaNs in input data
    nan_mask = np.isnan(X)
    if np.any(nan_mask):
        print(f"Warning: Input data contains {np.sum(nan_mask)} NaN values. Replacing with column means.")
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            col_nan = nan_mask[:, i]
            if np.any(col_nan):
                X[col_nan, i] = col_means[i]
    
    # Step 1: Apply denoising
    if denoising == 'RAW':
        X_denoised = X.copy()  # No denoising
    elif denoising == 'SG':
        X_denoised = apply_sg(X)
    elif denoising == 'DWT':
        X_denoised = apply_dwt(X)
    elif denoising == 'MSC':
        X_denoised = apply_msc(X)
    else:
        raise ValueError(f"Unknown denoising method: {denoising}")
    
    # Check for NaNs after denoising
    nan_count = np.sum(np.isnan(X_denoised))
    if nan_count > 0:
        print(f"Warning: Denoising produced {nan_count} NaN values. Fixing...")
        nan_mask = np.isnan(X_denoised)
        X_denoised[nan_mask] = X[nan_mask]  # Replace NaNs with original values
    
    # Step 2: Apply mathematical transform
    if transform == 'NONE':
        X_transformed = X_denoised  # No transform
    elif transform == 'FIRST_DERIVATIVE':
        X_transformed = apply_first_derivative(X_denoised)
    elif transform == 'SECOND_DERIVATIVE':
        X_transformed = apply_second_derivative(X_denoised)
    elif transform == 'DETREND':
        X_transformed = apply_detrend(X_denoised)
    elif transform == 'NORMAL':
        X_transformed = apply_normalization(X_denoised)
    else:
        raise ValueError(f"Unknown transform method: {transform}")
    
    # Final check for NaNs or Infs
    invalid_mask = np.isnan(X_transformed) | np.isinf(X_transformed)
    if np.any(invalid_mask):
        print(f"Warning: Final preprocessing result contains {np.sum(invalid_mask)} invalid values. Replacing with zeros.")
        X_transformed[invalid_mask] = 0
    
    return X_transformed

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
        
        # Enable dropout during training (it should be active during training)
        if hasattr(model, 'apply_dropout'):
            model.apply_dropout(True)
            
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
            model.apply_dropout(False)  # Disable dropout for evaluation
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        y_pred = model(X_tensor).squeeze().cpu().numpy()
    
    # Calculate metrics properly
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
    # First, split the data to ensure proper training/testing comparison
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # Train the model
    model, best_val_loss = train_model(
        X_train, y_train, input_dim=input_dim,
        model_type=model_type,
        attention_type=attention_type,
        device=device,
        dataset_name=dataset_name,
        target_column=target_column,
        epochs=hyperparams['epochs'],
        batch_size=hyperparams['batch_size'],
        learning_rate=hyperparams['learning_rate'],
        patience=hyperparams['patience']
    )
    
    # Calculate metrics on training set with dropout disabled
    train_metrics = evaluate_model(
        model, X_train, y_train, feature_columns, target_column, model_type,
        attention_type, dataset_name,
        title=f"{dataset_name} - {target_column} - Train - {model_type}",
        plot=False
    )
    
    # Calculate metrics on validation set
    test_metrics = evaluate_model(
        model, X_val, y_val, feature_columns, target_column,
        model_type, attention_type, dataset_name,
        title=f"{dataset_name} - {target_column} - {model_type}",
        plot=True
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

def process_dataset(X, y_dict, feature_columns, dataset_name, device, model_types, results, denoising=None, transform=None):
    """Process dataset with specific preprocessing combination"""
    
    # Apply preprocessing if specified
    if denoising and transform:
        X_processed = apply_preprocessing(X, denoising, transform)
        preprocessing_name = f"{denoising}_{transform}"
    else:
        X_processed = X
        preprocessing_name = "ORIGINAL"
    
    for target_column, y in y_dict.items():
        print(f"Processing {target_column} from {dataset_name} with {preprocessing_name} preprocessing")
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
                    model = PLSRegression(n_components=10, scale=True)
                    
                train_metrics, test_metrics = train_and_evaluate_ml_model(
                    model, X_processed, y, feature_columns, target_column, dataset_name, model_type, plot=False
                )
            else:
                # Deep learning models
                hyperparams = {
                    'epochs': 150,
                    'batch_size': 16,
                    'learning_rate': 2e-3,
                    'patience': 150
                }
                train_metrics, test_metrics, _ = train_and_evaluate(
                    X_processed, y, input_dim=X_processed.shape[1],
                    model_type=model_type,
                    attention_type=None,
                    device=device,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    dataset_name=dataset_name,
                    hyperparams=hyperparams
                )
            
            # Store results with preprocessing info
            results.append(
                (dataset_name, target_column, model_type, preprocessing_name, 
                 train_metrics, test_metrics)
            )
            print(f"Dataset: {dataset_name}, Preprocessing: {preprocessing_name}, Target: {target_column}, Model: {model_type}, "
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

def load_data_spectral(file_path, target_columns):
    """
    Load data from Excel file, extracting only spectral bands (350-2500nm) and target columns
    
    Args:
        file_path: Path to the Excel file
        target_columns: List of target column names
        
    Returns:
        X: Array containing spectral bands data (350-2500nm only)
        y_dict: Dictionary with target column names as keys and target values as numpy arrays
        feature_columns: List of spectral band column names that were found in the data
    """
    df = pd.read_excel(file_path)
    
    # Identify spectral bands in the range 350-2500nm
    spectral_bands = [f'{i}' for i in range(350, 2501)]
    existing_bands = [band for band in spectral_bands if band in df.columns]
    
    if len(existing_bands) < len(spectral_bands):
        print(f"Warning: Found only {len(existing_bands)} out of {len(spectral_bands)} spectral bands in the data.")
    
    if not existing_bands:
        raise ValueError("No spectral bands in the range 350-2500nm found in the dataset!")
    
    # Extract features (spectral bands only)
    X = df[existing_bands].values
    
    # Extract targets
    y_dict = {}
    for target in target_columns:
        if target in df.columns:
            y_dict[target] = df[target].values
        else:
            print(f"Warning: Target column '{target}' not found in the dataset.")
    
    return X, y_dict, existing_bands

def main():
    results = []
    global device

    for file_path, dataset_name in file_paths:
        # Use the new function to load only spectral bands (350-2500nm)
        X, y_dict, feature_columns = load_data_spectral(file_path, target_columns)
        
        print(f"Loaded {len(feature_columns)} spectral bands from {dataset_name}")
        print(f"Wavelength range: {feature_columns[0]}nm to {feature_columns[-1]}nm")
        
        # Loop through all preprocessing combinations
        for denoising in denoising_methods:
            for transform in math_transforms:
                print(f"\nApplying preprocessing: {denoising} + {transform}")
                
                # Apply preprocessing only to spectral bands
                X_processed = apply_preprocessing(X, denoising, transform)
                
                # Check for NaNs before PCA
                nan_count = np.sum(np.isnan(X_processed))
                if nan_count > 0:
                    print(f"Warning: Data contains {nan_count} NaN values before PCA. Replacing with zeros.")
                    X_processed[np.isnan(X_processed)] = 0
                
                # Apply PCA for dimensionality reduction
                try:
                    pca = PCA(n_components=min(50, X_processed.shape[1]))
                    X_processed = pca.fit_transform(X_processed)
                    print(f"Applied PCA, retained {pca.n_components_} components explaining {pca.explained_variance_ratio_.sum()*100:.2f}% of variance")
                except ValueError as e:
                    print(f"PCA failed: {e}")
                    print("Using original processed data without PCA")
                    # If PCA fails, normalize the data to prevent model issues
                    X_processed = (X_processed - np.mean(X_processed, axis=0)) / (np.std(X_processed, axis=0) + 1e-10)
                    X_processed[np.isnan(X_processed) | np.isinf(X_processed)] = 0
                
                # Process dataset with this preprocessing combination
                process_dataset(
                    X_processed, y_dict, feature_columns, 
                    f"{dataset_name}_{denoising}_{transform}",  # Modified dataset name to include preprocessing
                    device, model_types, results, denoising, transform
                )

    # Modified headers to include preprocessing info
    headers = ["Dataset", "Target", "Model", "Preprocessing", "Train R²", "Train RMSE", "Train RPD", "Test R²", "Test RMSE", "Test RPD"]
    table = [
        [dataset_name, target_column, model_type, preprocessing_name,
         f"{train_metrics[0]:.4f}", f"{train_metrics[1]:.4f}", f"{train_metrics[2]:.4f}", 
         f"{test_metrics[0]:.4f}", f"{test_metrics[1]:.4f}", f"{test_metrics[2]:.4f}"]
        for dataset_name, target_column, model_type, preprocessing_name, train_metrics, test_metrics in results
    ]

    print("\nResults Summary:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

    # Save results to Excel, including preprocessing information
    results_df = pd.DataFrame(table, columns=headers)
    results_df.to_excel(f'./output/results_summary_with_preprocessing.xlsx', index=False)
    
    # Find best preprocessing combination for each model and target
    best_results = {}
    for dataset_name, target_column, model_type, preprocessing_name, _, test_metrics in results:
        key = (dataset_name.split('_')[0], target_column, model_type)
        if key not in best_results or test_metrics[0] > best_results[key][1][0]:  # Compare by R²
            best_results[key] = (preprocessing_name, test_metrics)
    
    # Create and save summary of best preprocessing methods
    best_table = [
        [dataset, target, model, preproc, f"{metrics[0]:.4f}", f"{metrics[1]:.4f}", f"{metrics[2]:.4f}"]
        for (dataset, target, model), (preproc, metrics) in best_results.items()
    ]
    best_headers = ["Dataset", "Target", "Model", "Best Preprocessing", "R²", "RMSE", "RPD"]
    best_df = pd.DataFrame(best_table, columns=best_headers)
    best_df.to_excel('./output/best_preprocessing_results.xlsx', index=False)
    
    print("\nBest Preprocessing Results:")
    print(tabulate(best_table, headers=best_headers, tablefmt="grid"))

    # Optional: Run Optuna optimization on the best preprocessing combination
    study = optuna.create_study(direction='minimize')
    best_preprocessing = list(best_results.values())[0][0]  # Get first preprocessing method
    denoising, transform = best_preprocessing.split('_')
    X_best = apply_preprocessing(X, denoising, transform)
    
    for model_type in model_types:
        for target_column, y in y_dict.items():
            study.optimize(lambda trial: objective(trial, X_best, y, model_type, feature_columns, target_column, dataset_name), n_trials=20)
    
    best_params = study.best_params
    print("Best hyperparameters found by Optuna:", best_params)

if __name__ == "__main__":
    main()
