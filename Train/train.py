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

# Add imports for traditional ML models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor

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
    # ("../datasets/data_spectral_bands_sgd_dr.xlsx", "SBSD"),
    # ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
    # ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNSBE"),
    # ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD"),
    # ("../datasets/data_soil_nutrients_spectral_bands_sae.xlsx", "SNSBSAE"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment_sae.xlsx", "SNSBESAE"),
    # ("../datasets/data_spectral_bands_sae.xlsx", "SBSAE")
]
target_columns = ["SOC", "EOC", "WOC", "TC", "OM"]
model_types = [
    'DCNN', 'SE-DCNN', 'ECA-DCNN', 'CBAM-DCNN',
    'ResNet18', 'VGG7',
    # Add ML models for comparison
    'RandomForest', 'SVR', 'XGBoost', 'LinearRegression', 'Ridge', 'Lasso', 'GradientBoosting'
]

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def initialize_model(model_type, input_dim, attention_type=None):
    # Handle traditional ML models
    ml_models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    if model_type in ml_models:
        return ml_models[model_type]
    
    # Parse attention type from model name if it contains a hyphen
    if '-' in model_type:
        attention_type, base_model = model_type.split('-')
    else:
        base_model = model_type
        attention_type = None

    # Handle deep learning models
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

# Add a function to train ML models
def train_ml_model(model, X_train, y_train):
    # No need for epochs, batches, etc. - just fit the model
    model.fit(X_train, y_train)
    return model

def prepare_dataset(X_train, y_train, X_val, y_val, model_type):
    if 'MultiModal' in model_type:
        # Split X into spectral, environmental, and nutrient features
        spectral_features = X_train[:, :2150]  # Adjust indices based on your data
        env_features = X_train[:, 2150:2161]   # Adjust indices based on your data
        nutrient_features = X_train[:, 2161:]  # Adjust indices based on your data
        
        train_dataset = TensorDataset(
            torch.tensor(spectral_features, dtype=torch.float32).unsqueeze(1),
            torch.tensor(env_features, dtype=torch.float32),
            torch.tensor(nutrient_features, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        # Similar for validation set
        val_spectral_features = X_val[:, :2150]  # Adjust indices based on your data
        val_env_features = X_val[:, 2150:2161]   # Adjust indices based on your data
        val_nutrient_features = X_val[:, 2161:]  # Adjust indices based on your data
        
        val_dataset = TensorDataset(
            torch.tensor(val_spectral_features, dtype=torch.float32).unsqueeze(1),
            torch.tensor(val_env_features, dtype=torch.float32),
            torch.tensor(val_nutrient_features, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
    else:
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
    
    # Check if it's a machine learning model
    is_ml_model = model_type in ['RandomForest', 'SVR', 'XGBoost', 'LinearRegression', 'Ridge', 'Lasso', 'GradientBoosting']

    all_train_losses = []
    all_val_losses = []
    all_r2_scores = []
    all_rmse_values = []
    all_rpd_values = []

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/5")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # For ML models, we don't need to augment data the same way
        if not is_ml_model:
            # Enhance data augmentation
            X_train, y_train = augment_data(X_train, y_train)  # Ensure augment_data provides sufficient augmentation

        # Initialize the model
        model = initialize_model(model_type, input_dim, attention_type)
        
        # For ML models, we train differently
        if is_ml_model:
            model = train_ml_model(model, X_train, y_train)
            # Calculate metrics
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_loss = mean_squared_error(y_train, train_pred)
            val_loss = mean_squared_error(y_val, val_pred)
            
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            rmse = np.sqrt(val_loss)
            rpd = np.std(y_val) / rmse
            
            # Store just one epoch worth of metrics for ML models
            all_train_losses.append([train_loss])
            all_val_losses.append([val_loss])
            all_r2_scores.append([val_r2])
            all_rmse_values.append([rmse])
            all_rpd_values.append([rpd])
            
            # Update best model if needed
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
        else:
            # Move the model to the correct device
            model = model.to(device)
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
                scheduler.step(val_loss)

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

    # For deep learning models, compute average metrics
    if not is_ml_model:
        # Compute average metrics per epoch across all folds
        avg_r2_scores = np.mean(all_r2_scores, axis=0)
        avg_rmse_values = np.mean(all_rmse_values, axis=0)
        avg_rpd_values = np.mean(all_rpd_values, axis=0)

        # After training
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

def evaluate_model(model, X, y, feature_columns, target_column, model_type, attention_type, dataset_name,
                   title="模型评估", plot=False):
    # Check if it's a machine learning model
    is_ml_model = model_type in ['RandomForest', 'SVR', 'XGBoost', 'LinearRegression', 'Ridge', 'Lasso', 'GradientBoosting']
    
    if is_ml_model:
        # For ML models, just use predict method directly on numpy arrays
        y_pred = model.predict(X)
    else:
        # For deep learning models, use the existing code with tensors
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
            y_pred = model(X_tensor).squeeze().cpu().numpy()
            
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    rpd = np.std(y) / rmse

    # Only plot for deep learning models with good performance
    if plot and 0.85 <= r2 < 0.99 and not is_ml_model:
        plot_results(y, y_pred, title, model_type, sanitize_filename(target_column))
        shap_analysis(model, X, feature_columns, sanitize_filename(target_column), model_type, attention_type, dataset_name)
        lime_analysis(model, X, y, feature_columns, sanitize_filename(target_column), model_type, attention_type, dataset_name)
        
    return r2, rmse, rpd

def train_and_evaluate(X, y, input_dim, model_type, attention_type, device, feature_columns, target_column, dataset_name, hyperparams):
    # Check if it's a machine learning model
    is_ml_model = model_type in ['RandomForest', 'SVR', 'XGBoost', 'LinearRegression', 'Ridge', 'Lasso', 'GradientBoosting']
    
    if is_ml_model:
        # For ML models, create and train directly
        model = initialize_model(model_type, input_dim, attention_type)
        
        # Apply hyperparameters if available
        if model_type == 'RandomForest' and 'n_estimators' in hyperparams:
            model.n_estimators = hyperparams.get('n_estimators', 100)
            model.max_depth = hyperparams.get('max_depth', None)
        elif model_type == 'SVR' and 'C' in hyperparams:
            model.C = hyperparams.get('C', 1.0)
            model.epsilon = hyperparams.get('epsilon', 0.1)
        elif model_type == 'XGBoost' and 'n_estimators' in hyperparams:
            model.n_estimators = hyperparams.get('n_estimators', 100)
            model.learning_rate = hyperparams.get('learning_rate', 0.1)
            
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)
        model.fit(X_train, y_train)
        best_val_loss = mean_squared_error(y_val, model.predict(X_val))
    else:
        # For DL models, use the existing training function
        model, best_val_loss = train_model(
            X, y, input_dim=input_dim,
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
        
    # Split data for evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # Evaluate on test data
    test_metrics = evaluate_model(
        model, X_val, y_val, feature_columns, target_column,
        model_type, attention_type, dataset_name,
        title=f"{dataset_name} - {target_column} - {attention_type} - {model_type}" if attention_type else f"{dataset_name} - {target_column} - {model_type}",
        plot=True
    )
    
    # Evaluate on training data
    train_metrics = evaluate_model(
        model, X_train, y_train, feature_columns, target_column, model_type,
        attention_type, dataset_name,
        title=f"{dataset_name} - {target_column} - Train - {attention_type} - {model_type}" if attention_type else f"{dataset_name} - {target_column} - Train - {model_type}",
        plot=False
    )
    
    return train_metrics, test_metrics, best_val_loss

def process_dataset(X, y_dict, feature_columns, dataset_name, device, model_types, results):
    for target_column, y in y_dict.items():
        print(f"Processing {target_column} from {dataset_name}")
        for model_type in model_types:
            print(f"Training {model_type}")
            hyperparams = {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 1e-3,
                'patience': 100
            }
            train_metrics, test_metrics, _ = train_and_evaluate(
                X, y, input_dim=X.shape[1],
                model_type=model_type,
                attention_type=None,  # This is now ignored as attention type is parsed from model_type
                device=device,
                feature_columns=feature_columns,
                target_column=target_column,
                dataset_name=dataset_name,
                hyperparams=hyperparams
            )
            results.append(
                (dataset_name, target_column, model_type, train_metrics, test_metrics)
            )
            print(f"Dataset: {dataset_name}, Target: {target_column}, Model: {model_type}, Train R²: {train_metrics[0]}, Train Loss: {train_metrics[1]}, Val R²: {test_metrics[0]}, Val Loss: {test_metrics[1]}")

def objective(trial, X, y, model_type, feature_columns, target_column, dataset_name):
    # Different hyperparameters for ML and DL models
    is_ml_model = model_type in ['RandomForest', 'SVR', 'XGBoost', 'LinearRegression', 'Ridge', 'Lasso', 'GradientBoosting']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)
    
    if is_ml_model:
        # Define hyperparameters for ML models
        if model_type == 'RandomForest':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            max_depth = trial.suggest_int('max_depth', 5, 50, log=True)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
        elif model_type == 'SVR':
            C = trial.suggest_float('C', 0.1, 10.0, log=True)
            epsilon = trial.suggest_float('epsilon', 0.01, 1.0)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            model = SVR(C=C, epsilon=epsilon, gamma=gamma)
        elif model_type == 'XGBoost':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            model = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
        elif model_type == 'Ridge':
            alpha = trial.suggest_float('alpha', 0.01, 10.0, log=True)
            model = Ridge(alpha=alpha)
        elif model_type == 'Lasso':
            alpha = trial.suggest_float('alpha', 0.001, 1.0, log=True)
            model = Lasso(alpha=alpha)
        elif model_type == 'GradientBoosting':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
        else:  # LinearRegression
            model = LinearRegression()
            
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        val_loss = mean_squared_error(y_val, y_pred)
        return val_loss
    else:
        # Original hyperparameters for DL models
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
        
        # Train model with the hyperparameters
        model, val_loss = train_model(
            X_train, y_train, 
            input_dim=X.shape[1],
            model_type=model_type,
            attention_type=None,
            device=device,
            dataset_name=dataset_name,
            target_column=target_column,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience
        )
        
        return val_loss  # Return the validation loss from training

def main():
    results = []
    global device
    
    for file_path, dataset_name in file_paths:
        X, y_dict, feature_columns = load_data(file_path, target_columns)
        X = preprocess_data(X)
        
        # Optionally apply dimensionality reduction for better ML performance
        pca = PCA(n_components=min(50, X.shape[1]))
        X_reduced = pca.fit_transform(X)
        
        for target_column, y in y_dict.items():
            print(f"Processing {target_column} from {dataset_name}")
            
            # Run hyperparameter optimization for each model type and target
            best_hyperparams = {}
            for model_type in model_types:
                print(f"Optimizing hyperparameters for {model_type} on {target_column}")
                study = optuna.create_study(direction='minimize')
                study.optimize(
                    lambda trial: objective(trial, X_reduced, y, model_type, feature_columns, target_column, dataset_name), 
                    n_trials=10
                )
                best_hyperparams[model_type] = study.best_params
                print(f"Best params for {model_type}: {study.best_params}")
            
            # Train models with optimized hyperparameters
            for model_type in model_types:
                print(f"Training {model_type} with optimized hyperparameters")
                
                # Get the best hyperparameters or use defaults
                hyperparams = best_hyperparams.get(model_type, {})
                if model_type not in ['RandomForest', 'SVR', 'XGBoost', 'LinearRegression', 'Ridge', 'Lasso', 'GradientBoosting']:
                    # Set default DL hyperparams if not present
                    if 'epochs' not in hyperparams:
                        hyperparams.update({
                            'epochs': 100,
                            'batch_size': 32,
                            'learning_rate': 1e-3,
                            'patience': 100
                        })
                
                # Train and evaluate with best hyperparameters
                train_metrics, test_metrics, _ = train_and_evaluate(
                    X_reduced, y, input_dim=X_reduced.shape[1],
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
                
                print(f"Dataset: {dataset_name}, Target: {target_column}, Model: {model_type}")
                print(f"Train - R²: {train_metrics[0]:.4f}, RMSE: {train_metrics[1]:.4f}, RPD: {train_metrics[2]:.4f}")
                print(f"Test - R²: {test_metrics[0]:.4f}, RMSE: {test_metrics[1]:.4f}, RPD: {test_metrics[2]:.4f}")

    # Output results to table
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
    results_df = pd.DataFrame(table, columns=headers)
    results_df.to_excel(f'./output/results_summary.xlsx', index=False)

    # Save the best model
    best_row = max(table, key=lambda x: float(x[6]))  # Find row with highest Test R²
    best_dataset = best_row[0]
    best_target = best_row[1]
    best_model_type = best_row[2]
    
    print(f"\nBest model: {best_model_type} for {best_target} on {best_dataset}")
    
    # We could retrain and save the best model here if needed
    # For now we'll just note that the models are evaluated during the main loop

if __name__ == "__main__":
    main()
