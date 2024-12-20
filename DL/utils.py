import os
import random
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import lime
import lime.lime_tabular
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def sanitize_filename(filename):
    # 去除文件名中的非法字符
    return re.sub(r'[\/:*?"<>|]', '-', filename)

def plot_results(y_true, y_pred, title, model_type, target_column):
    save_title = sanitize_filename(title)
    save_path = f"output/{sanitize_filename(target_column)}/scatter/{save_title}_scatter.png"
    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, label='Predicted vs Actual', alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='1:1 Line')

    # 拟合线
    m, b = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, m * y_true + b, 'b-', label=f'Fit Line: y={float(m):.2f}x+{float(b):.2f}')

    # 计算 R², RMSE, RPD
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # 转换为 g·kg⁻¹
    rpd = np.std(y_true) / rmse

    # 左上角显示 R², RMSE, RPD
    plt.text(0.05, 0.95, f'R^2: {r2:.4f}\nRMSE: {rmse:.4f} g·kg^-1\nRPD: {rpd:.4f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)  # 保存图像
    plt.close()

def shap_analysis(model, X, feature_names, target_column, model_type, attention_type, dataset_name):
    save_title = sanitize_filename(f"SHAP_{dataset_name}_{target_column}_{attention_type}_{model_type}") if attention_type else sanitize_filename(f"SHAP_{dataset_name}_{target_column}_{model_type}")
    save_path = f"output/{sanitize_filename(target_column)}/shap/{save_title}.png"
    ensure_dir(os.path.dirname(save_path))
    model.eval()
    explainer = shap.GradientExplainer(model, torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(model.device))
    shap_values = explainer.shap_values(torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(model.device))

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim > 2:
        shap_values = shap_values.squeeze()
    if X.shape[1] != len(feature_names):
        feature_names = feature_names[:X.shape[1]]
    X_df = pd.DataFrame(X, columns=feature_names)

    shap.summary_plot(shap_values, X_df, show=False)
    plt.title(f"SHAP - {target_column} ({attention_type}-{model_type}) - {dataset_name}" if attention_type else f"SHAP - {target_column} ({model_type}) - {dataset_name}")
    plt.savefig(save_path, bbox_inches='tight')  # 保存图像
    plt.close()

def lime_analysis(model, X, y, feature_names, target_column, model_type, attention_type, dataset_name):
    save_title = sanitize_filename(f"LIME_{dataset_name}_{target_column}_{attention_type}_{model_type}") if attention_type else sanitize_filename(f"LIME_{dataset_name}_{target_column}_{model_type}")
    save_path = f"output/{sanitize_filename(target_column)}/lime/{save_title}.png"
    ensure_dir(os.path.dirname(save_path))
    model.eval()
    with torch.no_grad():
        # 定义预测函数
        def predict_fn(data):
            data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1).to(model.device)
            return model(data_tensor).cpu().detach().numpy().flatten()
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X, mode='regression', feature_names=feature_names, verbose=True, feature_selection='auto'
    )
    i = np.random.randint(0, X.shape[0])
    exp = explainer.explain_instance(
        X[i], predict_fn, num_features=10
    )
    # 生成并显示解释结果的图形
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(8, 4)  # 调整图形比例
    plt.title(f"LIME - {target_column} ({attention_type}-{model_type}) - {dataset_name}" if attention_type else f"LIME - {target_column} ({model_type}) - {dataset_name}")
    plt.tight_layout()  # 确保左侧列名显示完全
    plt.savefig(save_path)  # 保存图像
    # plt.show()
    plt.close()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def augment_data(X, y):
    # 通过多种增广操作，数据最终扩容为原来的5倍
    # Example data augmentation: adding Gaussian noise
    noise = np.random.normal(0, 0.01, X.shape)
    X_augmented = X + noise

    # Additional augmentation: scaling
    scale = np.random.uniform(0.9, 1.1, X.shape)
    X_scaled = X * scale

    # Additional augmentation: flipping
    X_flipped = np.flip(X, axis=1)

    # Additional augmentation: shifting
    shift = np.roll(X, shift=1, axis=1)

    X_augmented = np.concatenate([X, X_augmented, X_scaled, X_flipped, shift])
    y_augmented = np.concatenate([y, y, y, y, y])

    return X_augmented, y_augmented

def load_data(file_path, target_columns):
    # 数据集文件格式为 xlsx
    try:
        data = pd.read_excel(file_path)
        print(f"Data loaded successfully from {file_path}!")
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        exit()

    data.columns = data.columns.map(str)
    data = data.dropna(subset=target_columns)
    y_dict = {target_column: data[target_column].values for target_column in target_columns}
    data = data.drop(columns=target_columns)
    feature_columns = data.columns.tolist()
    x = data.select_dtypes(include=[np.number]).values

    print(f"Number of features: {len(feature_columns)}, Number of samples: {x.shape[0]}")
    return x, y_dict, feature_columns

def preprocess_data(X):
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    return scaler.fit_transform(X)