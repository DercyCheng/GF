import numpy as np
import pandas as pd
import pywt
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# 读取Excel文件
data = pd.read_excel('data.xlsx')

# 定义列名及其别名
soil_nutrients = {
    'PH': 'PH',
    '有机质(g/kg)': 'OM',
    '全氮(g/kg)': 'TN',
    '全磷(g/kg)': 'TP',
    '全钾(g/kg)': 'TK',
    '速效N(mg/kg)': 'AN',
    '速效p(mg/kg)': 'AP',
    '速效k(mg/kg)': 'AK',
    'B(mg/kg)': 'B',
    'Cu(mg/kg)': 'Cu',
    'Zn(mg/kg)': 'Zn',
    'Fe(mg/kg)': 'Fe',
    'Ca(mg/kg)': 'Ca',
    'Mg(mg/kg)': 'Mg',
    '全碳(g/kg)': 'TC',
    '易氧化有机碳(mg/g)': 'EOC',
    '有机碳含量(g/kg)': 'SOC',
    '水溶性有机碳(mg/g)': 'WOC'
}
spectral_bands = [f'{i}' for i in range(400, 2501)]  # Changed back to 2500 to match data dimensions
environment_info = {
    '海拔测量': 'ELEV',
    'Longitude': 'LONG',
    'latitude': 'LAT',
    '坡度': 'SLOPE',
    '坡向': 'ASPECT',
    '海拔': 'ALT',
    '大于10度积温': 'GDD10',
    '年均降雨': 'AN_RAIN',
    '年均温度': 'AN_TEMP',
    '代数': 'GEN',
    '林龄': 'FOREST_AGE'
}
target_columns = {
    '有机质(g/kg)': 'OM',
    '全碳(g/kg)': 'TC',
    '易氧化有机碳(mg/g)': 'EOC',
    '有机碳含量(g/kg)': 'SOC',
    '水溶性有机碳(mg/g)': 'WOC'
}

# 将光谱波段列名转换为字符串类型并添加单位
data.columns = data.columns.map(lambda x: f'{x}' if isinstance(x, int) else x)

# 使用别名替换列名
data.rename(columns={**soil_nutrients, **environment_info}, inplace=True)

# 分割数据集为训练集和测试集（80% 训练，20% 测试）
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# KNN缺失值插补函数 - 分别对训练集和测试集进行处理，避免数据泄漏
def impute_missing_values(train_df, test_df, n_neighbors=5):
    """
    使用KNN方法填补缺失值，确保不会发生数据泄漏
    
    参数:
    train_df: 训练数据集
    test_df: 测试数据集
    n_neighbors: KNN算法的邻居数量
    
    返回:
    train_imputed: 填补后的训练数据
    test_imputed: 填补后的测试数据
    """
    print(f"处理缺失值前 - 训练集中的NaN值数量: {train_df.isna().sum().sum()}")
    print(f"处理缺失值前 - 测试集中的NaN值数量: {test_df.isna().sum().sum()}")
    
    # 创建KNN插补器
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # 获取所有数值型列（包括光谱波段和其他数值型特征）
    numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # 对数值型列进行标准化处理
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df[numeric_cols]), 
        columns=numeric_cols,
        index=train_df.index
    )
    # 使用训练集的均值和标准差来标准化测试集
    test_scaled = pd.DataFrame(
        scaler.transform(test_df[numeric_cols]),
        columns=numeric_cols,
        index=test_df.index
    )
    
    # 分别对训练集和测试集进行KNN插补
    train_imputed_values = imputer.fit_transform(train_scaled)
    # 只使用fit_transform后的imputer对测试集进行transform，避免使用测试集数据训练imputer
    test_imputed_values = imputer.transform(test_scaled)
    
    # 将标准化的数据转换回原始尺度
    train_imputed_values = scaler.inverse_transform(train_imputed_values)
    test_imputed_values = scaler.inverse_transform(test_imputed_values)
    
    # 创建填补后的数据框
    train_imputed = train_df.copy()
    test_imputed = test_df.copy()
    
    # 更新填补后的值
    train_imputed[numeric_cols] = train_imputed_values
    test_imputed[numeric_cols] = test_imputed_values
    
    print(f"处理缺失值后 - 训练集中的NaN值数量: {train_imputed.isna().sum().sum()}")
    print(f"处理缺失值后 - 测试集中的NaN值数量: {test_imputed.isna().sum().sum()}")
    
    return train_imputed, test_imputed

# 应用KNN插补处理缺失值
train_data, test_data = impute_missing_values(train_data, test_data, n_neighbors=5)

# 数据预处理函数

# Function to apply Savitzky-Golay filter and first derivative
def apply_sgd(train_data, test_data, bands, window_length=5, polyorder=2):
    # Process training data
    train_sgd_filtered = savgol_filter(train_data[bands], window_length=window_length, polyorder=polyorder, deriv=0,
                                       axis=0)
    train_sgd_derivative = savgol_filter(train_sgd_filtered, window_length=window_length, polyorder=polyorder, deriv=1,
                                         axis=0)

    # Process test data with same parameters
    test_sgd_filtered = savgol_filter(test_data[bands], window_length=window_length, polyorder=polyorder, deriv=0,
                                      axis=0)
    test_sgd_derivative = savgol_filter(test_sgd_filtered, window_length=window_length, polyorder=polyorder, deriv=1,
                                        axis=0)

    return pd.DataFrame(train_sgd_derivative, columns=bands), pd.DataFrame(test_sgd_derivative, columns=bands)


# Function to apply Standard Normal Variate (SNV) normalization
def apply_snv(train_data, test_data, bands):
    # Calculate mean and std from training data
    train_mean = train_data[bands].mean(axis=0)
    train_std = train_data[bands].std(axis=0)

    # Apply to training data
    train_snv = (train_data[bands] - train_mean) / train_std

    # Apply same transformation to test data (using training parameters)
    test_snv = (test_data[bands] - train_mean) / train_std

    return train_snv, test_snv


# Function to apply Multiplicative Scatter Correction (MSC)
def apply_msc(train_data, test_data, bands):
    # Calculate mean spectrum from training data
    mean_spectrum = np.mean(train_data[bands], axis=0)

    # Apply MSC to training data
    train_msc = np.zeros_like(train_data[bands])
    for i in range(len(train_data)):
        fit = np.polyfit(mean_spectrum, train_data[bands].iloc[i], 1)
        train_msc[i] = (train_data[bands].iloc[i] - fit[1]) / fit[0]

    # Apply MSC to test data using same reference spectrum
    test_msc = np.zeros_like(test_data[bands])
    for i in range(len(test_data)):
        fit = np.polyfit(mean_spectrum, test_data[bands].iloc[i], 1)
        test_msc[i] = (test_data[bands].iloc[i] - fit[1]) / fit[0]

    return pd.DataFrame(train_msc, columns=bands), pd.DataFrame(test_msc, columns=bands)


# Function to apply Discrete Wavelet Transform (DWT)
def apply_dwt(train_data, test_data, bands, wavelet='db4', level=3):
    # Apply DWT to training data
    train_dwt = np.zeros((len(train_data), len(bands)))
    for i in range(len(train_data)):
        coeffs = pywt.wavedec(train_data[bands].iloc[i], wavelet, level=level)
        rec = pywt.waverec(coeffs, wavelet)
        if len(rec) > len(bands):
            rec = rec[:len(bands)]
        elif len(rec) < len(bands):
            rec = np.pad(rec, (0, len(bands) - len(rec)))
        train_dwt[i] = rec

    # Apply DWT to test data
    test_dwt = np.zeros((len(test_data), len(bands)))
    for i in range(len(test_data)):
        coeffs = pywt.wavedec(test_data[bands].iloc[i], wavelet, level=level)
        rec = pywt.waverec(coeffs, wavelet)
        if len(rec) > len(bands):
            rec = rec[:len(bands)]
        elif len(rec) < len(bands):
            rec = np.pad(rec, (0, len(bands) - len(rec)))
        test_dwt[i] = rec

    return pd.DataFrame(train_dwt, columns=bands), pd.DataFrame(test_dwt, columns=bands)


# Function to apply Competitive Adaptive Reweighted Sampling (CARS)
def apply_cars(X, y, num_features=200, num_splits=50, max_elim_rate=0.3):
    """
    Apply Competitive Adaptive Reweighted Sampling (CARS) to select features

    Parameters:
    X: DataFrame of features (spectral bands)
    y: Series of target variable
    num_features: Target number of features to select
    num_splits: Number of sampling runs
    max_elim_rate: Maximum elimination rate in exponential function

    Returns:
    selected_features: List of selected feature indices
    """
    n_samples, n_features = X.shape

    # Calculate weights based on absolute correlation coefficients
    weights = np.zeros(n_features)
    for i in range(n_features):
        weights[i] = abs(np.corrcoef(X.iloc[:, i], y)[0, 1])

    # Normalize weights
    weights = weights / np.sum(weights)

    # Create elimination ratio for each split
    elim_ratio = np.exp(np.linspace(0, max_elim_rate, num_splits))

    # Number of features to keep in each split
    n_keep = np.round(n_features * (1 - elim_ratio)).astype(int)
    n_keep[-1] = min(num_features, n_keep[-1])  # Ensure we don't select more than target

    # Select features based on weights
    selected_indices = np.argsort(-weights)[:n_keep[-1]]

    return list(selected_indices)


# 创建各种处理方法的数据集

# 数据集1：土壤养分含量+光谱波段
train_dataset1 = train_data[list(soil_nutrients.values()) + spectral_bands]
test_dataset1 = test_data[list(soil_nutrients.values()) + spectral_bands]

# 数据集2：土壤养分含量+光谱波段+环境信息
train_dataset2 = train_data[list(soil_nutrients.values()) + spectral_bands + list(environment_info.values())]
test_dataset2 = test_data[list(soil_nutrients.values()) + spectral_bands + list(environment_info.values())]

# 数据集3：土壤养分含量+光谱波段，进行SGD降噪以及DR一阶微分
train_sgd, test_sgd = apply_sgd(train_data, test_data, spectral_bands)
train_dataset3 = pd.concat([train_data[list(soil_nutrients.values())], train_sgd], axis=1)
test_dataset3 = pd.concat([test_data[list(soil_nutrients.values())], test_sgd], axis=1)

# 数据集4：土壤养分含量+光谱波段+环境信息，进行SGD降噪以及一阶微分
train_dataset4 = pd.concat(
    [train_data[list(soil_nutrients.values())], train_sgd, train_data[list(environment_info.values())]], axis=1)
test_dataset4 = pd.concat(
    [test_data[list(soil_nutrients.values())], test_sgd, test_data[list(environment_info.values())]], axis=1)

# 数据集5：target_columns + 经过SGD+DR处理的光谱波段
train_dataset5 = pd.concat([train_data[list(target_columns.values())], train_sgd], axis=1)
test_dataset5 = pd.concat([test_data[list(target_columns.values())], test_sgd], axis=1)

# 数据集6：土壤养分含量+光谱波段，进行snv处理
train_snv, test_snv = apply_snv(train_data, test_data, spectral_bands)
train_dataset6 = pd.concat([train_data[list(soil_nutrients.values())], train_snv], axis=1)
test_dataset6 = pd.concat([test_data[list(soil_nutrients.values())], test_snv], axis=1)

# 数据集7：土壤养分含量+光谱波段+环境信息，进行snv处理
train_dataset7 = pd.concat(
    [train_data[list(soil_nutrients.values())], train_snv, train_data[list(environment_info.values())]], axis=1)
test_dataset7 = pd.concat(
    [test_data[list(soil_nutrients.values())], test_snv, test_data[list(environment_info.values())]], axis=1)

# 数据集8：target_columns + 经过snv处理的光谱波段
train_dataset8 = pd.concat([train_data[list(target_columns.values())], train_snv], axis=1)
test_dataset8 = pd.concat([test_data[list(target_columns.values())], test_snv], axis=1)

# 数据集9：土壤养分含量+光谱波段，进行MSC处理
train_msc, test_msc = apply_msc(train_data, test_data, spectral_bands)
train_dataset9 = pd.concat([train_data[list(soil_nutrients.values())], train_msc], axis=1)
test_dataset9 = pd.concat([test_data[list(soil_nutrients.values())], test_msc], axis=1)

# 数据集10：土壤养分含量+光谱波段+环境信息，进行MSC处理
train_dataset10 = pd.concat(
    [train_data[list(soil_nutrients.values())], train_msc, train_data[list(environment_info.values())]], axis=1)
test_dataset10 = pd.concat(
    [test_data[list(soil_nutrients.values())], test_msc, test_data[list(environment_info.values())]], axis=1)

# 数据集11：target_columns + 经过MSC处理的光谱波段
train_dataset11 = pd.concat([train_data[list(target_columns.values())], train_msc], axis=1)
test_dataset11 = pd.concat([test_data[list(target_columns.values())], test_msc], axis=1)

# 数据集12：土壤养分含量+光谱波段，进行DWT处理
train_dwt, test_dwt = apply_dwt(train_data, test_data, spectral_bands)
train_dataset12 = pd.concat([train_data[list(soil_nutrients.values())], train_dwt], axis=1)
test_dataset12 = pd.concat([test_data[list(soil_nutrients.values())], test_dwt], axis=1)

# 数据集13：土壤养分含量+光谱波段+环境信息，进行DWT处理
train_dataset13 = pd.concat(
    [train_data[list(soil_nutrients.values())], train_dwt, train_data[list(environment_info.values())]], axis=1)
test_dataset13 = pd.concat(
    [test_data[list(soil_nutrients.values())], test_dwt, test_data[list(environment_info.values())]], axis=1)

# 数据集14：target_columns + 经过DWT处理的光谱波段
train_dataset14 = pd.concat([train_data[list(target_columns.values())], train_dwt], axis=1)
test_dataset14 = pd.concat([test_data[list(target_columns.values())], test_dwt], axis=1)

# 数据集15：土壤养分含量+光谱波段，进行MSC+DWT处理
train_msc_dwt, test_msc_dwt = apply_dwt(pd.concat([train_msc], axis=1), pd.concat([test_msc], axis=1), spectral_bands)
train_dataset15 = pd.concat([train_data[list(soil_nutrients.values())], train_msc_dwt], axis=1)
test_dataset15 = pd.concat([test_data[list(soil_nutrients.values())], test_msc_dwt], axis=1)

# 数据集16：土壤养分含量+光谱波段+环境信息，进行MSC+DWT处理
train_dataset16 = pd.concat(
    [train_data[list(soil_nutrients.values())], train_msc_dwt, train_data[list(environment_info.values())]], axis=1)
test_dataset16 = pd.concat(
    [test_data[list(soil_nutrients.values())], test_msc_dwt, test_data[list(environment_info.values())]], axis=1)

# Apply CARS feature selection to SGD processed data
# Using all target variables from target_columns for CARS feature selection
selected_features_by_target = {}
for target_name, target_col in target_columns.items():
    selected_indices = apply_cars(train_sgd, train_data[target_col], num_features=200)
    selected_features = [spectral_bands[i] for i in selected_indices]
    selected_features_by_target[target_col] = selected_features

# Create combined list of unique selected features across all targets
all_selected_features = list(set(feature for features in selected_features_by_target.values() for feature in features))

# Create subset of spectral bands with CARS selection
train_sgd_cars = train_sgd[all_selected_features]
test_sgd_cars = test_sgd[all_selected_features]

# Also create target-specific datasets
target_specific_datasets = {}
for target_name, target_col in target_columns.items():
    target_specific_datasets[target_col] = {
        'train': train_sgd[selected_features_by_target[target_col]],
        'test': test_sgd[selected_features_by_target[target_col]]
    }

# Create new datasets with CARS feature selection

# Dataset 17: target_columns + CARS selected spectral bands after SGD+DR
train_dataset17 = pd.concat([train_data[list(target_columns.values())], train_sgd_cars], axis=1)
test_dataset17 = pd.concat([test_data[list(target_columns.values())], test_sgd_cars], axis=1)

# Dataset 18: soil_nutrients + CARS selected spectral bands after SGD+DR
train_dataset18 = pd.concat([train_data[list(soil_nutrients.values())], train_sgd_cars], axis=1)
test_dataset18 = pd.concat([test_data[list(soil_nutrients.values())], test_sgd_cars], axis=1)

# Dataset 19: soil_nutrients + CARS selected spectral bands after SGD+DR + environment_info
train_dataset19 = pd.concat(
    [train_data[list(soil_nutrients.values())], train_sgd_cars, train_data[list(environment_info.values())]], axis=1)
test_dataset19 = pd.concat(
    [test_data[list(soil_nutrients.values())], test_sgd_cars, test_data[list(environment_info.values())]], axis=1)

# Also create target-specific CARS datasets
for idx, (target_name, target_col) in enumerate(target_columns.items(), start=20):
    # Create dataset: target column + its specific CARS selected features
    globals()[f'train_dataset{idx}'] = pd.concat(
        [train_data[[target_col]], target_specific_datasets[target_col]['train']], axis=1)
    globals()[f'test_dataset{idx}'] = pd.concat(
        [test_data[[target_col]], target_specific_datasets[target_col]['test']], axis=1)

# 创建保存目录（如果不存在）
import os

os.makedirs("../datasets/train", exist_ok=True)
os.makedirs("../datasets/test", exist_ok=True)


# 计算target_columns的统计特征
def calculate_statistics(data, columns, sample_type):
    stats = []
    for col in columns:
        values = data[col].values
        sample_size = len(values)
        min_val = np.min(values)
        max_val = np.max(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv_val = (std_val / mean_val) * 100 if mean_val != 0 else 0

        stats.append({
            "样本类型": sample_type,
            "指标": col,
            "样本量": sample_size,
            "最小值": min_val,
            "最大值": max_val,
            "平均值": mean_val,
            "标准差": std_val,
            "变异系数(%)": cv_val
        })
    return stats


# 计算全部数据、训练集和测试集的统计特征
all_stats = calculate_statistics(data, list(target_columns.values()), "全部样本")
train_stats = calculate_statistics(train_data, list(target_columns.values()), "训练集")
test_stats = calculate_statistics(test_data, list(target_columns.values()), "测试集")

# 合并统计结果
all_statistics = pd.DataFrame(all_stats + train_stats + test_stats)

# 保存统计结果到CSV文件
all_statistics.to_excel("../datasets/target_columns_statistics.xlsx", index=False)

# 保存数据集到不同的CSV文件 - 按照更有逻辑的顺序重新组织
dataset_list = [
    # 基础数据集
    (train_dataset5, test_dataset5, '01_raw_spectral_bands_sgd_dr'),
    (train_dataset8, test_dataset8, '02_raw_spectral_bands_snv'),
    (train_dataset11, test_dataset11, '03_raw_spectral_bands_msc'),
    (train_dataset14, test_dataset14, '04_raw_spectral_bands_dwt'),
    (train_dataset17, test_dataset17, '17_raw_spectral_bands_sgd_dr_cars'),

    # 土壤养分 + 光谱波段（无环境信息）- 按照处理方法递进
    (train_dataset1, test_dataset1, '05_soil_nutrients_spectral_bands_raw'),
    (train_dataset3, test_dataset3, '06_soil_nutrients_spectral_bands_sgd_dr'),
    (train_dataset6, test_dataset6, '07_soil_nutrients_spectral_bands_snv'),
    (train_dataset9, test_dataset9, '08_soil_nutrients_spectral_bands_msc'),
    (train_dataset12, test_dataset12, '09_soil_nutrients_spectral_bands_dwt'),
    (train_dataset15, test_dataset15, '10_soil_nutrients_spectral_bands_msc_dwt'),
    (train_dataset18, test_dataset18, '18_soil_nutrients_spectral_bands_sgd_dr_cars'),

    # 土壤养分 + 光谱波段 + 环境信息 - 最复杂的组合
    (train_dataset2, test_dataset2, '11_soil_nutrients_spectral_bands_environment_raw'),
    (train_dataset4, test_dataset4, '12_soil_nutrients_spectral_bands_environment_sgd_dr'),
    (train_dataset7, test_dataset7, '13_soil_nutrients_spectral_bands_environment_snv'),
    (train_dataset10, test_dataset10, '14_soil_nutrients_spectral_bands_environment_msc'),
    (train_dataset13, test_dataset13, '15_soil_nutrients_spectral_bands_environment_dwt'),
    (train_dataset16, test_dataset16, '16_soil_nutrients_spectral_bands_environment_msc_dwt'),
    (train_dataset19, test_dataset19, '19_soil_nutrients_spectral_bands_environment_sgd_dr_cars')
]

# Add the target-specific CARS datasets
for idx, (_, target_col) in enumerate(target_columns.items(), start=20):
    dataset_list.append((
        globals()[f'train_dataset{idx}'], 
        globals()[f'test_dataset{idx}'], 
        f'{idx:02d}_{target_col}_specific_cars'
    ))

# Save all datasets
for i, (train_dataset, test_dataset, name) in enumerate(dataset_list):
    train_dataset.to_csv(f'../datasets/train/{name}.csv', index=False)
    test_dataset.to_csv(f'../datasets/test/{name}.csv', index=False)

print("所有数据集已处理完成，训练集和测试集已分别保存到 ../datasets/train 和 ../datasets/test 目录。")
print("目标列的统计特征已保存到 ../datasets/target_columns_statistics.xlsx 文件。")
print(f"为每个目标变量创建了特定的CARS特征选择数据集，从dataset20到dataset{19+len(target_columns)}。")