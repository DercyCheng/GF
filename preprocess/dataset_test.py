import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os

# Set output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

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
spectral_bands = [f'{i}' for i in range(350, 2501)]
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

# Function to clean data and handle non-numeric values
def clean_data(df, columns_to_clean):
    cleaned_df = df.copy()
    
    for col in columns_to_clean:
        if col in cleaned_df.columns:
            # Convert empty strings, spaces, and other non-numeric values to NaN
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    return cleaned_df

# Clean spectral bands and numeric columns first
numeric_columns = list(soil_nutrients.values()) + list(environment_info.values()) + spectral_bands
data = clean_data(data, numeric_columns)

# Function to preprocess soil nutrients data
def preprocess_soil_nutrients(data, nutrient_cols):
    # Create a copy to avoid modifying the original dataframe
    processed_data = data[nutrient_cols].copy()
    
    # Handle missing values - fill with median
    processed_data = processed_data.fillna(processed_data.median())
    
    # Handle outliers using IQR method
    for col in processed_data.columns:
        Q1 = processed_data[col].quantile(0.25)
        Q3 = processed_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        processed_data[col] = processed_data[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Apply standardization (z-score normalization)
    processed_data = (processed_data - processed_data.mean()) / processed_data.std()
    
    return processed_data

# Function to preprocess environmental information
def preprocess_environment_info(data, env_cols):
    # Create a copy to avoid modifying the original dataframe
    processed_data = data[env_cols].copy()
    
    # Handle missing values - fill with median
    processed_data = processed_data.fillna(processed_data.median())
    
    # Handle outliers using IQR method
    for col in processed_data.columns:
        Q1 = processed_data[col].quantile(0.25)
        Q3 = processed_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        processed_data[col] = processed_data[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Apply min-max normalization for environmental data
    processed_data = (processed_data - processed_data.min()) / (processed_data.max() - processed_data.min())
    
    # Apply log transformation to highly skewed features
    for col in processed_data.columns:
        # Check skewness
        skewness = processed_data[col].skew()
        if abs(skewness) > 1:  # if data is highly skewed
            # Add small constant to handle zeros
            processed_data[col] = np.log1p(processed_data[col] - processed_data[col].min() + 1e-6)
    
    return processed_data

# Function to apply Savitzky-Golay filter and first derivative
def apply_sgd(data, bands):
    # Ensure data is numeric and handle NaNs before filtering
    filtered_data = data[bands].fillna(data[bands].median())
    sgd_filtered = savgol_filter(filtered_data, window_length=5, polyorder=2, deriv=0, axis=0)
    sgd_derivative = savgol_filter(sgd_filtered, window_length=5, polyorder=2, deriv=1, axis=0)
    return sgd_derivative

# Function to apply Standard Normal Variate (SNV) normalization
def apply_sae(data, bands):
    # Ensure data is numeric and handle NaNs before normalization
    filtered_data = data[bands].fillna(data[bands].median())
    sae_normalized = (filtered_data - filtered_data.mean(axis=0)) / filtered_data.std(axis=0)
    return sae_normalized

# Preprocess soil nutrients and environment info
preprocessed_nutrients = preprocess_soil_nutrients(data, list(soil_nutrients.values()))
preprocessed_env_info = preprocess_environment_info(data, list(environment_info.values()))

# 数据集1：预处理后的土壤养分含量+光谱波段
dataset1 = pd.concat([preprocessed_nutrients, data[spectral_bands]], axis=1)

# 数据集2：预处理后的土壤养分含量+光谱波段+预处理后的环境信息
dataset2 = pd.concat([preprocessed_nutrients, data[spectral_bands], preprocessed_env_info], axis=1)

# 数据集3：预处理后的土壤养分含量+光谱波段，进行SGD降噪以及DR一阶微分
dataset3 = pd.concat([preprocessed_nutrients, pd.DataFrame(apply_sgd(data, spectral_bands), columns=spectral_bands)], axis=1)

# 数据集4：预处理后的土壤养分含量+光谱波段+预处理后的环境信息，进行SGD降噪以及一阶微分
dataset4 = pd.concat([preprocessed_nutrients, pd.DataFrame(apply_sgd(data, spectral_bands), columns=spectral_bands), 
                     preprocessed_env_info], axis=1)

# 数据集5：target_columns + 经过SGD+DR处理的光谱波段
dataset5 = pd.concat([data[list(target_columns.values())], pd.DataFrame(apply_sgd(data, spectral_bands), columns=spectral_bands)], axis=1)

# 数据集6：预处理后的土壤养分含量+光谱波段，进行SAE处理
dataset6 = pd.concat([preprocessed_nutrients, pd.DataFrame(apply_sae(data, spectral_bands), columns=spectral_bands)], axis=1)

# 数据集7：预处理后的土壤养分含量+光谱波段+预处理后的环境信息，进行SAE处理
dataset7 = pd.concat([preprocessed_nutrients, pd.DataFrame(apply_sae(data, spectral_bands), columns=spectral_bands), 
                     preprocessed_env_info], axis=1)

# 数据集8：target_columns + 经过SAE处理的光谱波段
dataset8 = pd.concat([data[list(target_columns.values())], pd.DataFrame(apply_sae(data, spectral_bands), columns=spectral_bands)], axis=1)

# 保存数据集到不同的Excel文件
dataset1.to_excel(os.path.join(output_dir, 'data_soil_nutrients_spectral_bands.xlsx'), index=False)
dataset2.to_excel(os.path.join(output_dir, 'data_soil_nutrients_spectral_bands_environment.xlsx'), index=False)
dataset3.to_excel(os.path.join(output_dir, 'data_soil_nutrients_spectral_bands_sgd_dr.xlsx'), index=False)
dataset4.to_excel(os.path.join(output_dir, 'data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx'), index=False)
dataset5.to_excel(os.path.join(output_dir, 'data_spectral_bands_sgd_dr.xlsx'), index=False)
dataset6.to_excel(os.path.join(output_dir, 'data_soil_nutrients_spectral_bands_sae.xlsx'), index=False)
dataset7.to_excel(os.path.join(output_dir, 'data_soil_nutrients_spectral_bands_environment_sae.xlsx'), index=False)
dataset8.to_excel(os.path.join(output_dir, 'data_spectral_bands_sae.xlsx'), index=False)