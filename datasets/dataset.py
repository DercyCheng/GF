import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

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
spectral_bands = [f'{i}' for i in range(350, 2500)]
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
#
# # Apply SNV normalization to spectral bands
# data[spectral_bands] = data[spectral_bands].apply(lambda x: (x - x.mean()) / x.std(), axis=1)
#
# # Normalize other columns (soil nutrients and environment info)
# non_spectral_columns = list(soil_nutrients.values()) + list(environment_info.values())
# numeric_columns = data[non_spectral_columns].select_dtypes(include=[np.number]).columns
# data[numeric_columns] = data[numeric_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
#
# # Apply PCA for dimensionality reduction
# pca = PCA(n_components=10)
# pca_components = pca.fit_transform(data[spectral_bands])

# Function to apply Savitzky-Golay filter and first derivative
def apply_sgd(data, bands):
    sgd_filtered = savgol_filter(data[bands], window_length=5, polyorder=2, deriv=0, axis=0)
    sgd_derivative = savgol_filter(sgd_filtered, window_length=5, polyorder=2, deriv=1, axis=0)
    return sgd_derivative

# Function to apply Standard Normal Variate (SNV) normalization
def apply_sae(data, bands):
    sae_normalized = (data[bands] - data[bands].mean(axis=0)) / data[bands].std(axis=0)
    return sae_normalized

# 数据集1：土壤养分含量+光谱波段
dataset1 = data[list(soil_nutrients.values()) + spectral_bands]

# 数据集2：土壤养分含量+光谱波段+环境信息
dataset2 = data[list(soil_nutrients.values()) + spectral_bands + list(environment_info.values())]

# 数据集3：土壤养分含量+光谱波段，进行SGD降噪以及DR一阶微分
dataset3 = pd.concat([data[list(soil_nutrients.values())], pd.DataFrame(apply_sgd(data, spectral_bands), columns=spectral_bands)], axis=1)

# 数据集4：土壤养分含量+光谱波段+环境信息，进行SGD降噪以及一阶微分
dataset4 = pd.concat([data[list(soil_nutrients.values())], pd.DataFrame(apply_sgd(data, spectral_bands), columns=spectral_bands), data[list(environment_info.values())]], axis=1)

# 数据集5：target_columns + 经过SGD+DR处理的光谱波段
dataset5 = pd.concat([data[list(target_columns.values())], pd.DataFrame(apply_sgd(data, spectral_bands), columns=spectral_bands)], axis=1)

# 数据集6：土壤养分含量+光谱波段，进行SAE处理
dataset6 = pd.concat([data[list(soil_nutrients.values())], pd.DataFrame(apply_sae(data, spectral_bands), columns=spectral_bands)], axis=1)

# 数据集7：土壤养分含量+光谱波段+环境信息，进行SAE处理
dataset7 = pd.concat([data[list(soil_nutrients.values())], pd.DataFrame(apply_sae(data, spectral_bands), columns=spectral_bands), data[list(environment_info.values())]], axis=1)

# 数据集8：target_columns + 经过SAE处理的光谱波段
dataset8 = pd.concat([data[list(target_columns.values())], pd.DataFrame(apply_sae(data, spectral_bands), columns=spectral_bands)], axis=1)

# 保存数据集到不同的Excel文件
dataset1.to_excel('data_soil_nutrients_spectral_bands.xlsx', index=False)
dataset2.to_excel('data_soil_nutrients_spectral_bands_environment.xlsx', index=False)
dataset3.to_excel('data_soil_nutrients_spectral_bands_sgd_dr.xlsx', index=False)
dataset4.to_excel('data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx', index=False)
dataset5.to_excel('data_spectral_bands_sgd_dr.xlsx', index=False)
dataset6.to_excel('data_soil_nutrients_spectral_bands_sae.xlsx', index=False)
dataset7.to_excel('data_soil_nutrients_spectral_bands_environment_sae.xlsx', index=False)
dataset8.to_excel('data_spectral_bands_sae.xlsx', index=False)