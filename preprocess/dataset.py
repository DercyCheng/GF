import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

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

# 将字符串类型数据转换为数值类型
soil_nutrient_cols = list(soil_nutrients.values())
env_cols = list(environment_info.values())

# 确保所有土壤养分和环境信息列都是数值类型
for col in soil_nutrient_cols + env_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 对土壤养分数据进行差值补缺
data[soil_nutrient_cols] = data[soil_nutrient_cols].interpolate(method='linear').fillna(data[soil_nutrient_cols].mean())

# 对环境信息数据进行差值补缺
data[env_cols] = data[env_cols].interpolate(method='linear').fillna(data[env_cols].mean())

# 对土壤养分数据进行线性归一化
scaler_soil = MinMaxScaler()
data[soil_nutrient_cols] = scaler_soil.fit_transform(data[soil_nutrient_cols])

# 对环境信息数据进行线性归一化
scaler_env = MinMaxScaler()
data[env_cols] = scaler_env.fit_transform(data[env_cols])

# 数据集1：土壤养分含量+光谱波段
dataset1 = data[list(soil_nutrients.values()) + spectral_bands]

# 数据集2：土壤养分含量+光谱波段+环境信息
dataset2 = data[list(soil_nutrients.values()) + spectral_bands + list(environment_info.values())]

# dataset3：土壤养分含量+光谱波段
dataset3 = pd.concat([data[list(soil_nutrients.values())], data[spectral_bands]], axis=1)

# dataset4：土壤养分含量+光谱波段+环境信息
dataset4 = pd.concat([data[list(soil_nutrients.values())], data[spectral_bands], data[list(environment_info.values())]], axis=1)

# dataset5：target_columns + 光谱波段
dataset5 = pd.concat([
    data[list(target_columns.values())],
    data[spectral_bands]
], axis=1)

# 保存数据集到不同的Excel文件
dataset1.to_excel('data_soil_nutrients_spectral_bands.xlsx', index=False)
dataset2.to_excel('data_soil_nutrients_spectral_bands_environment.xlsx', index=False)
# dataset3.to_excel('data_soil_nutrients_spectral_bands_sgd_dr.xlsx', index=False)
# dataset4.to_excel('data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx', index=False)
# dataset5.to_excel('data_spectral_bands_sgd_dr.xlsx', index=False)