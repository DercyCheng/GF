import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

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
spectral_bands = [f'{i}' for i in range(400, 2400, 10)]
environment_info = {
    '海拔测量': 'Elevation',
    'Longitude': 'Longitude',
    'latitude': 'Latitude',
    '坡度': 'Slope',
    '坡向': 'Aspect',
    '海拔': 'Altitude',
    '大于10度积温': 'GDD10',
    '年均降雨': 'AnnualRainfall',
    '年均温度': 'AnnualTemp',
    '代数': 'Generation',
    '林龄': 'ForestAge'
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

# 数据集1：土壤养分含量+光谱波段
dataset1 = data[list(soil_nutrients.values()) + spectral_bands]

# 数据集2：土壤养分含量+光谱波段+环境信息
dataset2 = data[list(soil_nutrients.values()) + spectral_bands + list(environment_info.values())]

# 数据集3：土壤养分含量+光谱波段，进行SGD降噪以及DR一阶微分
# 进行SGD降噪
sgd_filtered = savgol_filter(data[spectral_bands], window_length=5, polyorder=2, deriv=0, axis=0)
# 进行一阶微分
sgd_derivative = savgol_filter(sgd_filtered, window_length=5, polyorder=2, deriv=1, axis=0)
dataset3 = pd.concat([data[list(soil_nutrients.values())], pd.DataFrame(sgd_derivative, columns=spectral_bands)], axis=1)

# 数据集4：土壤养分含量+光谱波段+环境信息，进行SGD降噪以及一阶微分
# 进行SGD降噪
sgd_filtered_env = savgol_filter(data[spectral_bands], window_length=5, polyorder=2, deriv=0, axis=0)
# 进行一阶微分
sgd_derivative_env = savgol_filter(sgd_filtered_env, window_length=5, polyorder=2, deriv=1, axis=0)
dataset4 = pd.concat([data[list(soil_nutrients.values())], pd.DataFrame(sgd_derivative_env, columns=spectral_bands), data[list(environment_info.values())]], axis=1)

# 数据集5：经过SGD+DR处理的光谱波段+target_columns
sgd_filtered_target = savgol_filter(data[spectral_bands], window_length=5, polyorder=2, deriv=0, axis=0)
sgd_derivative_target = savgol_filter(sgd_filtered_target, window_length=5, polyorder=2, deriv=1, axis=0)
dataset5 = pd.concat([
    pd.DataFrame(sgd_derivative_target, columns=spectral_bands),
    data[list(target_columns.values())]  # 确保包含 'OM' 和 'WOC'
], axis=1)


# 保存数据集到不同的Excel文件
dataset1.to_excel('data_soil_nutrients_spectral_bands.xlsx', index=False)
dataset2.to_excel('data_soil_nutrients_spectral_bands_environment.xlsx', index=False)
dataset3.to_excel('data_soil_nutrients_spectral_bands_sgd_dr.xlsx', index=False)
dataset4.to_excel('data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx', index=False)
dataset5.to_excel('data_spectral_bands_sgd_dr.xlsx', index=False)