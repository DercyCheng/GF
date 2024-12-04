import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# 读取Excel文件
data = pd.read_excel('data.xlsx')

# 定义列名
soil_nutrients = ['PH', '有机质(g/kg)', '全氮(g/kg)', '全磷(g/kg)', '全钾(g/kg)', '速效N(mg/kg)', '速效p(mg/kg)', '速效k(mg/kg)', 'B(mg/kg)', 'Cu(mg/kg)', 'Zn(mg/kg)', 'Fe(mg/kg)', 'Ca(mg/kg)', 'Mg(mg/kg)', '全碳(g/kg)', '易氧化有机碳(mg/g)', '有机碳含量(g/kg)', '水溶性有机碳(mg/g)']
spectral_bands = [str(i) for i in range(400, 2400, 10)]
environment_info = ['海拔测量', 'Longitude', 'latitude', '坡度', '坡向', '海拔', '大于10度积温', '年均降雨', '年均温度', '代数', '林龄']

# 将光谱波段列名转换为字符串类型
data.columns = data.columns.map(str)

# 数据集1：土壤养分含量+光谱波段
dataset1 = data[soil_nutrients + spectral_bands]

# 数据集2：土壤养分含量+光谱波段+环境信息
dataset2 = data[soil_nutrients + spectral_bands + environment_info]

# 数据集3：土壤养分含量+光谱波段，进行SGD降噪以及DR一阶微分
# 进行SGD降噪
sgd_filtered = savgol_filter(data[spectral_bands], window_length=5, polyorder=2, deriv=0, axis=0)
# 进行一阶微分
sgd_derivative = savgol_filter(sgd_filtered, window_length=5, polyorder=2, deriv=1, axis=0)
dataset3 = pd.concat([data[soil_nutrients], pd.DataFrame(sgd_derivative, columns=spectral_bands)], axis=1)

# 数据集4：土壤养分含量+光谱波段+环境信息，进行SGD降噪以及一阶微分
# 进行SGD降噪
sgd_filtered_env = savgol_filter(data[spectral_bands], window_length=5, polyorder=2, deriv=0, axis=0)
# 进行一阶微分
sgd_derivative_env = savgol_filter(sgd_filtered_env, window_length=5, polyorder=2, deriv=1, axis=0)
dataset4 = pd.concat([data[soil_nutrients], pd.DataFrame(sgd_derivative_env, columns=spectral_bands), data[environment_info]], axis=1)

# 保存数据集到不同的Excel文件
dataset1.to_excel('data_soil_nutrients_spectral_bands.xlsx', index=False)
dataset2.to_excel('data_soil_nutrients_spectral_bands_environment.xlsx', index=False)
dataset3.to_excel('data_soil_nutrients_spectral_bands_sgd_dr.xlsx', index=False)
dataset4.to_excel('data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx', index=False)