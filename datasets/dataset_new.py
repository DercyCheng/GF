import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

# 读取Excel文件
data = pd.read_excel('data.xlsx')

# 定义列名
soil_nutrients = ['PH', 'OM', '全氮(g/kg)', '全磷(g/kg)', '全钾(g/kg)', '速效N(mg/kg)', '速效p(mg/kg)', '速效k(mg/kg)', 'B(mg/kg)', 'Cu(mg/kg)', 'Zn(mg/kg)', 'Fe(mg/kg)', 'Ca(mg/kg)', 'Mg(mg/kg)', 'TC']
spectral_bands = [str(i) for i in range(400, 2400, 10)]
environment_info = ['海拔测量', 'Longitude', 'latitude', '坡度', '坡向', '海拔', '大于10度积温', '年均降雨', '年均温度', '代数', '林龄']
soc_columns = ['EOC', 'SOC', 'WOC']
# 将光谱波段列名转换为字符串类型
data.columns = data.columns.map(str)

# 1. 数据集1：soc+soil
soc_soil = data[soc_columns + soil_nutrients]

# 2. 数据集2：soc+soil_nutrients+spectral_bands
dataset1 = data[soc_columns + soil_nutrients + spectral_bands]

# 3. 数据集3：soc+soil_nutrients+spectral_bands+environment_info
dataset2 = data[soc_columns + soil_nutrients + spectral_bands + environment_info]

# 4. 对 1 进行 SGD-DR-PCA
sgd_filtered_soc_soil = savgol_filter(soc_soil, window_length=5, polyorder=2, deriv=0, axis=0)
sgd_derivative_soc_soil = savgol_filter(sgd_filtered_soc_soil, window_length=5, polyorder=2, deriv=1, axis=0)
pca_soc_soil = PCA(n_components=5).fit_transform(sgd_derivative_soc_soil)
dataset5 = pd.DataFrame(pca_soc_soil, columns=[f'PC{i+1}' for i in range(pca_soc_soil.shape[1])])

# 5. 对 2 进行 SGD-DR-PCA
sgd_filtered_soc_spectral = savgol_filter(dataset1, window_length=5, polyorder=2, deriv=0, axis=0)
sgd_derivative_soc_spectral = savgol_filter(sgd_filtered_soc_spectral, window_length=5, polyorder=2, deriv=1, axis=0)
pca_soc_spectral = PCA(n_components=5).fit_transform(sgd_derivative_soc_spectral)
dataset6 = pd.DataFrame(pca_soc_spectral, columns=[f'PC{i+1}' for i in range(pca_soc_spectral.shape[1])])

# 6. 对 3 进行 SGD-DR-PCA
sgd_filtered_soc_env = savgol_filter(dataset2, window_length=5, polyorder=2, deriv=0, axis=0)
sgd_derivative_soc_env = savgol_filter(sgd_filtered_soc_env, window_length=5, polyorder=2, deriv=1, axis=0)
pca_soc_env = PCA(n_components=5).fit_transform(sgd_derivative_soc_env)
dataset7 = pd.DataFrame(pca_soc_env, columns=[f'PC{i+1}' for i in range(pca_soc_env.shape[1])])

# 保存数据集到不同的CSV文件
soc_soil.to_csv('processed/data_soc_soil.csv', index=False)
dataset1.to_csv('processed/data_soc_soil_nutrients_spectral_bands.csv', index=False)
dataset2.to_csv('processed/data_soc_soil_nutrients_spectral_bands_environment.csv', index=False)
dataset5.to_csv('processed/data_soc_soil_sgd_dr_pca.csv', index=False)
dataset6.to_csv('processed/data_soc_spectral_sgd_dr_pca.csv', index=False)
dataset7.to_csv('processed/data_soc_env_sgd_dr_pca.csv', index=False)
