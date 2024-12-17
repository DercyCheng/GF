import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import platform
import pandas as pd
from matplotlib import rcParams
# 设置字体以支持中文
if platform.system() == 'Windows':
    rcParams['font.sans-serif'] = ['SimHei']
elif platform.system() == 'Darwin':  # macOS
    rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:  # Linux
    rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 读取Excel文件
data = pd.read_excel('../data.xlsx')

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
}
spectral_bands = [f'{i}nm' for i in range(400, 2400, 10)]
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

# 生成相关性曲线图
carbon_columns = {
    '全碳(g/kg)': 'TC',
    '易氧化有机碳(mg/g)': 'EOC',
    '有机碳含量(g/kg)': 'SOC',
    '水溶性有机碳(mg/g)': 'WOC'
}

def plot_correlation_heatmap(data, x_column, y_columns, title):
    columns = [x_column] + y_columns
    corr = data[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.show()

# 生成 carbon_columns 中每项与 soil_nutrients、spectral_bands、environment_info 之间的相关性曲线图
for carbon_key, carbon_value in carbon_columns.items():
    plot_correlation_heatmap(data, carbon_key, list(soil_nutrients.keys()) + spectral_bands + list(environment_info.keys()), f'{carbon_value} Correlation with Soil Nutrients, Spectral Bands, and Environment Info')

# 预处理函数：SGD（S-G 降噪）+ DR（一阶微分）
def preprocess_data(data):
    # S-G 降噪
    data_sg = savgol_filter(data, window_length=11, polyorder=2, deriv=0)
    # 一阶微分
    data_dr = savgol_filter(data_sg, window_length=11, polyorder=2, deriv=1)
    return data_dr

# 对数据进行预处理
data_preprocessed = data.copy()
for column in spectral_bands:
    data_preprocessed[column] = preprocess_data(data[column])

# 生成经过预处理的 carbon_columns 中每项与 soil_nutrients、spectral_bands、environment_info 之间的相关性曲线图
for carbon_key, carbon_value in carbon_columns.items():
    plot_correlation_heatmap(data_preprocessed, carbon_key, list(soil_nutrients.keys()) + spectral_bands + list(environment_info.keys()), f'{carbon_value} Correlation with Soil Nutrients, Spectral Bands, and Environment Info (Preprocessed)')

