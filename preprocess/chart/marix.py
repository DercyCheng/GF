import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import platform

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
    '全碳(g/kg)': 'TC',
    '易氧化有机碳(mg/g)': 'EOC',
    '有机碳含量(g/kg)': 'SOC',
    '水溶性有机碳(mg/g)': 'WOC'
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

# 选择相关列
selected_columns = list(soil_nutrients.keys()) + list(environment_info.keys())
selected_columns += [col for col in spectral_bands if col in data.columns]
data_selected = data[selected_columns]

# 将非数值数据转换为NaN，并填充NaN值
data_selected = data_selected.apply(pd.to_numeric, errors='coerce').fillna(0)

# 计算相关性矩阵
correlation_matrix = data_selected.corr()

# 绘制相关性矩阵图
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='viridis', linewidths=0.5)
plt.title('Correlation Matrix of Carbon Content with Spectral Bands, Soil Nutrients, and Environmental Factors', fontsize=16)
plt.xticks(ticks=range(len(selected_columns)), labels=[soil_nutrients.get(col, environment_info.get(col, col)) for col in selected_columns], rotation=90)
plt.yticks(ticks=range(len(selected_columns)), labels=[soil_nutrients.get(col, environment_info.get(col, col)) for col in selected_columns], rotation=0)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
plt.show()