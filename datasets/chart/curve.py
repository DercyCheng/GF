import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

plt.rcParams['font.family'] = ['Arial Unicode MS']  # 设置字体为SimHei，并添加一个备用字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义列名
soil_nutrients = ['PH', '有机质(g/kg)', '全氮(g/kg)', '全磷(g/kg)', '全钾(g/kg)', '速效N(mg/kg)', '速效p(mg/kg)',
                  '速效k(mg/kg)', 'B(mg/kg)', 'Cu(mg/kg)', 'Zn(mg/kg)', 'Fe(mg/kg)', 'Ca(mg/kg)', 'Mg(mg/kg)',
                  '全碳(g/kg)', '易氧化有机碳(mg/g)', '有机碳含量(g/kg)', '水溶性有机碳(mg/g)']
spectral_bands = [str(i) for i in range(500, 2400, 10)]
environment_info = ['海拔测量', 'Longitude', 'latitude', '坡度', '坡向', '海拔', '大于10度积温', '年均降雨', '年均温度',
                    '代数', '林龄']

# 定义列名映射
soil_nutrients_mapping = {
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
    '水溶性有机碳(mg/g)': 'WOC'  # 修正此行
}

environment_info_mapping = {
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

# 添加soc_columns的英文映射
soc_columns_mapping = {
    '有机质(g/kg)': 'OM',
    '全碳(g/kg)': 'TC',
    '易氧化有机碳(mg/g)': 'EOC',
    '有机碳含量(g/kg)': 'SOC',
    '水溶性有机碳(mg/g)': 'WOC'
}

# 加载数据
data = pd.read_excel('../data.xlsx')

# 将无法转换为浮点数的值替换为 NaN
data.replace(' ', np.nan, inplace=True)

# 移除非数值数据
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 将所有列转换为浮点数
data = data.astype(float)

# 目标列
soc_columns = ['有机质(g/kg)','全碳(g/kg)', '易氧化有机碳(mg/g)', '有机碳含量(g/kg)', '水溶性有机碳(mg/g)']

# 计算相关性
correlation_results = {}
correlation_results_sgdr = {}

for soc in soc_columns:
    correlations = {}
    correlations_sgdr = {}
    for col in spectral_bands + soil_nutrients + environment_info:
        if col in data.columns:
            correlation = data[soc].corr(data[col])
            correlations[col] = correlation

            # 检查并处理缺失值或异常值
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mean(), inplace=True)
            if data[col].std() == 0:
                continue

            # S-G降噪
            sg_filtered = savgol_filter(data[col], window_length=11, polyorder=3)
            # 一阶微分
            dr_filtered = np.gradient(sg_filtered)
            correlation_sgdr = data[soc].corr(pd.Series(dr_filtered))
            correlations_sgdr[col] = correlation_sgdr

    correlation_results[soc] = correlations
    correlation_results_sgdr[soc] = correlations_sgdr

# 打印相关性结果
for soc, correlations in correlation_results.items():
    soc_english = soc_columns_mapping.get(soc, soc)
    print(f"Correlation for {soc_english}:")
    for col, corr in correlations.items():
        print(f"  {col}: {corr}")

# 定义不同的颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 添加绘图函数以减少重复代码
def plot_correlation(correlations, title, xlabel, save_path, is_spectral=False):
    plt.figure(figsize=(14, 8))
    for i, (soc, corr_values) in enumerate(correlations.items()):
        soc_english = soc_columns_mapping.get(soc, soc)
        if is_spectral:
            # 仅过滤光谱波段
            spectral_correlations = {k: v for k, v in corr_values.items() if k in spectral_bands}
            if not spectral_correlations:
                continue
            x = np.arange(len(spectral_correlations))
            y = list(spectral_correlations.values())
        else:
            x = np.arange(len(corr_values))
            y = list(corr_values.values())
        
        if len(y) < 4:  # 确保 window_length 小于数据长度
            continue

        spl = make_interp_spline(x, y, k=3)
        x_smooth = np.linspace(x.min(), x.max(), 300)
        y_smooth = spl(x_smooth)
        
        plt.plot(x_smooth, y_smooth, color=colors[i % len(colors)], linewidth=2, label=soc_english)
    
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Correlation Coefficient', fontsize=14)
    plt.title(title, fontsize=16)
    
    # 生成标签
    labels = []
    first_soc = next(iter(correlations))
    for key in correlations[first_soc].keys():
        if key in soil_nutrients_mapping:
            labels.append(soil_nutrients_mapping[key])
        elif key in environment_info_mapping:
            labels.append(environment_info_mapping[key])
        else:
            labels.append(key)  # 保持光谱波段不变
    
    plt.xticks(ticks=np.linspace(0, len(labels)-1, len(labels)), labels=labels, rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)  # 保存图像
    # plt.show()

# 使用绘图函数替换现有的绘图代码

# 可视化相关性 - 所有 RAW 在一张图
plot_correlation(
    correlations=correlation_results,
    title='Correlation between SOC Columns and Features (RAW)',
    xlabel='Features',
    save_path='correlation_raw.png'
)

# 可视化相关性 - 所有 SGD-DR 在一张图
plot_correlation(
    correlations=correlation_results_sgdr,
    title='Correlation between SOC Columns and Features (SGD-DR)',
    xlabel='Features',
    save_path='correlation_sgdr.png'
)