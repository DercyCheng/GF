import pandas as pd
import numpy as np
from tabulate import tabulate
from scipy import stats

def print_statistics(y, sample_type):
    n = len(y)
    min_val = np.min(y)
    max_val = np.max(y)
    range_val = f"{min_val:.2f}-{max_val:.2f}"
    mean_val = np.mean(y)
    median_val = np.median(y)
    std_val = np.std(y)
    kurtosis_val = stats.kurtosis(y)
    cv = (std_val / mean_val) * 100
    return [sample_type, n, range_val, f"{mean_val:.2f}", f"{median_val:.2f}", f"{std_val:.2f}", f"{kurtosis_val:.2f}", f"{cv:.2f}"]

target_columns = ["易氧化有机碳(mg/g)", "有机碳含量(g/kg)","全碳(g/kg)","水溶性有机碳(mg/g)"]

file_paths = [
    ("./data.xlsx", ""),
    # ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
    # ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNSBE"),
    # ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
    # ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD")
]

# 生成统计特征
table = []
headers = ["数据集", "样本量", "含量范围", "均值(g/kg)", "中位数(g/kg)", "标准差", "峰度", "变异系数(%)"]
for file_path, dataset_name in file_paths:
    data = pd.read_excel(file_path)
    train_size = int(len(data) * 0.8)  # 80% for training
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    for column in target_columns:
        table.append(print_statistics(data[column], f"{dataset_name} - 总体样本 - {column}"))
        table.append(print_statistics(train_data[column], f"{dataset_name} - 训练样本 - {column}"))
        table.append(print_statistics(test_data[column], f"{dataset_name} - 测试样本 - {column}"))

print(tabulate(table, headers=headers, tablefmt="grid"))

# 将统计特征保存到一个.xlsx文件中
df = pd.DataFrame(table, columns=headers)
df.to_excel('statistical_characteristics.xlsx', index=False)
