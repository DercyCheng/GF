import pandas as pd
import numpy as np
from tabulate import tabulate
import csv

def print_statistics(y, sample_type):
    n = len(y)
    min_val = np.min(y)
    max_val = np.max(y)
    mean_val = np.mean(y)
    std_val = np.std(y)
    cv = (std_val / mean_val) * 100
    return [sample_type, n, f"{min_val:.2f}", f"{max_val:.2f}", f"{mean_val:.2f}", f"{std_val:.2f}", f"{cv:.2f}"]

target_columns = ["易氧化有机碳(mg/g)", "有机碳含量(g/kg)", "水溶性有机碳(mg/g)"]

file_paths = [
    ("../datasets/data_soil_nutrients_spectral_bands.xlsx", "SNSB"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment.xlsx", "SNDBE"),
    ("../datasets/data_soil_nutrients_spectral_bands_sgd_dr.xlsx", "SNSBSD"),
    ("../datasets/data_soil_nutrients_spectral_bands_environment_sgd_dr.xlsx", "SNSBESD")
]

# 样本数量
train_samples = 215

# 生成统计特征
table = []
headers = ["样本类型", "样本量", "最小值", "最大值", "平均值", "标准差", "变异系数(%)"]
for file_path, dataset_name in file_paths:
    data = pd.read_excel(file_path)
    train_data = data.iloc[:train_samples]
    validation_data = data.iloc[train_samples:]
    
    for column in target_columns:
        table.append(print_statistics(data[column], f"{dataset_name} - 总体样本 - {column}"))
        table.append(print_statistics(train_data[column], f"{dataset_name} - 训练样本 - {column}"))
        table.append(print_statistics(validation_data[column], f"{dataset_name} - 验证样本 - {column}"))

print(tabulate(table, headers=headers, tablefmt="grid"))

# 将统计特征保存到一个.csv文件中
with open('statistical_characteristics.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(table)
