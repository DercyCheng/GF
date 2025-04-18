import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 定义各指标的评分标准 - 修正版内梅罗指数法
def calculate_ph_score(ph):
    """pH评分函数：区分酸性(<7.0)和碱性(≥7.0)土壤"""
    if ph < 7.0:
        # 酸性土壤
        if ph <= 4.5:  # 差一级
            return ph / 4.5
        elif 4.5 < ph <= 5.5:  # 中等一级
            return 1 + (ph - 4.5) / (5.5 - 4.5)
        elif 5.5 < ph <= 6.5:  # 较好一级
            return 2 + (ph - 5.5) / (6.5 - 5.5)
        else:  # 好一级 (6.5 < ph < 7.0)
            return 3.0
    else:
        # 碱性土壤 - 简化处理
        if ph <= 7.5:
            return 2.5  # 较好
        elif 7.5 < ph <= 8.0:
            return 2.0  # 中等
        elif 8.0 < ph <= 8.5:
            return 1.0  # 较差
        else:
            return 0.5  # 差

def calculate_standardized_score(value, xa, xc, xp):
    """
    根据分级标准计算标准化得分
    (1) 差一级 Ci ≤ Xa: Pi = Ci/Xa (Pi ≤ 1)
    (2) 中等一级 Xa < Ci ≤ Xc: Pi = 1+(Ci-Xa)/(Xc-Xa) (1 < Pi ≤ 2)
    (3) 较好一级 Xc < Ci ≤ Xp: Pi = 2+(Ci-Xc)/(Xp-Xc) (2 < Pi ≤ 3)
    (4) 好一级 Ci > Xp: Pi = 3
    """
    if value <= xa:
        return value / xa
    elif xa < value <= xc:
        return 1 + (value - xa) / (xc - xa)
    elif xc < value <= xp:
        return 2 + (value - xc) / (xp - xc)
    else:  # value > xp
        return 3.0

# 各指标分级标准（根据表3）
standard_levels = {
    'OM': {'xa': 10, 'xc': 30, 'xp': 50},     # 有机质(g/kg)
    'TN': {'xa': 1, 'xc': 1.5, 'xp': 2.5},    # 全氮(g/kg)
    'TP': {'xa': 0.5, 'xc': 1.5, 'xp': 2.5},  # 全磷(g/kg)
    'TK': {'xa': 10, 'xc': 30, 'xp': 50},     # 全钾(g/kg)
    'SN': {'xa': 60, 'xc': 100, 'xp': 150},   # 速效N(mg/kg)
    'SP': {'xa': 1, 'xc': 5, 'xp': 10},       # 有效磷(mg/kg)
    'SK': {'xa': 50, 'xc': 100, 'xp': 200}    # 速效钾(mg/kg)
}

# 计算各指标评分
df['pH_score'] = df['PH'].apply(calculate_ph_score)
df['有机质_score'] = df['OM'].apply(lambda x: calculate_standardized_score(x, **standard_levels['OM']))
df['全氮_score'] = df['TN'].apply(lambda x: calculate_standardized_score(x, **standard_levels['TN']))
df['全磷_score'] = df['TP'].apply(lambda x: calculate_standardized_score(x, **standard_levels['TP']))
df['全钾_score'] = df['TK'].apply(lambda x: calculate_standardized_score(x, **standard_levels['TK']))
df['速效N_score'] = df['SN'].apply(lambda x: calculate_standardized_score(x, **standard_levels['SN']))
df['速效P_score'] = df['SP'].apply(lambda x: calculate_standardized_score(x, **standard_levels['SP']))
df['速效K_score'] = df['SK'].apply(lambda x: calculate_standardized_score(x, **standard_levels['SK']))

# 应用修正的内梅罗公式计算肥力指数
score_columns = ['pH_score', '有机质_score', '全氮_score', '全磷_score', 
                '全钾_score', '速效N_score', '速效P_score', '速效K_score']

def modified_nemero_index(row, n):
    """修正的内梅罗指数计算方法"""
    avg = np.mean(row)
    min_score = np.min(row)
    return np.sqrt((avg**2 + min_score**2) / 2)

df['Soil_Fertility_Index'] = df[score_columns].apply(
    lambda row: modified_nemero_index(row, len(score_columns)), axis=1)

# 添加肥力等级分类
def classify_fertility(pi):
    if pi >= 2.7:
        return "很肥沃"
    elif 1.8 <= pi < 2.7:
        return "肥沃"
    elif 0.9 <= pi < 1.8:
        return "一般"
    else:
        return "贫瘠"

df['肥力等级'] = df['Soil_Fertility_Index'].apply(classify_fertility)

# 保存结果
df.to_excel('soil_fertility_results.xlsx', index=False)

print("土壤肥力指数计算完成，结果已保存至soil_fertility_results.xlsx")