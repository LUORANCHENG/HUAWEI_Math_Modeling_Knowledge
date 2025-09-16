"""
滑动窗口通用模板
==============

基于论文《草原放牧策略研究》中的滑动窗口实现
优先使用pandas等现有API，提供最精简的通用模板
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple


def sliding_window_transform(data: Union[pd.DataFrame, np.ndarray, list], 
                           window_size: int = 3, 
                           prediction_steps: int = 1,
                           drop_na: bool = True) -> pd.DataFrame:
    """
    滑动窗口转换函数 - 最精简通用模板
    
    将时间序列数据转换为监督学习格式，基于论文中的核心原理实现
    
    Parameters:
    -----------
    data : pandas.DataFrame, numpy.ndarray, or list
        时间序列数据，可以是单变量或多变量
    window_size : int, default=3
        滑动窗口大小（输入特征的时间步数）
        论文中不同模型的最佳窗口大小：
        - BP神经网络: 68
        - LSTM网络: 48  
        - Attention-BiLSTM网络: 72
    prediction_steps : int, default=1
        预测步数（输出目标的时间步数）
    drop_na : bool, default=True
        是否删除包含缺失值的行
        
    Returns:
    --------
    pandas.DataFrame
        转换后的监督学习数据集
        前面的列是输入特征(t-n, ..., t-1)
        后面的列是预测目标(t, t+1, ..., t+m)
        
    Example:
    --------
    >>> # 示例数据：土壤湿度时间序列
    >>> soil_moisture = [10, 12, 15, 18, 20, 22, 25, 23, 20, 16, 13, 11]
    >>> result = sliding_window_transform(soil_moisture, window_size=3, prediction_steps=1)
    >>> print(result)
    
    基于论文原理：
    - 第一个样本：输入[10,12,15] → 输出[18]
    - 第二个样本：输入[12,15,18] → 输出[20]
    - 以此类推...
    """
    
    # 转换为DataFrame格式（利用pandas API）
    if isinstance(data, (list, np.ndarray)):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # 获取变量数量
    n_vars = 1 if df.ndim == 1 else df.shape[1]
    
    # 存储所有滑动窗口列
    cols = []
    column_names = []
    
    # 构建输入特征列 (t-window_size, t-window_size+1, ..., t-1)
    # 使用pandas的shift()方法 - 这是最高效的API
    for i in range(window_size, 0, -1):
        shifted_col = df.shift(i)
        cols.append(shifted_col)
        # 为每个变量和时间步生成列名
        if n_vars == 1:
            column_names.append(f'feature_t-{i}')
        else:
            for j in range(n_vars):
                column_names.append(f'var{j+1}_t-{i}')
    
    # 构建预测目标列 (t, t+1, ..., t+prediction_steps-1)
    for i in range(prediction_steps):
        shifted_col = df.shift(-i)
        cols.append(shifted_col)
        # 为预测目标生成列名
        if n_vars == 1:
            if i == 0:
                column_names.append('target_t')
            else:
                column_names.append(f'target_t+{i}')
        else:
            for j in range(n_vars):
                if i == 0:
                    column_names.append(f'var{j+1}_target_t')
                else:
                    column_names.append(f'var{j+1}_target_t+{i}')
    
    # 使用pandas.concat()合并所有列 - 高效的API调用
    result = pd.concat(cols, axis=1)
    result.columns = column_names
    
    # 删除包含NaN的行（可选）
    if drop_na:
        result = result.dropna()
    
    return result


def prepare_train_test_data(transformed_data: pd.DataFrame, 
                          test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    数据集划分函数
    
    将滑动窗口转换后的数据划分为训练集和测试集
    
    Parameters:
    -----------
    transformed_data : pandas.DataFrame
        滑动窗口转换后的数据
    test_size : float, default=0.2
        测试集比例，论文中使用0.2
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : pandas.DataFrame
        训练和测试的特征与目标数据
    """
    
    # 分离特征和目标
    # 包含'target'的列是目标变量
    target_cols = [col for col in transformed_data.columns if 'target' in col]
    feature_cols = [col for col in transformed_data.columns if 'target' not in col]
    
    X = transformed_data[feature_cols]
    y = transformed_data[target_cols]
    
    # 按时间顺序划分（不打乱）- 符合时间序列特点
    split_idx = int(len(transformed_data) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test


# 使用示例和测试代码
if __name__ == "__main__":
    print("=== 滑动窗口通用模板测试 ===")
    
    # 示例1：单变量时间序列（模拟土壤湿度数据）
    print("\n1. 单变量时间序列示例：")
    soil_moisture = [10, 12, 15, 18, 20, 22, 25, 23, 20, 16, 13, 11]
    print(f"原始数据: {soil_moisture}")
    
    # 使用窗口大小为3进行转换
    result_single = sliding_window_transform(soil_moisture, window_size=3, prediction_steps=1)
    print("\n转换后的监督学习数据:")
    print(result_single)
    
    # 示例2：多变量时间序列（模拟土壤湿度+蒸发量）
    print("\n\n2. 多变量时间序列示例：")
    multi_data = pd.DataFrame({
        'soil_moisture': [10, 12, 15, 18, 20, 22, 25, 23, 20, 16],
        'evaporation': [5, 6, 7, 8, 9, 10, 11, 10, 9, 8]
    })
    print("原始数据:")
    print(multi_data)
    
    result_multi = sliding_window_transform(multi_data, window_size=3, prediction_steps=1)
    print("\n转换后的监督学习数据:")
    print(result_multi)
    
    # 示例3：数据集划分
    print("\n\n3. 数据集划分示例：")
    X_train, X_test, y_train, y_test = prepare_train_test_data(result_multi, test_size=0.3)
    print(f"训练集特征形状: {X_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")
    print(f"训练集目标形状: {y_train.shape}")
    print(f"测试集目标形状: {y_test.shape}")
    
    # 示例4：论文中的窗口大小配置
    print("\n\n4. 论文中不同模型的最佳窗口大小:")
    print("BP神经网络: window_size=68")
    print("LSTM网络: window_size=48")  
    print("Attention-BiLSTM网络: window_size=72")
    
    print("\n=== 测试完成 ===")


"""
使用说明:
========

1. 基本用法:
   result = sliding_window_transform(your_data, window_size=3)

2. 论文中的配置:
   # 对于BP神经网络
   result = sliding_window_transform(data, window_size=68)
   
   # 对于LSTM网络
   result = sliding_window_transform(data, window_size=48)
   
   # 对于Attention-BiLSTM网络
   result = sliding_window_transform(data, window_size=72)

3. 多步预测:
   result = sliding_window_transform(data, window_size=10, prediction_steps=3)

4. 完整流程:
   # 转换数据
   transformed = sliding_window_transform(raw_data, window_size=5)
   # 划分数据集
   X_train, X_test, y_train, y_test = prepare_train_test_data(transformed)
   # 后续可以直接用于机器学习模型训练

核心优势:
========
- 优先使用pandas/numpy API，性能优异
- 代码精简，易于理解和修改
- 支持单变量和多变量时间序列
- 自动生成有意义的列名
- 符合论文中的实现原理
- 可配置窗口大小和预测步数
""" 