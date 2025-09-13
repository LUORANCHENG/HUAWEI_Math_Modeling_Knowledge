#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Borderline-SMOTE过采样算法通用模板
基于论文《出血性脑卒中临床智能诊疗建模》中的数据预处理方法
适用于小样本数据的类别不平衡问题
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

def borderline_smote_template(X, y, random_state=42, k_neighbors=5):
    """
    Borderline-SMOTE过采样算法通用模板
    
    参数:
    X: 特征数据 (numpy array 或 pandas DataFrame)
    y: 标签数据 (numpy array 或 pandas Series)
    random_state: 随机种子，确保结果可重复
    k_neighbors: 近邻数量，默认为5
    
    返回:
    X_resampled: 过采样后的特征数据
    y_resampled: 过采样后的标签数据
    """
    
    # 1. 数据预处理
    print("原始数据分布:")
    print(Counter(y))
    
    # 2. 创建Borderline-SMOTE实例
    borderline_smote = BorderlineSMOTE(
        random_state=random_state,
        k_neighbors=k_neighbors,
        kind='borderline-1'  # 使用borderline-1变体
    )
    
    # 3. 执行过采样
    X_resampled, y_resampled = borderline_smote.fit_resample(X, y)
    
    # 4. 输出结果
    print("过采样后数据分布:")
    print(Counter(y_resampled))
    
    return X_resampled, y_resampled

def complete_preprocessing_pipeline(X, y, test_size=0.2, random_state=42):
    """
    完整的数据预处理管道（包含过采样和标准化）
    
    参数:
    X: 特征数据
    y: 标签数据
    test_size: 测试集比例
    random_state: 随机种子
    
    返回:
    X_train_resampled: 训练集特征（已过采样和标准化）
    X_test_scaled: 测试集特征（已标准化）
    y_train_resampled: 训练集标签（已过采样）
    y_test: 测试集标签
    scaler: 标准化器对象
    """
    
    # 1. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 2. 标准化处理（在过采样前）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. 执行Borderline-SMOTE过采样（只在训练集上）
    X_train_resampled, y_train_resampled = borderline_smote_template(
        X_train_scaled, y_train, random_state=random_state
    )
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, scaler

# 使用示例
if __name__ == "__main__":
    # 示例：生成模拟数据（模拟医学数据的类别不平衡情况）
    from sklearn.datasets import make_classification
    
    # 生成不平衡数据集
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=7,  # 模拟7分类问题（mRS评分0-6）
        weights=[0.4, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02],  # 模拟不平衡分布
        random_state=42
    )
    
    print("=== 基础Borderline-SMOTE使用示例 ===")
    X_resampled, y_resampled = borderline_smote_template(X, y)
    
    print("\n=== 完整预处理管道使用示例 ===")
    X_train_res, X_test_scaled, y_train_res, y_test, scaler = complete_preprocessing_pipeline(X, y)
    
    print(f"训练集形状: {X_train_res.shape}")
    print(f"测试集形状: {X_test_scaled.shape}")
    
    # 可选：保存处理后的数据
    # np.save('X_train_resampled.npy', X_train_res)
    # np.save('y_train_resampled.npy', y_train_res)