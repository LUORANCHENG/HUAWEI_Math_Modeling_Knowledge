#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SVM算法通用模板
基于scikit-learn实现的支持向量机分类器
包含训练、预测、评估三个核心功能
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class SVMTemplate:
    """
    SVM算法通用模板类
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42):
        """
        初始化SVM分类器
        
        参数:
        kernel: 核函数类型，默认'rbf'
        C: 正则化参数，默认1.0
        gamma: 核函数系数，默认'scale'
        probability: 是否启用概率估计，默认True
        random_state: 随机种子，默认42
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        训练SVM模型
        
        参数:
        X: 特征矩阵
        y: 标签向量
        test_size: 测试集比例，默认0.2
        random_state: 随机种子，默认42
        
        返回:
        训练集和测试集的划分结果
        """
        print("=== SVM模型训练开始 ===")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        print(f"训练集样本数: {X_train.shape[0]}")
        print(f"测试集样本数: {X_test.shape[0]}")
        print(f"特征维度: {X_train.shape[1]}")
        print("=== SVM模型训练完成 ===\n")
        
        return X_train, X_test, y_train, y_test
    
    def predict(self, X, return_proba=False):
        """
        使用训练好的模型进行预测
        
        参数:
        X: 待预测的特征矩阵
        return_proba: 是否返回预测概率，默认False
        
        返回:
        预测结果或预测概率
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 标准化输入数据
        X_scaled = self.scaler.transform(X)
        
        if return_proba:
            # 返回预测概率（置信度）
            return self.model.predict_proba(X_scaled)
        else:
            # 返回预测类别
            return self.model.predict(X_scaled)
    
    def evaluate(self, X_test, y_test, detailed=True):
        """
        评估模型性能
        
        参数:
        X_test: 测试集特征
        y_test: 测试集标签
        detailed: 是否输出详细评估结果，默认True
        
        返回:
        评估指标字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        print("=== SVM模型评估开始 ===")
        
        # 预测
        y_pred = self.predict(X_test)
        y_proba = self.predict(X_test, return_proba=True)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # 构建评估结果
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        if detailed:
            print(f"准确率 (Accuracy): {accuracy:.4f}")
            print(f"精确率 (Precision): {precision:.4f}")
            print(f"召回率 (Recall): {recall:.4f}")
            print("\n分类报告:")
            print(classification_report(y_test, y_pred))
            print("\n混淆矩阵:")
            print(confusion_matrix(y_test, y_pred))
        
        print("=== SVM模型评估完成 ===\n")
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        交叉验证评估
        
        参数:
        X: 特征矩阵
        y: 标签向量
        cv: 交叉验证折数，默认5
        
        返回:
        交叉验证分数
        """
        if not self.is_fitted:
            print("注意: 模型尚未训练，将使用默认参数进行交叉验证")
        
        # 标准化数据
        X_scaled = self.scaler.fit_transform(X) if not self.is_fitted else self.scaler.transform(X)
        
        # 执行交叉验证
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        
        print(f"=== {cv}折交叉验证结果 ===")
        print(f"各折准确率: {cv_scores}")
        print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("=== 交叉验证完成 ===\n")
        
        return cv_scores

def demo_usage():
    """
    演示SVM模板的使用方法
    """
    print("SVM算法通用模板使用演示\n")
    
    # 生成示例数据
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # 创建SVM实例
    svm_model = SVMTemplate(kernel='rbf', C=1.0, probability=True)
    
    # 1. 训练模型
    X_train, X_test, y_train, y_test = svm_model.train(X, y)
    
    # 2. 预测
    predictions = svm_model.predict(X_test)
    probabilities = svm_model.predict(X_test, return_proba=True)
    
    print("预测结果示例:")
    print(f"前10个预测类别: {predictions[:10]}")
    print(f"前5个预测概率: {probabilities[:5]}\n")
    
    # 3. 评估
    metrics = svm_model.evaluate(X_test, y_test)
    
    # 4. 交叉验证
    cv_scores = svm_model.cross_validate(X, y, cv=5)

if __name__ == "__main__":
    demo_usage() 