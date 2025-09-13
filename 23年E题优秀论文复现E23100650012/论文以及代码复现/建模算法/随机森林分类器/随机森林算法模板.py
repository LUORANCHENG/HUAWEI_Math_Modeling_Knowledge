#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
随机森林算法精简模板 - 基于《出血性脑卒中临床智能诊疗建模》论文实现

精简版本，只包含核心功能：训练、预测、评估
支持分类和回归任务
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')


class RandomForestTemplate:
    """
    随机森林算法精简模板类
    
    基于论文《出血性脑卒中临床智能诊疗建模》中的随机森林实现
    只包含核心功能：训练、预测、评估
    """
    
    def __init__(self, task_type='classification', n_estimators=100, 
                 max_depth=None, random_state=42):
        """
        初始化随机森林模型
        
        参数：
        - task_type: str, 任务类型 ('classification' 或 'regression')
        - n_estimators: int, 决策树数量，默认100
        - max_depth: int, 树的最大深度，默认None
        - random_state: int, 随机种子，确保可重复性
        """
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # 初始化模型
        if self.task_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:  # regression
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # 数据预处理器
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        训练随机森林模型
        
        参数：
        - X: array-like, 训练特征
        - y: array-like, 训练标签
        """
        print("开始训练随机森林模型...")
        
        # 数据预处理
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_processed = self.scaler.fit_transform(X)
        
        if self.task_type == 'classification':
            y_processed = self.label_encoder.fit_transform(y)
        else:
            y_processed = np.array(y)
        
        # 训练模型
        self.model.fit(X_processed, y_processed)
        self.is_fitted = True
        
        print(f"模型训练完成！")
        print(f"- 任务类型: {self.task_type}")
        print(f"- 决策树数量: {self.n_estimators}")
        print(f"- 训练样本数: {len(X_processed)}")
    
    def predict(self, X):
        """
        模型预测
        
        参数：
        - X: array-like, 测试特征
        
        返回：
        - predictions: 预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit()方法")
        
        # 数据预处理
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_processed = self.scaler.transform(X)
        
        # 预测
        predictions = self.model.predict(X_processed)
        
        # 分类任务需要逆转换标签
        if self.task_type == 'classification':
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X):
        """
        预测概率（仅分类任务）
        
        基于论文中的置信度计算方法：
        "利用多数投票的结果数占总投票数的比例作为置信度"
        
        参数：
        - X: array-like, 测试特征
        
        返回：
        - probabilities: 预测概率
        """
        if self.task_type != 'classification':
            raise ValueError("概率预测仅适用于分类任务")
        
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit()方法")
        
        # 数据预处理
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_processed = self.scaler.transform(X)
        
        # 预测概率
        probabilities = self.model.predict_proba(X_processed)
        
        return probabilities
    
    def evaluate(self, X_test, y_test):
        """
        模型评估
        
        基于论文中使用的评估指标：
        - 分类任务：精确率、召回率、F1分数、准确率
        - 回归任务：MAE、MSE、RMSE、R²
        
        参数：
        - X_test: array-like, 测试特征
        - y_test: array-like, 测试标签
        
        返回：
        - metrics: dict, 评估指标
        """
        predictions = self.predict(X_test)
        
        if self.task_type == 'classification':
            # 分类指标
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print("=== 分类模型评估结果 ===")
            print(f"准确率 (Accuracy): {accuracy:.4f}")
            print(f"精确率 (Precision): {precision:.4f}")
            print(f"召回率 (Recall): {recall:.4f}")
            print(f"F1分数: {f1:.4f}")
        
        else:
            # 回归指标（论文中使用的指标）
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            print("=== 回归模型评估结果 ===")
            print(f"平均绝对误差 (MAE): {mae:.4f}")
            print(f"均方误差 (MSE): {mse:.4f}")
            print(f"均方根误差 (RMSE): {rmse:.4f}")
            print(f"决定系数 (R²): {r2:.4f}")
        
        return metrics


def demo_classification():
    """分类任务演示"""
    print("=" * 50)
    print("随机森林分类任务演示")
    print("=" * 50)
    
    # 生成模拟数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_classes=2, weights=[0.77, 0.23],  # 模拟类别不平衡
        random_state=42
    )
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 创建和训练模型
    rf_clf = RandomForestTemplate(
        task_type='classification',
        n_estimators=100,
        max_depth=7,
        random_state=42
    )
    
    # 训练
    rf_clf.fit(X_train, y_train)
    
    # 预测
    predictions = rf_clf.predict(X_test)
    probabilities = rf_clf.predict_proba(X_test[:5])
    
    print("\n前5个样本的预测结果:")
    for i in range(5):
        print(f"样本{i+1}: 预测类别={predictions[i]}, "
              f"置信度=[{probabilities[i][0]:.3f}, {probabilities[i][1]:.3f}]")
    
    # 评估
    print("\n模型评估:")
    metrics = rf_clf.evaluate(X_test, y_test)
    
    return rf_clf, metrics


def demo_regression():
    """回归任务演示"""
    print("=" * 50)
    print("随机森林回归任务演示")
    print("=" * 50)
    
    # 生成模拟数据
    X, y = make_regression(
        n_samples=800, n_features=15, n_informative=10,
        noise=0.1, random_state=42
    )
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 创建和训练模型
    rf_reg = RandomForestTemplate(
        task_type='regression',
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    
    # 训练
    rf_reg.fit(X_train, y_train)
    
    # 预测
    predictions = rf_reg.predict(X_test)
    
    print(f"\n前5个样本的预测结果:")
    for i in range(5):
        print(f"样本{i+1}: 真实值={y_test[i]:.3f}, 预测值={predictions[i]:.3f}")
    
    # 评估
    print("\n模型评估:")
    metrics = rf_reg.evaluate(X_test, y_test)
    
    return rf_reg, metrics


if __name__ == "__main__":
    """主函数：运行演示程序"""
    print("随机森林算法精简模板演示")
    print("基于《出血性脑卒中临床智能诊疗建模》论文实现")
    print("=" * 60)
    
    # 演示1：分类任务
    try:
        rf_clf, clf_metrics = demo_classification()
        print("\n✅ 分类任务演示完成")
    except Exception as e:
        print(f"❌ 分类任务演示失败: {e}")
    
    print("\n" + "=" * 60)
    
    # 演示2：回归任务
    try:
        rf_reg, reg_metrics = demo_regression()
        print("\n✅ 回归任务演示完成")
    except Exception as e:
        print(f"❌ 回归任务演示失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 所有演示完成！")
    
    # 使用说明
    print("\n📖 使用说明:")
    print("1. 分类任务: RandomForestTemplate(task_type='classification')")
    print("2. 回归任务: RandomForestTemplate(task_type='regression')")
    print("3. 核心方法: fit(), predict(), evaluate()")
    print("4. 分类额外: predict_proba() - 概率预测") 