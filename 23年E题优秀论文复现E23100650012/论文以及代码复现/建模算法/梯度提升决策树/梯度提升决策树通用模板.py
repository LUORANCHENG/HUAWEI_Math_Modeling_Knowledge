#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
梯度提升决策树通用模板
基于论文《出血性脑卒中临床智能诊疗建模》中的梯度提升决策树实现
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class GBDTTemplate:
    """
    梯度提升决策树通用模板类
    
    基于论文中的最优参数配置：
    - criterion: gini
    - max_depth: 7
    - min_samples_split: 3
    - min_samples_leaf: 1
    - learning_rate: 0.05
    - random_state: 1
    """
    
    def __init__(self, task_type='classification'):
        """
        初始化梯度提升决策树模型
        
        Args:
            task_type (str): 任务类型，'classification' 或 'regression'
        """
        self.task_type = task_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # 基于论文的最优参数设置
        self.best_params = {
            'criterion': 'friedman_mse',  # 对于GradientBoostingClassifier
            'max_depth': 7,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'learning_rate': 0.05,
            'random_state': 1,
            'n_estimators': 100  # 默认值，可通过网格搜索优化
        }
    
    def train(self, X, y, use_grid_search=False, cv_folds=5):
        """
        训练梯度提升决策树模型
        
        Args:
            X (array-like): 特征矩阵
            y (array-like): 目标变量
            use_grid_search (bool): 是否使用网格搜索优化参数
            cv_folds (int): 交叉验证折数
            
        Returns:
            dict: 训练结果信息
        """
        print("=" * 50)
        print("开始训练梯度提升决策树模型...")
        
        # 数据预处理
        X = np.array(X)
        y = np.array(y)
        
        # 标准化特征（论文中使用了min-max标准化，这里使用StandardScaler）
        X_scaled = self.scaler.fit_transform(X)
        
        # 初始化模型
        if self.task_type == 'classification':
            self.model = GradientBoostingClassifier(**self.best_params)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(**self.best_params)
        
        # 网格搜索优化（基于论文的优化策略）
        if use_grid_search:
            print("执行网格搜索参数优化...")
            param_grid = {
                'max_depth': [5, 7, 9],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 200]
            }
            
            scoring = 'neg_mean_absolute_error' if self.task_type == 'regression' else 'accuracy'
            grid_search = GridSearchCV(
                self.model, param_grid, cv=cv_folds, 
                scoring=scoring, n_jobs=-1, verbose=1
            )
            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_
            print(f"最优参数: {grid_search.best_params_}")
        else:
            # 使用默认最优参数训练
            self.model.fit(X_scaled, y)
        
        # K折交叉验证评估
        if self.task_type == 'classification':
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv_folds, scoring='accuracy')
            print(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        else:
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv_folds, scoring='neg_mean_absolute_error')
            print(f"交叉验证MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.is_fitted = True
        print("模型训练完成！")
        
        return {
            'model': self.model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def predict(self, X, return_proba=False):
        """
        使用训练好的模型进行预测
        
        Args:
            X (array-like): 测试特征矩阵
            return_proba (bool): 是否返回概率预测（仅分类任务）
            
        Returns:
            array: 预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用train方法！")
        
        X = np.array(X)
        X_scaled = self.scaler.transform(X)
        
        if self.task_type == 'classification' and return_proba:
            # 返回正类概率（论文中用于概率预测）
            predictions = self.model.predict_proba(X_scaled)[:, 1]
        else:
            predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def evaluate(self, X_test, y_test, print_results=True):
        """
        评估模型性能
        
        Args:
            X_test (array-like): 测试特征矩阵
            y_test (array-like): 测试目标变量
            print_results (bool): 是否打印结果
            
        Returns:
            dict: 评估指标结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用train方法！")
        
        # 获取预测结果
        y_pred = self.predict(X_test)
        
        results = {}
        
        if self.task_type == 'classification':
            # 分类任务评估指标
            results['accuracy'] = accuracy_score(y_test, y_pred)
            results['precision'] = precision_score(y_test, y_pred, average='weighted')
            results['recall'] = recall_score(y_test, y_pred, average='weighted')
            results['f1'] = f1_score(y_test, y_pred, average='weighted')
            
            # 概率预测评估（论文中的主要评估方式）
            y_pred_proba = self.predict(X_test, return_proba=True)
            results['mae'] = mean_absolute_error(y_test, y_pred_proba)
            results['mse'] = mean_squared_error(y_test, y_pred_proba)
            results['r2'] = r2_score(y_test, y_pred_proba)
            
            if print_results:
                print("=" * 50)
                print("模型评估结果:")
                print("-" * 30)
                print("分类指标:")
                print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
                print(f"精确率 (Precision): {results['precision']:.4f}")
                print(f"召回率 (Recall): {results['recall']:.4f}")
                print(f"F1分数: {results['f1']:.4f}")
                print("-" * 30)
                print("概率预测指标 (论文主要评估方式):")
                print(f"平均绝对误差 (MAE): {results['mae']:.4f}")
                print(f"均方误差 (MSE): {results['mse']:.4f}")
                print(f"决定系数 (R²): {results['r2']:.4f}")
                print("-" * 30)
                print("混淆矩阵:")
                print(confusion_matrix(y_test, y_pred))
                print("-" * 30)
                print("详细分类报告:")
                print(classification_report(y_test, y_pred))
        
        else:
            # 回归任务评估指标
            results['mae'] = mean_absolute_error(y_test, y_pred)
            results['mse'] = mean_squared_error(y_test, y_pred)
            results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            results['r2'] = r2_score(y_test, y_pred)
            
            if print_results:
                print("=" * 50)
                print("模型评估结果:")
                print("-" * 30)
                print(f"平均绝对误差 (MAE): {results['mae']:.4f}")
                print(f"均方误差 (MSE): {results['mse']:.4f}")
                print(f"均方根误差 (RMSE): {results['rmse']:.4f}")
                print(f"决定系数 (R²): {results['r2']:.4f}")
        
        return results

def demo_classification():
    """
    分类任务演示（基于论文的血肿扩张预测场景）
    """
    print("=" * 60)
    print("梯度提升决策树分类任务演示")
    print("场景：出血性脑卒中血肿扩张预测")
    print("=" * 60)
    
    # 生成模拟数据（模拟论文中的特征）
    np.random.seed(42)
    n_samples = 100
    n_features = 47  # 论文中经过特征选择后的特征数
    
    # 模拟特征：年龄、性别、血压等
    X = np.random.randn(n_samples, n_features)
    # 模拟目标：血肿扩张（23个正样本，77个负样本，与论文一致）
    y = np.concatenate([np.ones(23), np.zeros(77)])
    np.random.shuffle(y)
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建模型实例
    gbdt = GBDTTemplate(task_type='classification')
    
    # 训练模型
    train_results = gbdt.train(X_train, y_train, use_grid_search=False)
    
    # 预测
    predictions = gbdt.predict(X_test)
    probabilities = gbdt.predict(X_test, return_proba=True)
    
    print(f"预测结果示例: {predictions[:5]}")
    print(f"概率预测示例: {probabilities[:5]}")
    
    # 评估模型
    evaluation_results = gbdt.evaluate(X_test, y_test)
    
    return gbdt, evaluation_results

def demo_regression():
    """
    回归任务演示（基于论文的水肿体积预测场景）
    """
    print("=" * 60)
    print("梯度提升决策树回归任务演示")
    print("场景：水肿体积随时间进展预测")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X, axis=1) + np.random.normal(0, 0.1, n_samples)
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建模型实例
    gbdt = GBDTTemplate(task_type='regression')
    
    # 训练模型
    train_results = gbdt.train(X_train, y_train, use_grid_search=False)
    
    # 预测
    predictions = gbdt.predict(X_test)
    
    print(f"预测结果示例: {predictions[:5]}")
    
    # 评估模型
    evaluation_results = gbdt.evaluate(X_test, y_test)
    
    return gbdt, evaluation_results

if __name__ == "__main__":
    print("梯度提升决策树通用模板")
    print("基于论文《出血性脑卒中临床智能诊疗建模》")
    print("论文参数配置：MAE=0.0524, MSE=0.00477, R²=0.97429")
    print()
    
    # 运行分类演示
    gbdt_clf, clf_results = demo_classification()
    
    print("\n" + "="*60 + "\n")
    
    # 运行回归演示
    gbdt_reg, reg_results = demo_regression()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("使用说明：")
    print("1. 创建实例：gbdt = GBDTTemplate(task_type='classification')")
    print("2. 训练模型：gbdt.train(X_train, y_train)")
    print("3. 预测结果：predictions = gbdt.predict(X_test)")
    print("4. 评估性能：results = gbdt.evaluate(X_test, y_test)")
    print("="*60) 