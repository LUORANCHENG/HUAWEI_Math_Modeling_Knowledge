#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线性回归通用模板
基于论文《出血性脑卒中临床智能诊疗建模》中的线性回归实现
包含训练、预测、评估三个核心部分
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False # 解决负号显示问题

class LinearRegressionTemplate:
    """
    线性回归通用模板类
    
    功能：
    1. 训练线性回归模型
    2. 进行预测
    3. 模型评估（MAE, MSE, RMSE, R²）
    """
    
    def __init__(self, normalize=True):
        """
        初始化模型
        
        Args:
            normalize (bool): 是否进行数据标准化，默认True
        """
        self.model = LinearRegression()
        self.scaler = StandardScaler() if normalize else None
        self.normalize = normalize
        self.is_trained = False
        
    def train(self, X, y):
        """
        训练线性回归模型
        
        Args:
            X (array-like): 输入特征，shape为(n_samples, n_features)
            y (array-like): 目标变量，shape为(n_samples,)
            
        Returns:
            dict: 训练结果信息
        """
        try:
            # 确保输入为numpy数组
            X = np.array(X)
            y = np.array(y)
            
            # 处理一维输入
            if X.ndim == 1:
                X = X.reshape(-1, 1)
                
            # 数据标准化
            if self.normalize:
                X = self.scaler.fit_transform(X)
            
            # 训练模型
            self.model.fit(X, y)
            self.is_trained = True
            
            # 计算训练集上的性能
            y_pred = self.model.predict(X)
            train_score = self.model.score(X, y)
            
            result = {
                'status': 'success',
                'message': '模型训练完成',
                'train_samples': len(X),
                'features': X.shape[1],
                'train_r2': train_score,
                'coefficients': self.model.coef_,
                'intercept': self.model.intercept_
            }
            
            print(f"训练完成! 样本数：{len(X)}, 特征数：{X.shape[1]}, 训练R²：{train_score:.4f}")
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'训练失败: {str(e)}'
            }
    
    def predict(self, X):
        """
        使用训练好的模型进行预测
        
        Args:
            X (array-like): 待预测的输入特征
            
        Returns:
            array: 预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
            
        try:
            # 确保输入为numpy数组
            X = np.array(X)
            
            # 处理一维输入
            if X.ndim == 1:
                X = X.reshape(-1, 1)
                
            # 数据标准化
            if self.normalize:
                X = self.scaler.transform(X)
            
            # 预测
            predictions = self.model.predict(X)
            return predictions
            
        except Exception as e:
            raise ValueError(f"预测失败: {str(e)}")
    
    def evaluate(self, X, y, detailed=True):
        """
        评估模型性能
        
        Args:
            X (array-like): 测试集输入特征
            y (array-like): 测试集真实标签
            detailed (bool): 是否返回详细评估信息
            
        Returns:
            dict: 评估指标（MAE, MSE, RMSE, R²）
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
            
        try:
            # 获取预测值
            y_pred = self.predict(X)
            y_true = np.array(y)
            
            # 计算评估指标（论文中使用的指标）
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # 计算残差
            residuals = y_true - y_pred
            
            results = {
                'MAE': mae,
                'MSE': mse, 
                'RMSE': rmse,
                'R²': r2,
                'samples': len(y_true)
            }
            
            if detailed:
                results.update({
                    'residuals': residuals,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'mean_residual': np.mean(residuals),
                    'std_residual': np.std(residuals)
                })
            
            # 打印评估结果
            print("\n=== 模型评估结果 ===")
            print(f"样本数量: {len(y_true)}")
            print(f"平均绝对误差 (MAE): {mae:.4f}")
            print(f"均方误差 (MSE): {mse:.4f}")
            print(f"均方根误差 (RMSE): {rmse:.4f}")
            print(f"决定系数 (R²): {r2:.4f}")
            
            # R²解释
            if r2 > 0.4:
                print("✓ 拟合效果较好 (R² > 0.4)")
            else:
                print("✗ 拟合效果较差 (R² ≤ 0.4)")
                
            return results
            
        except Exception as e:
            raise ValueError(f"评估失败: {str(e)}")
    
    def plot_results(self, X, y, title="线性回归结果"):
        """
        可视化预测结果
        
        Args:
            X (array-like): 输入特征
            y (array-like): 真实标签  
            title (str): 图表标题
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
            
        try:
            y_pred = self.predict(X)
            
            plt.figure(figsize=(12, 4))
            
            # 子图1：预测值vs真实值
            plt.subplot(1, 2, 1)
            plt.scatter(y, y_pred, alpha=0.6)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.title('预测值 vs 真实值')
            plt.grid(True, alpha=0.3)
            
            # 子图2：残差图
            plt.subplot(1, 2, 2)
            residuals = y - y_pred
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('预测值')
            plt.ylabel('残差')
            plt.title('残差图')
            plt.grid(True, alpha=0.3)
            
            plt.suptitle(title)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"绘图失败: {str(e)}")


def demo_usage():
    """
    演示线性回归模板的使用方法
    """
    print("=== 线性回归通用模板演示 ===\n")
    
    # 1. 生成示例数据
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 1)
    y = 3 * X.flatten() + 2 + np.random.randn(n_samples) * 0.5
    
    print("1. 生成示例数据")
    print(f"   样本数: {n_samples}, 特征数: 1")
    
    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. 创建并训练模型
    print("\n2. 训练线性回归模型")
    lr_model = LinearRegressionTemplate(normalize=True)
    train_result = lr_model.train(X_train, y_train)
    
    # 4. 进行预测
    print("\n3. 进行预测")
    y_pred = lr_model.predict(X_test)
    print(f"   预测样本数: {len(y_pred)}")
    
    # 5. 评估模型
    print("\n4. 评估模型性能")
    eval_result = lr_model.evaluate(X_test, y_test, detailed=True)
    
    # 6. 可视化结果
    print("\n5. 可视化结果")
    lr_model.plot_results(X_test, y_test, "线性回归演示结果")
    
    return lr_model, eval_result


if __name__ == "__main__":
    # 运行演示
    model, results = demo_usage()
    
    print("\n=== 使用说明 ===")
    print("1. 创建模型: model = LinearRegressionTemplate()")
    print("2. 训练模型: model.train(X_train, y_train)")
    print("3. 进行预测: predictions = model.predict(X_test)")
    print("4. 评估模型: results = model.evaluate(X_test, y_test)")
    print("5. 可视化: model.plot_results(X_test, y_test)") 