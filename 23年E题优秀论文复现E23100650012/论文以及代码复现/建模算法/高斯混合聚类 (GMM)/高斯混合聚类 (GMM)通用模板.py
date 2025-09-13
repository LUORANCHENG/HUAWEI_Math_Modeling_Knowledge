#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高斯混合聚类 (GMM) 通用模板
基于论文：出血性脑卒中临床智能诊疗建模
作者：天津师范大学 邢雅媛、张庆薇、李祺
参赛队号：23100650012

高斯混合聚类采用概率模型来表达聚类原型。
其基本思想是用多个高斯分布函数去近似任意形状的概率分布。
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False # 解决负号显示问题

class GMMClustering:
    """
    高斯混合聚类通用模板
    
    基于论文中的描述：
    - 高斯混合聚类就是由多个单高斯密度分布组成的，线性加和即为高斯混合聚类的概率密度函数
    - 将待聚类的数据点看成是分布的采样点，通过采样点利用类似极大似然估计的方法估计高斯分布的参数
    """
    
    def __init__(self, n_components=3, covariance_type='full', random_state=42):
        """
        初始化高斯混合聚类模型
        
        参数:
        - n_components: 聚类数量，默认为3（论文中通过肘部法则确定为3）
        - covariance_type: 协方差类型，'full'表示完整协方差矩阵
        - random_state: 随机种子，保证结果可重现
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def train(self, X):
        """
        训练高斯混合聚类模型
        
        参数:
        - X: 训练数据，形状为 (n_samples, n_features)
        
        返回:
        - self: 返回自身，支持链式调用
        """
        print(f"开始训练高斯混合聚类模型...")
        print(f"数据形状: {X.shape}")
        print(f"聚类数量: {self.n_components}")
        
        # 数据标准化（论文中使用了min-max标准化，这里使用StandardScaler）
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建高斯混合模型
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=100,
            tol=1e-3
        )
        
        # 训练模型
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        print(f"模型训练完成!")
        print(f"收敛状态: {'已收敛' if self.model.converged_ else '未收敛'}")
        print(f"迭代次数: {self.model.n_iter_}")
        
        return self
    
    def predict(self, X):
        """
        预测聚类标签
        
        参数:
        - X: 待预测数据，形状为 (n_samples, n_features)
        
        返回:
        - labels: 聚类标签
        - probabilities: 每个样本属于各聚类的概率
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        # 数据标准化
        X_scaled = self.scaler.transform(X)
        
        # 预测聚类标签
        labels = self.model.predict(X_scaled)
        
        # 预测概率（软聚类结果）
        probabilities = self.model.predict_proba(X_scaled)
        
        print(f"预测完成，共{len(set(labels))}个聚类")
        
        return labels, probabilities
    
    def evaluate(self, X, labels=None, true_labels=None):
        """
        评估聚类效果
        
        参数:
        - X: 数据
        - labels: 预测的聚类标签（如果为None，会自动预测）
        - true_labels: 真实标签（如果有的话，用于计算ARI）
        
        返回:
        - evaluation_results: 评估结果字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        # 如果没有提供标签，自动预测
        if labels is None:
            labels, _ = self.predict(X)
        
        # 数据标准化
        X_scaled = self.scaler.transform(X)
        
        evaluation_results = {}
        
        # 1. 平均轮廓系数（论文中使用的评价指标）
        silhouette_avg = silhouette_score(X_scaled, labels)
        evaluation_results['平均轮廓系数'] = silhouette_avg
        
        # 2. BIC和AIC（高斯混合模型特有的评价指标）
        evaluation_results['BIC'] = self.model.bic(X_scaled)
        evaluation_results['AIC'] = self.model.aic(X_scaled)
        
        # 3. 对数似然
        evaluation_results['对数似然'] = self.model.score(X_scaled)
        
        # 4. 如果有真实标签，计算ARI
        if true_labels is not None:
            ari = adjusted_rand_score(true_labels, labels)
            evaluation_results['调整兰德指数(ARI)'] = ari
        
        # 5. 聚类统计信息
        unique_labels, counts = np.unique(labels, return_counts=True)
        evaluation_results['聚类分布'] = dict(zip(unique_labels, counts))
        
        print("=== 聚类评估结果 ===")
        for key, value in evaluation_results.items():
            if key != '聚类分布':
                print(f"{key}: {value:.4f}")
        print(f"聚类分布: {evaluation_results['聚类分布']}")
        
        return evaluation_results
    
    def plot_results(self, X, labels=None, title="高斯混合聚类结果"):
        """
        可视化聚类结果（仅支持2D数据）
        
        参数:
        - X: 数据（必须是2维）
        - labels: 聚类标签
        - title: 图表标题
        """
        if X.shape[1] != 2:
            print("警告: 只支持2维数据的可视化")
            return
        
        if labels is None:
            labels, _ = self.predict(X)
        
        plt.figure(figsize=(10, 8))
        
        # 绘制散点图
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
        
        # 绘制聚类中心
        if self.is_fitted:
            X_scaled = self.scaler.transform(X)
            centers_scaled = self.model.means_
            # 反向标准化得到原始尺度的中心点
            centers = self.scaler.inverse_transform(centers_scaled)
            plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3, label='聚类中心')
        
        plt.colorbar(scatter)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def demo():
    """
    演示高斯混合聚类的使用方法
    使用论文中的设置：3个聚类
    """
    print("=== 高斯混合聚类演示 ===")
    
    # 1. 生成示例数据（模拟论文中的患者数据）
    print("生成示例数据...")
    X, true_labels = make_blobs(n_samples=100, centers=3, n_features=2, 
                               cluster_std=1.5, random_state=42)
    
    # 2. 创建并训练模型
    gmm = GMMClustering(n_components=3, random_state=42)
    gmm.train(X)
    
    # 3. 预测
    labels, probabilities = gmm.predict(X)
    
    # 4. 评估
    results = gmm.evaluate(X, labels, true_labels)
    
    # 5. 可视化
    gmm.plot_results(X, labels, "高斯混合聚类演示结果")
    
    # 6. 显示概率信息（软聚类特性）
    print("\n=== 前5个样本的聚类概率 ===")
    for i in range(5):
        print(f"样本{i+1}: 聚类{labels[i]}, 概率分布: {probabilities[i]}")
    
    return gmm, X, labels, results

if __name__ == "__main__":
    # 运行演示
    model, data, cluster_labels, eval_results = demo()
    
    print("\n=== 模板使用说明 ===")
    print("1. 创建模型: gmm = GMMClustering(n_components=3)")
    print("2. 训练模型: gmm.train(X)")
    print("3. 预测聚类: labels, probabilities = gmm.predict(X)")
    print("4. 评估效果: results = gmm.evaluate(X, labels)")
    print("5. 可视化结果: gmm.plot_results(X, labels)") 