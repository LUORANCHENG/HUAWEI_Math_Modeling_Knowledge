#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
谱聚类通用模板
基于23年数学建模竞赛E题论文实现

论文中的应用场景：
- 将出血性脑卒中患者按个体差异分为3个亚组
- 使用平均轮廓系数和戴维森-堡丁指数进行评估
- 谱聚类获得最佳聚类效果（轮廓系数0.4149，DB指数0.8818）
"""

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False # 解决负号显示问题

class SpectralClusteringTemplate:
    """
    谱聚类通用模板类
    
    基于图论的聚类方法，适用于处理非凸形状的聚类问题
    特别适合医学数据中患者个体差异的分析
    """
    
    def __init__(self, n_clusters=3, gamma=1.0, affinity='rbf', random_state=42):
        """
        初始化谱聚类模型
        
        Parameters:
        -----------
        n_clusters : int, default=3
            聚类数目，论文中通过肘部法则确定为3
        gamma : float, default=1.0
            RBF核的参数
        affinity : str, default='rbf'
            构建相似矩阵的方法
        random_state : int, default=42
            随机种子，确保结果可重现
        """
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.affinity = affinity
        self.random_state = random_state
        
        # 初始化模型和预处理器
        self.model = SpectralClustering(
            n_clusters=n_clusters,
            gamma=gamma,
            affinity=affinity,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        
        # 存储结果
        self.labels_ = None
        self.scaled_data_ = None
        self.is_fitted = False
        
    def fit(self, X):
        """
        训练谱聚类模型
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练数据
            
        Returns:
        --------
        self : object
            返回自身以支持链式调用
        """
        # 数据标准化（论文中使用min-max标准化）
        print("正在进行数据标准化...")
        self.scaled_data_ = self.scaler.fit_transform(X)
        
        # 执行谱聚类
        print(f"正在执行谱聚类，聚类数目: {self.n_clusters}")
        self.labels_ = self.model.fit_predict(self.scaled_data_)
        
        self.is_fitted = True
        print("谱聚类训练完成！")
        
        return self
    
    def predict(self, X=None):
        """
        预测聚类标签
        
        注意：谱聚类是一种无监督学习方法，通常不支持对新数据的预测
        这里返回训练数据的聚类结果
        
        Parameters:
        -----------
        X : array-like, optional
            新数据（谱聚类通常不支持新数据预测）
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            聚类标签
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        if X is not None:
            print("警告：谱聚类通常不支持对新数据的预测，返回训练数据的聚类结果")
            
        return self.labels_
    
    def evaluate(self, X=None, labels=None, verbose=True):
        """
        评估聚类效果
        
        使用论文中的两个关键指标：
        1. 平均轮廓系数 (Silhouette Score) - 越高越好
        2. 戴维森-堡丁指数 (Davies-Bouldin Index) - 越低越好
        
        Parameters:
        -----------
        X : array-like, optional
            评估数据，默认使用训练数据
        labels : array-like, optional
            聚类标签，默认使用训练结果
        verbose : bool, default=True
            是否打印详细结果
            
        Returns:
        --------
        metrics : dict
            包含评价指标的字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 使用训练数据和结果
        data = self.scaled_data_ if X is None else self.scaler.transform(X)
        cluster_labels = self.labels_ if labels is None else labels
        
        # 计算评价指标
        silhouette_avg = silhouette_score(data, cluster_labels)
        davies_bouldin = davies_bouldin_score(data, cluster_labels)
        
        # 计算每个聚类的大小
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))
        
        metrics = {
            'silhouette_score': silhouette_avg,
            'davies_bouldin_score': davies_bouldin,
            'cluster_sizes': cluster_sizes,
            'n_clusters': len(unique_labels)
        }
        
        if verbose:
            print("\n" + "="*50)
            print("谱聚类评估结果")
            print("="*50)
            print(f"平均轮廓系数: {silhouette_avg:.4f}")
            print(f"戴维森-堡丁指数: {davies_bouldin:.4f}")
            print(f"聚类数目: {len(unique_labels)}")
            print("各聚类大小:")
            for label, size in cluster_sizes.items():
                print(f"  聚类 {label}: {size} 个样本")
            print("\n论文中谱聚类的结果:")
            print("  平均轮廓系数: 0.4149")
            print("  戴维森-堡丁指数: 0.8818")
            print("="*50)
            
        return metrics
    
    def visualize(self, X=None, labels=None, method='tsne', figsize=(10, 8)):
        """
        可视化聚类结果
        
        使用t-SNE降维可视化（论文中使用的方法）
        
        Parameters:
        -----------
        X : array-like, optional
            可视化数据，默认使用训练数据
        labels : array-like, optional
            聚类标签，默认使用训练结果
        method : str, default='tsne'
            降维方法
        figsize : tuple, default=(10, 8)
            图形大小
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 使用训练数据和结果
        data = self.scaled_data_ if X is None else self.scaler.transform(X)
        cluster_labels = self.labels_ if labels is None else labels
        
        # t-SNE降维（论文中使用的可视化方法）
        print("正在进行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=self.random_state)
        data_2d = tsne.fit_transform(data)
        
        # 绘制聚类结果
        plt.figure(figsize=figsize)
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i in range(self.n_clusters):
            mask = cluster_labels == i
            plt.scatter(data_2d[mask, 0], data_2d[mask, 1], 
                       c=colors[i % len(colors)], 
                       label=f'聚类 {i}', 
                       alpha=0.7, s=50)
        
        plt.title('谱聚类结果可视化 (t-SNE)', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE 维度 1', fontsize=12)
        plt.ylabel('t-SNE 维度 2', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("可视化完成！")
    
    def get_cluster_info(self):
        """
        获取聚类详细信息
        
        Returns:
        --------
        info : dict
            聚类详细信息
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        
        info = {
            'n_samples': len(self.labels_),
            'n_clusters': len(unique_labels),
            'cluster_sizes': dict(zip(unique_labels, counts)),
            'labels': self.labels_
        }
        
        return info


def demo_usage():
    """
    演示谱聚类模板的使用方法
    
    模拟论文中的患者数据聚类场景
    """
    print("谱聚类模板演示")
    print("="*50)
    
    # 1. 生成模拟数据（模拟患者个体属性数据）
    np.random.seed(42)
    
    # 模拟3个不同的患者群体
    # 群体1：年轻患者，血压正常
    group1 = np.random.multivariate_normal([0.3, 0.2, 0.1], [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], 61)
    # 群体2：中年患者，血压偏高
    group2 = np.random.multivariate_normal([0.6, 0.7, 0.4], [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], 18)
    # 群体3：老年患者，多种疾病
    group3 = np.random.multivariate_normal([0.8, 0.5, 0.9], [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], 21)
    
    X = np.vstack([group1, group2, group3])
    print(f"生成模拟数据: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
    
    # 2. 训练谱聚类模型
    spectral_model = SpectralClusteringTemplate(n_clusters=3, random_state=42)
    spectral_model.fit(X)
    
    # 3. 预测聚类标签
    labels = spectral_model.predict()
    print(f"预测完成，聚类标签: {np.unique(labels)}")
    
    # 4. 评估聚类效果
    metrics = spectral_model.evaluate()
    
    # 5. 可视化结果
    spectral_model.visualize()
    
    # 6. 获取聚类详细信息
    cluster_info = spectral_model.get_cluster_info()
    print(f"\n聚类详细信息: {cluster_info}")
    
    return spectral_model, X, labels, metrics


if __name__ == "__main__":
    # 运行演示
    model, data, labels, metrics = demo_usage()
    
    print("\n" + "="*50)
    print("谱聚类模板使用完成！")
    print("="*50) 