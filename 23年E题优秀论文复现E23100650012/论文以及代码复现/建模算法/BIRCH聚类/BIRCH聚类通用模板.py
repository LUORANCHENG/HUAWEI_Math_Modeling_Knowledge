#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BIRCH聚类算法通用模板
基于23年E题论文中的BIRCH聚类算法实现

论文描述：
- BIRCH算法的主要步骤是建立CF树和CF树的瘦身、离群点的处理
- 该方法的计算速度快，可以识别噪声点
- 在制定聚类类别数k的情况下，叶结点会按照距离远近进行合并，直到叶结点中CF数量等于k
"""

import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False # 解决负号显示问题

class BIRCHClusteringTemplate:
    """
    BIRCH聚类算法通用模板
    
    基于论文中的BIRCH算法描述实现，包含训练、预测、评估三个核心功能
    """
    
    def __init__(self, n_clusters=3, threshold=0.5, branching_factor=50, random_state=42):
        """
        初始化BIRCH聚类模型
        
        Parameters:
        -----------
        n_clusters : int, default=3
            聚类数量，论文中使用肘部法则确定为3
        threshold : float, default=0.5
            CF树中子簇的半径阈值
        branching_factor : int, default=50
            CF树中每个节点的最大分支因子
        random_state : int, default=42
            随机种子，确保结果可重复
        """
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.random_state = random_state
        
        # 初始化模型组件
        self.scaler = StandardScaler()
        self.birch_model = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.is_fitted = False
        
        print(f"初始化BIRCH聚类模型：聚类数={n_clusters}, 阈值={threshold}")
    
    def fit(self, X, feature_names=None):
        """
        训练BIRCH聚类模型
        
        根据论文描述：
        1. 数据标准化处理（论文中使用min-max标准化）
        2. 建立CF树和CF树的瘦身
        3. 离群点的处理
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练数据
        feature_names : list, optional
            特征名称列表
            
        Returns:
        --------
        self : object
            返回自身实例
        """
        print("开始训练BIRCH聚类模型...")
        
        # 数据预处理
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            if feature_names is None:
                feature_names = X.columns.tolist()
        else:
            X_array = np.array(X)
            
        # 数据标准化（论文中提到需要标准化处理）
        X_scaled = self.scaler.fit_transform(X_array)
        
        # 初始化BIRCH模型
        self.birch_model = Birch(
            n_clusters=self.n_clusters,
            threshold=self.threshold,
            branching_factor=self.branching_factor
        )
        
        # 训练模型
        print("构建CF树并进行聚类...")
        self.birch_model.fit(X_scaled)
        
        # 获取聚类标签和聚类中心
        self.labels_ = self.birch_model.labels_
        self.cluster_centers_ = self.birch_model.subcluster_centers_
        
        # 标记模型已训练
        self.is_fitted = True
        
        print(f"训练完成！识别出 {len(np.unique(self.labels_))} 个聚类")
        print(f"各聚类样本数量: {np.bincount(self.labels_)}")
        
        return self
    
    def predict(self, X):
        """
        对新数据进行聚类预测
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            待预测的数据
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            预测的聚类标签
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        print("对新数据进行聚类预测...")
        
        # 数据预处理
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
            
        # 使用训练时的标准化参数
        X_scaled = self.scaler.transform(X_array)
        
        # 预测聚类标签
        labels = self.birch_model.predict(X_scaled)
        
        print(f"预测完成！{len(X_array)} 个样本的聚类分布: {np.bincount(labels)}")
        
        return labels
    
    def evaluate(self, X, labels=None, show_plot=True):
        """
        评估聚类效果
        
        根据论文中使用的评价指标：
        1. 平均轮廓指数 (Average Silhouette Score)
        2. 戴维森-堡丁指数 (Davies-Bouldin Index)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            原始数据
        labels : array-like, optional
            聚类标签，如果为None则使用训练时的标签
        show_plot : bool, default=True
            是否显示可视化结果
            
        Returns:
        --------
        metrics : dict
            评估指标字典
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
            
        print("评估聚类效果...")
        
        # 数据预处理
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
            
        X_scaled = self.scaler.transform(X_array)
        
        # 使用提供的标签或训练时的标签
        if labels is None:
            labels = self.labels_
            
        # 计算评价指标
        silhouette_avg = silhouette_score(X_scaled, labels)
        davies_bouldin_idx = davies_bouldin_score(X_scaled, labels)
        
        # 构建评估结果
        metrics = {
            '平均轮廓系数': silhouette_avg,
            '戴维森-堡丁指数': davies_bouldin_idx,
            '聚类数量': len(np.unique(labels)),
            '样本数量': len(X_array),
            '各聚类样本数': np.bincount(labels).tolist()
        }
        
        # 打印评估结果（对比论文中的结果）
        print("\n=== 聚类效果评估 ===")
        print(f"平均轮廓系数: {silhouette_avg:.6f}")
        print(f"戴维森-堡丁指数: {davies_bouldin_idx:.6f}")
        print(f"聚类数量: {len(np.unique(labels))}")
        print(f"各聚类样本数: {np.bincount(labels)}")
        
        # 与论文结果对比
        print(f"\n论文中BIRCH算法结果对比:")
        print(f"论文轮廓系数: 0.36668680501600653")
        print(f"论文戴维森-堡丁指数: 1.0776920727383936")
        
        # 可视化结果
        if show_plot:
            self._plot_results(X_scaled, labels)
            
        return metrics
    
    def _plot_results(self, X_scaled, labels):
        """
        可视化聚类结果
        
        使用t-SNE降维可视化（论文中使用的方法）
        """
        print("生成聚类可视化图...")
        
        # 使用t-SNE降维（论文中使用的方法）
        if X_scaled.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=30)
            X_tsne = tsne.fit_transform(X_scaled)
        else:
            X_tsne = X_scaled
            
        # 创建图形
        plt.figure(figsize=(12, 5))
        
        # 子图1: t-SNE可视化
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('BIRCH聚类结果 (t-SNE可视化)')
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        
        # 子图2: 聚类分布统计
        plt.subplot(1, 2, 2)
        unique_labels, counts = np.unique(labels, return_counts=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        bars = plt.bar(unique_labels, counts, color=colors, alpha=0.7)
        plt.title('各聚类样本数量分布')
        plt.xlabel('聚类标签')
        plt.ylabel('样本数量')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def get_cluster_info(self):
        """
        获取聚类详细信息
        
        Returns:
        --------
        info : dict
            聚类信息字典
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
            
        info = {
            'n_clusters': self.n_clusters,
            'threshold': self.threshold,
            'branching_factor': self.branching_factor,
            'labels': self.labels_,
            'cluster_centers': self.cluster_centers_,
            'n_samples': len(self.labels_),
            'cluster_sizes': np.bincount(self.labels_).tolist()
        }
        
        return info


def demo_birch_clustering():
    """
    BIRCH聚类算法演示示例
    
    模拟论文中的出血性脑卒中患者聚类场景
    """
    print("=== BIRCH聚类算法演示 ===")
    
    # 生成模拟数据（模拟患者特征）
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # 模拟三个患者亚组的数据
    # 亚组1: 年龄较小，血压正常
    group1 = np.random.normal([0.3, 0.2, 0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.3, 0.2], 0.1, (40, n_features))
    # 亚组2: 年龄较大，高血压
    group2 = np.random.normal([0.8, 0.7, 0.6, 0.7, 0.8, 0.6, 0.7, 0.6, 0.8, 0.7], 0.1, (35, n_features))
    # 亚组3: 中等年龄，有吸烟饮酒史
    group3 = np.random.normal([0.5, 0.4, 0.6, 0.5, 0.4, 0.7, 0.5, 0.8, 0.4, 0.6], 0.1, (25, n_features))
    
    # 合并数据
    X = np.vstack([group1, group2, group3])
    
    # 创建特征名称
    feature_names = [f'特征_{i+1}' for i in range(n_features)]
    
    print(f"生成模拟数据: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
    
    # 1. 训练阶段
    print("\n1. 训练BIRCH聚类模型")
    birch_model = BIRCHClusteringTemplate(n_clusters=3, threshold=0.5)
    birch_model.fit(X, feature_names)
    
    # 2. 预测阶段（使用相同数据进行演示）
    print("\n2. 预测聚类标签")
    predicted_labels = birch_model.predict(X)
    
    # 3. 评估阶段
    print("\n3. 评估聚类效果")
    metrics = birch_model.evaluate(X, show_plot=True)
    
    # 4. 获取聚类信息
    print("\n4. 聚类详细信息")
    cluster_info = birch_model.get_cluster_info()
    print(f"聚类中心数量: {len(cluster_info['cluster_centers'])}")
    print(f"各聚类大小: {cluster_info['cluster_sizes']}")
    
    return birch_model, metrics


if __name__ == "__main__":
    # 运行演示
    model, results = demo_birch_clustering()
    
    print("\n=== 演示完成 ===")
    print("该模板可用于出血性脑卒中患者水肿体积进展模式的聚类分析")
    print("使用方法:")
    print("1. model = BIRCHClusteringTemplate(n_clusters=3)")
    print("2. model.fit(训练数据)")
    print("3. labels = model.predict(新数据)")
    print("4. metrics = model.evaluate(数据)") 