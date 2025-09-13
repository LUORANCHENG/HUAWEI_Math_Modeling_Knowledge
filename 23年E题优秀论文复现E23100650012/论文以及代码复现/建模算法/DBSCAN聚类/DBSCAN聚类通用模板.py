"""
DBSCAN聚类算法通用模板
适用于医疗数据聚类分析，特别是患者水肿体积进展模式的个体差异探究
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False # 解决负号显示问题

class DBSCANClusteringTemplate:
    """
    DBSCAN聚类算法通用模板
    
    参数说明：
    - eps: ε-邻域半径，控制邻域大小
    - min_samples: 核心点的最小邻域样本数
    - metric: 距离度量方式，默认欧氏距离
    """
    
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.model = None
        self.scaler = StandardScaler()
        self.labels_ = None
        self.n_clusters_ = None
        self.n_noise_ = None
        
    def train(self, X, feature_names=None):
        """
        训练DBSCAN聚类模型
        
        参数:
        X: 训练数据，形状为 (n_samples, n_features)
        feature_names: 特征名称列表，用于后续分析
        
        返回:
        self: 返回训练后的模型实例
        """
        print("=" * 50)
        print("开始DBSCAN聚类训练...")
        print(f"数据形状: {X.shape}")
        print(f"参数设置: eps={self.eps}, min_samples={self.min_samples}")
        
        # 数据标准化处理（参考论文中的数据预处理步骤）
        X_scaled = self.scaler.fit_transform(X)
        
        # 初始化DBSCAN模型
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric
        )
        
        # 执行聚类
        self.labels_ = self.model.fit_predict(X_scaled)
        
        # 计算聚类统计信息
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.n_noise_ = list(self.labels_).count(-1)
        
        print(f"聚类完成！")
        print(f"发现聚类数量: {self.n_clusters_}")
        print(f"噪声点数量: {self.n_noise_}")
        print("=" * 50)
        
        return self
    
    def predict(self, X_new=None):
        """
        预测新数据的聚类标签
        
        注意：DBSCAN是非参数方法，无法直接预测新样本
        这里返回训练数据的聚类结果，实际应用中需要重新训练
        
        参数:
        X_new: 新数据（可选）
        
        返回:
        labels: 聚类标签数组
        """
        if self.labels_ is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        if X_new is not None:
            print("警告：DBSCAN无法直接预测新样本，建议将新数据加入训练集重新训练")
        
        return self.labels_
    
    def evaluate(self, X, detailed=True):
        """
        评估聚类效果
        
        参数:
        X: 原始数据
        detailed: 是否输出详细评估结果
        
        返回:
        metrics: 包含评估指标的字典
        """
        if self.labels_ is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        print("=" * 50)
        print("开始聚类效果评估...")
        
        # 标准化数据用于评估
        X_scaled = self.scaler.transform(X)
        
        # 计算评估指标
        metrics = {}
        
        # 1. 平均轮廓系数（参考论文表10）
        if self.n_clusters_ > 1:
            silhouette_avg = silhouette_score(X_scaled, self.labels_)
            metrics['平均轮廓系数'] = silhouette_avg
        else:
            metrics['平均轮廓系数'] = -1  # 无法计算
        
        # 2. 戴维森-堡丁指数（参考论文中的评价指标）
        try:
            from sklearn.metrics import davies_bouldin_score
            if self.n_clusters_ > 1:
                db_score = davies_bouldin_score(X_scaled, self.labels_)
                metrics['戴维森-堡丁指数'] = db_score
            else:
                metrics['戴维森-堡丁指数'] = float('inf')
        except ImportError:
            print("警告：无法计算戴维森-堡丁指数，请安装sklearn>=0.20")
            metrics['戴维森-堡丁指数'] = None
        
        # 3. 聚类基本统计
        metrics['聚类数量'] = self.n_clusters_
        metrics['噪声点数量'] = self.n_noise_
        metrics['噪声点比例'] = self.n_noise_ / len(self.labels_)
        
        if detailed:
            print(f"聚类评估结果:")
            print(f"  聚类数量: {metrics['聚类数量']}")
            print(f"  噪声点数量: {metrics['噪声点数量']}")
            print(f"  噪声点比例: {metrics['噪声点比例']:.4f}")
            if metrics['平均轮廓系数'] != -1:
                print(f"  平均轮廓系数: {metrics['平均轮廓系数']:.6f}")
            if metrics['戴维森-堡丁指数'] is not None:
                print(f"  戴维森-堡丁指数: {metrics['戴维森-堡丁指数']:.6f}")
        
        print("=" * 50)
        return metrics
    
    def visualize(self, X, title="DBSCAN聚类结果可视化"):
        """
        可视化聚类结果（参考论文中的t-SNE可视化方法）
        
        参数:
        X: 原始数据
        title: 图标题
        """
        if self.labels_ is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        print("正在生成聚类可视化...")
        
        # 使用t-SNE降维（参考论文5.4.2节）
        X_scaled = self.scaler.transform(X)
        
        if X_scaled.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X_scaled.shape[0]-1))
            X_tsne = tsne.fit_transform(X_scaled)
        else:
            X_tsne = X_scaled
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        
        # 为每个聚类分配颜色
        unique_labels = set(self.labels_)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # 噪声点用黑色表示
                col = 'black'
                marker = 'x'
                label = '噪声点'
            else:
                marker = 'o'
                label = f'聚类 {k}'
            
            class_member_mask = (self.labels_ == k)
            xy = X_tsne[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
                       s=50, alpha=0.7, label=label)
        
        plt.title(title, fontsize=14)
        plt.xlabel('t-SNE 维度 1', fontsize=12)
        plt.ylabel('t-SNE 维度 2', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_cluster_info(self, X, feature_names=None):
        """
        获取各聚类的详细信息
        
        参数:
        X: 原始数据
        feature_names: 特征名称列表
        
        返回:
        cluster_info: 各聚类的统计信息
        """
        if self.labels_ is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        cluster_info = {}
        
        for cluster_id in set(self.labels_):
            if cluster_id == -1:
                continue  # 跳过噪声点
            
            mask = self.labels_ == cluster_id
            cluster_data = X[mask]
            
            info = {
                '样本数量': np.sum(mask),
                '样本比例': np.sum(mask) / len(self.labels_),
                '特征均值': np.mean(cluster_data, axis=0),
                '特征标准差': np.std(cluster_data, axis=0)
            }
            
            if feature_names is not None:
                info['特征名称'] = feature_names
            
            cluster_info[f'聚类_{cluster_id}'] = info
        
        return cluster_info


def demo_usage():
    """
    演示DBSCAN聚类模板的使用方法
    模拟论文中患者水肿体积数据的聚类分析
    """
    print("DBSCAN聚类算法演示")
    print("=" * 60)
    
    # 生成模拟医疗数据（模拟论文中的患者特征数据）
    np.random.seed(42)
    n_samples = 100
    
    # 模拟患者特征：年龄、血压、病史等（参考论文表1数据）
    age = np.random.normal(65, 15, n_samples)  # 年龄
    blood_pressure_sys = np.random.normal(140, 20, n_samples)  # 收缩压
    blood_pressure_dia = np.random.normal(90, 10, n_samples)   # 舒张压
    edema_volume = np.random.lognormal(8, 1, n_samples)        # 水肿体积
    
    # 组合特征数据
    X = np.column_stack([age, blood_pressure_sys, blood_pressure_dia, edema_volume])
    feature_names = ['年龄', '收缩压', '舒张压', '水肿体积']
    
    print(f"生成模拟数据: {X.shape[0]} 个患者, {X.shape[1]} 个特征")
    
    # 1. 训练阶段
    dbscan_model = DBSCANClusteringTemplate(eps=0.8, min_samples=3)
    dbscan_model.train(X, feature_names)
    
    # 2. 预测阶段
    labels = dbscan_model.predict()
    print(f"预测结果: {len(set(labels))} 个聚类")
    
    # 3. 评估阶段
    metrics = dbscan_model.evaluate(X, detailed=True)
    
    # 4. 可视化
    dbscan_model.visualize(X, "患者聚类结果 - DBSCAN算法")
    
    # 5. 聚类信息分析
    cluster_info = dbscan_model.get_cluster_info(X, feature_names)
    
    print("\n各聚类详细信息:")
    for cluster_name, info in cluster_info.items():
        print(f"\n{cluster_name}:")
        print(f"  样本数量: {info['样本数量']}")
        print(f"  样本比例: {info['样本比例']:.3f}")
        for i, feature_name in enumerate(feature_names):
            print(f"  {feature_name}: 均值={info['特征均值'][i]:.2f}, "
                  f"标准差={info['特征标准差'][i]:.2f}")


if __name__ == "__main__":
    demo_usage() 