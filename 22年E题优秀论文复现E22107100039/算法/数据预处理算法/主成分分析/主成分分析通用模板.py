"""
主要功能：
1. 数据标准化处理
2. 主成分分析降维
3. 特征相关性分析
4. 可视化展示
5. 信息保留率计算

适用场景：
- 高维数据降维（如论文中的168维→12维）
- 特征选择和数据预处理
- 数据可视化和探索性分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class PCATemplate:
    """
    主成分分析通用模板类
    
    基于论文方法实现的PCA工具，包含完整的数据处理流程
    """
    
    def __init__(self, n_components=None, variance_threshold=0.99):
        """
        初始化PCA模板
        
        Parameters:
        -----------
        n_components : int or None
            主成分数量，如果为None则根据方差阈值自动确定
        variance_threshold : float
            信息保留率阈值，默认0.99（保留99%信息，对应论文设置）
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = None
        self.pipeline = None
        self.feature_names = None
        self.pc_names = None
        
    def fit(self, X, feature_names=None):
        """
        训练PCA模型
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            输入数据
        feature_names : list, optional
            特征名称列表
        
        Returns:
        --------
        self : PCATemplate
            返回自身以支持链式调用
        """
        # 转换为DataFrame格式便于处理
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = np.array(X)
            self.feature_names = feature_names or [f'Feature_{i+1}' for i in range(X_array.shape[1])]
        
        print(f"原始数据维度: {X_array.shape}")
        print(f"原始特征数量: {X_array.shape[1]}")
        
        # 步骤1：数据标准化（论文中提到的标准化处理）
        X_scaled = self.scaler.fit_transform(X_array)
        
        # 步骤2：确定主成分数量
        if self.n_components is None:
            # 根据方差阈值自动确定主成分数量（论文方法）
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            cumsum_ratio = np.cumsum(pca_temp.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum_ratio >= self.variance_threshold) + 1
            print(f"根据{self.variance_threshold*100}%方差阈值，自动确定主成分数量: {self.n_components}")
        
        # 步骤3：执行PCA降维
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        
        # 创建完整的处理流水线
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('pca', self.pca)
        ])
        
        # 生成主成分名称
        self.pc_names = [f'PC{i+1}' for i in range(self.n_components)]
        
        # 输出关键信息
        total_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"降维后维度: {self.n_components}")
        print(f"信息保留率: {total_variance:.6f} ({total_variance*100:.4f}%)")
        
        return self
    
    def transform(self, X):
        """
        对数据进行PCA变换
        
        Parameters:
        -----------
        X : array-like
            待变换的数据
            
        Returns:
        --------
        X_transformed : array-like
            变换后的数据
        """
        if self.pipeline is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
            
        return self.pipeline.transform(X_array)
    
    def fit_transform(self, X, feature_names=None):
        """
        训练并变换数据
        
        Parameters:
        -----------
        X : array-like
            输入数据
        feature_names : list, optional
            特征名称列表
            
        Returns:
        --------
        X_transformed : array-like
            变换后的数据
        """
        return self.fit(X, feature_names).transform(X)
    
    def get_components_dataframe(self):
        """
        获取主成分系数DataFrame
        
        Returns:
        --------
        components_df : pd.DataFrame
            主成分系数矩阵，行为原始特征，列为主成分
        """
        if self.pca is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return pd.DataFrame(
            self.pca.components_.T,
            index=self.feature_names,
            columns=self.pc_names
        )
    
    def plot_variance_ratio(self, figsize=(10, 6)):
        """
        绘制方差解释比例图
        
        Parameters:
        -----------
        figsize : tuple
            图形大小
        """
        if self.pca is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        plt.figure(figsize=figsize)
        
        # 子图1：各主成分方差解释比例
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                self.pca.explained_variance_ratio_)
        plt.xlabel('主成分')
        plt.ylabel('方差解释比例')
        plt.title('各主成分方差解释比例')
        plt.grid(True, alpha=0.3)
        
        # 子图2：累积方差解释比例
        plt.subplot(1, 2, 2)
        cumsum_ratio = np.cumsum(self.pca.explained_variance_ratio_)
        plt.plot(range(1, len(cumsum_ratio) + 1), cumsum_ratio, 'b-o')
        plt.axhline(y=self.variance_threshold, color='r', linestyle='--', 
                   label=f'阈值 {self.variance_threshold*100}%')
        plt.xlabel('主成分数量')
        plt.ylabel('累积方差解释比例')
        plt.title('累积方差解释比例')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, X_transformed, figsize=(12, 10)):
        """
        绘制主成分相关性热力图（对应论文中的图5-9和5-10）
        
        Parameters:
        -----------
        X_transformed : array-like
            PCA变换后的数据
        figsize : tuple
            图形大小
        """
        # 转换为DataFrame
        pc_df = pd.DataFrame(X_transformed, columns=self.pc_names)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Pearson相关系数热力图
        pearson_corr = pc_df.corr()
        sns.heatmap(pearson_corr, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title('主成分Pearson相关系数热力图')
        
        # Spearman相关系数热力图
        spearman_corr = pc_df.corr(method='spearman')
        sns.heatmap(spearman_corr, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax2, cbar_kws={'shrink': 0.8})
        ax2.set_title('主成分Spearman相关系数热力图')
        
        plt.tight_layout()
        plt.show()
        
        # 输出相关性统计信息
        print("主成分相关性分析:")
        print(f"Pearson相关系数最大绝对值: {np.abs(pearson_corr.values[np.triu_indices_from(pearson_corr.values, k=1)]).max():.6f}")
        print(f"Spearman相关系数最大绝对值: {np.abs(spearman_corr.values[np.triu_indices_from(spearman_corr.values, k=1)]).max():.6f}")
        print("经过PCA处理后的数据特征相关性非常低，符合论文预期效果")
    
    def get_feature_importance(self, pc_weights=None):
        """
        计算特征重要性（基于主成分贡献度）
        
        Parameters:
        -----------
        pc_weights : array-like, optional
            主成分权重，如果为None则使用方差解释比例作为权重
            
        Returns:
        --------
        importance_df : pd.DataFrame
            特征重要性排序
        """
        if self.pca is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        if pc_weights is None:
            pc_weights = self.pca.explained_variance_ratio_
        
        # 计算加权特征重要性
        components = self.pca.components_
        weighted_components = components * pc_weights.reshape(-1, 1)
        feature_importance = np.sum(np.abs(weighted_components), axis=0)
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def summary(self):
        """
        输出PCA分析摘要信息
        """
        if self.pca is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        print("=" * 50)
        print("主成分分析(PCA)摘要报告")
        print("=" * 50)
        print(f"原始特征数量: {len(self.feature_names)}")
        print(f"主成分数量: {self.n_components}")
        print(f"降维比例: {(1 - self.n_components/len(self.feature_names))*100:.2f}%")
        print(f"信息保留率: {np.sum(self.pca.explained_variance_ratio_)*100:.4f}%")
        print(f"信息压缩率: {len(self.feature_names)/self.n_components:.2f}:1")
        
        print("\n各主成分方差解释比例:")
        for i, ratio in enumerate(self.pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {ratio*100:.4f}%")
        
        print(f"\n前3个主成分累积解释比例: {np.sum(self.pca.explained_variance_ratio_[:3])*100:.4f}%")
        print("=" * 50)


def demo_pca_analysis():
    """
    演示PCA分析的完整流程（模拟论文中的应用场景）
    """
    print("主成分分析演示 - 模拟草原研究数据")
    print("=" * 60)
    
    # 1. 生成模拟数据（模拟论文中168维特征的情况）
    np.random.seed(42)
    n_samples = 132  # 对应论文中的样本数量
    n_features = 168  # 对应论文中的特征数量
    
    # 生成具有一定相关性的高维数据
    true_components = 12  # 真实的有效维度
    latent_factors = np.random.randn(n_samples, true_components)
    noise_level = 0.1
    
    # 生成混合矩阵
    mixing_matrix = np.random.randn(n_features, true_components)
    noise = np.random.randn(n_samples, n_features) * noise_level
    
    # 生成观测数据
    X = latent_factors @ mixing_matrix.T + noise
    
    # 创建特征名称（模拟植物种类+特征类型）
    plants = [f'植物{i+1}' for i in range(42)]  # 42种植物
    features = ['营养苗', '株/丛数', '鲜重(g)', '干重(g)']  # 4种特征类型
    feature_names = [f'{plant}_{feature}' for plant in plants for feature in features]
    
    # 创建DataFrame
    data = pd.DataFrame(X, columns=feature_names)
    
    print(f"模拟数据生成完成:")
    print(f"  样本数量: {data.shape[0]}")
    print(f"  特征数量: {data.shape[1]}")
    print(f"  数据形状: {data.shape}")
    
    # 2. 执行PCA分析
    print("\n开始PCA分析...")
    pca_model = PCATemplate(variance_threshold=0.99)  # 对应论文中的99%设置
    X_transformed = pca_model.fit_transform(data)
    
    # 3. 输出分析结果
    pca_model.summary()
    
    # 4. 可视化分析
    print("\n生成可视化图表...")
    pca_model.plot_variance_ratio()
    pca_model.plot_correlation_heatmap(X_transformed)
    
    # 5. 特征重要性分析
    print("\n特征重要性分析（前10个重要特征）:")
    importance = pca_model.get_feature_importance()
    print(importance.head(10))
    
    # 6. 获取主成分系数
    print("\n主成分系数矩阵（前5个特征，前3个主成分）:")
    components_df = pca_model.get_components_dataframe()
    print(components_df.head().iloc[:, :3])
    
    return pca_model, data, X_transformed


def quick_pca(X, n_components=None, variance_threshold=0.99, plot=True):
    """
    快速PCA分析函数（一行代码完成PCA）
    
    Parameters:
    -----------
    X : array-like or DataFrame
        输入数据
    n_components : int or None
        主成分数量
    variance_threshold : float
        方差阈值
    plot : bool
        是否绘制图表
        
    Returns:
    --------
    X_transformed : array-like
        变换后的数据
    pca_model : PCATemplate
        训练好的PCA模型
    """
    pca_model = PCATemplate(n_components=n_components, 
                           variance_threshold=variance_threshold)
    X_transformed = pca_model.fit_transform(X)
    
    if plot:
        pca_model.plot_variance_ratio()
        pca_model.plot_correlation_heatmap(X_transformed)
    
    pca_model.summary()
    return X_transformed, pca_model


if __name__ == "__main__":
    # 运行演示
    demo_pca_analysis()
    
    print("\n" + "="*60)
    print("模板使用说明:")
    print("="*60)
    print("1. 基础使用:")
    print("   pca = PCATemplate(variance_threshold=0.99)")
    print("   X_transformed = pca.fit_transform(data)")
    print("   pca.summary()")
    print("")
    print("2. 快速使用:")
    print("   X_transformed, pca_model = quick_pca(data)")
    print("")
    print("3. 自定义主成分数量:")
    print("   pca = PCATemplate(n_components=12)")
    print("   X_transformed = pca.fit_transform(data)")
    print("")
    print("4. 可视化分析:")
    print("   pca.plot_variance_ratio()")
    print("   pca.plot_correlation_heatmap(X_transformed)")
    print("")
    print("5. 特征重要性:")
    print("   importance = pca.get_feature_importance()")
    print("="*60) 