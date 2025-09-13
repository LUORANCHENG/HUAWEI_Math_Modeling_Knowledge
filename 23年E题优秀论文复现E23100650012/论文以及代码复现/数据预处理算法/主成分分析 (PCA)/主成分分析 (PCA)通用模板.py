"""
主成分分析 (PCA) 通用模板
基于23年E题优秀论文实现

主要功能：
1. 数据标准化处理 (min-max标准化)
2. 常量特征检测与删除
3. PCA降维分析
4. 主成分贡献率计算
5. 关键特征识别

适用场景：
- 高维度多源异质数据降维
- 医学数据特征提取
- 影响因素分析
- 数据可视化预处理
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional

plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False # 解决负号显示问题

class SimplePCA:
    """
    最精简的主成分分析实现
    基于论文中的方法和参数设置
    """
    
    def __init__(self, variance_threshold: float = 0.0, pca_threshold: float = 0.9):
        """
        初始化PCA分析器
        
        Args:
            variance_threshold: 常量特征过滤阈值，方差小于此值的特征将被删除
            pca_threshold: 主成分累计贡献率阈值，默认0.9（保留90%信息）
        """
        self.variance_threshold = variance_threshold
        self.pca_threshold = pca_threshold
        self.scaler = MinMaxScaler()
        self.pca = None
        self.removed_features = []
        self.feature_names = None
        self.n_components = None
        
    def preprocess_data(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        数据预处理：标准化 + 常量特征删除
        
        Args:
            X: 输入数据矩阵 (n_samples, n_features)
            feature_names: 特征名称列表
            
        Returns:
            处理后的数据矩阵
        """
        print("=" * 50)
        print("开始数据预处理...")
        
        # 保存特征名称
        if feature_names is not None:
            self.feature_names = feature_names.copy()
        else:
            self.feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        print(f"原始数据形状: {X.shape}")
        
        # 1. 检测并删除常量特征（方差为0或接近0的特征）
        variances = np.var(X, axis=0)
        constant_features = np.where(variances <= self.variance_threshold)[0]
        
        if len(constant_features) > 0:
            print(f"检测到 {len(constant_features)} 个常量特征:")
            for idx in constant_features:
                print(f"  - {self.feature_names[idx]}: 方差 = {variances[idx]:.6f}")
            
            # 记录被删除的特征
            self.removed_features = [self.feature_names[i] for i in constant_features]
            
            # 删除常量特征
            X_filtered = np.delete(X, constant_features, axis=1)
            self.feature_names = [name for i, name in enumerate(self.feature_names) 
                                 if i not in constant_features]
        else:
            X_filtered = X.copy()
            print("未检测到常量特征")
        
        print(f"过滤后数据形状: {X_filtered.shape}")
        
        # 2. Min-Max标准化（论文中使用的方法）
        print("执行Min-Max标准化...")
        X_scaled = self.scaler.fit_transform(X_filtered)
        
        print("数据预处理完成！")
        return X_scaled
    
    def fit_pca(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        执行PCA分析
        
        Args:
            X_scaled: 预处理后的数据
            
        Returns:
            transformed_data: 降维后的数据
            pca_info: PCA分析信息
        """
        print("=" * 50)
        print("开始PCA分析...")
        
        # 1. 计算所有主成分
        pca_full = PCA()
        pca_full.fit(X_scaled)
        
        # 2. 根据累计贡献率确定主成分数量
        cumsum_ratio = np.cumsum(pca_full.explained_variance_ratio_)
        self.n_components = np.argmax(cumsum_ratio >= self.pca_threshold) + 1
        
        print(f"根据累计贡献率阈值 {self.pca_threshold}，选择 {self.n_components} 个主成分")
        
        # 3. 使用确定的主成分数量重新拟合PCA
        self.pca = PCA(n_components=self.n_components)
        transformed_data = self.pca.fit_transform(X_scaled)
        
        # 4. 计算PCA信息
        eigenvalues = self.pca.explained_variance_
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_ratio = np.cumsum(explained_variance_ratio)
        
        pca_info = {
            'eigenvalues': eigenvalues,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_ratio': cumulative_ratio,
            'components': self.pca.components_,
            'n_components': self.n_components
        }
        
        # 5. 打印结果表格（模仿论文表14格式）
        print("\n主成分分析结果:")
        print("-" * 50)
        print(f"{'主成分':<8} {'特征值':<12} {'贡献度':<12} {'累计贡献度':<12}")
        print("-" * 50)
        
        for i in range(self.n_components):
            print(f"PC{i+1:<7} {eigenvalues[i]:<12.6f} {explained_variance_ratio[i]:<12.6f} {cumulative_ratio[i]:<12.6f}")
        
        print("-" * 50)
        print(f"总累计贡献度: {cumulative_ratio[-1]:.6f}")
        
        return transformed_data, pca_info
    
    def get_feature_importance(self, top_k: int = 7) -> Dict[str, List]:
        """
        获取特征重要性分析（模仿论文表15）
        
        Args:
            top_k: 返回前k个最重要的特征
            
        Returns:
            特征重要性字典
        """
        if self.pca is None:
            raise ValueError("请先执行fit_pca方法")
        
        print("=" * 50)
        print("特征重要性分析...")
        
        # 计算每个特征在所有主成分中的总贡献度
        components = np.abs(self.pca.components_)
        
        # 加权计算特征重要性（按主成分的解释方差比例加权）
        feature_importance = np.zeros(len(self.feature_names))
        for i, weight in enumerate(self.pca.explained_variance_ratio_):
            feature_importance += components[i] * weight
        
        # 获取最重要的特征
        top_indices = np.argsort(feature_importance)[::-1][:top_k]
        top_features = [self.feature_names[i] for i in top_indices]
        top_scores = feature_importance[top_indices]
        
        print(f"\n贡献度最大的前{top_k}个特征:")
        print("-" * 60)
        print(f"{'排名':<4} {'特征名称':<25} {'重要性得分':<15}")
        print("-" * 60)
        
        for rank, (feature, score) in enumerate(zip(top_features, top_scores), 1):
            print(f"{rank:<4} {feature:<25} {score:<15.6f}")
        
        return {
            'feature_names': top_features,
            'importance_scores': top_scores,
            'all_importance': feature_importance
        }
    
    def plot_results(self, pca_info: Dict, save_path: Optional[str] = None):
        """
        绘制PCA分析结果图表
        
        Args:
            pca_info: PCA分析信息
            save_path: 保存路径
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 特征值碎石图
        ax1.plot(range(1, len(pca_info['eigenvalues']) + 1), 
                pca_info['eigenvalues'], 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('主成分')
        ax1.set_ylabel('特征值')
        ax1.set_title('特征值碎石图')
        ax1.grid(True, alpha=0.3)
        
        # 2. 贡献率图
        ax2.bar(range(1, len(pca_info['explained_variance_ratio']) + 1), 
               pca_info['explained_variance_ratio'], alpha=0.7, color='skyblue')
        ax2.set_xlabel('主成分')
        ax2.set_ylabel('贡献率')
        ax2.set_title('各主成分贡献率')
        ax2.grid(True, alpha=0.3)
        
        # 3. 累计贡献率图
        ax3.plot(range(1, len(pca_info['cumulative_ratio']) + 1), 
                pca_info['cumulative_ratio'], 'ro-', linewidth=2, markersize=8)
        ax3.axhline(y=self.pca_threshold, color='g', linestyle='--', 
                   label=f'阈值 ({self.pca_threshold})')
        ax3.set_xlabel('主成分')
        ax3.set_ylabel('累计贡献率')
        ax3.set_title('累计贡献率图')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 主成分载荷图（前两个主成分）
        if len(pca_info['components']) >= 2:
            pc1_loadings = pca_info['components'][0]
            pc2_loadings = pca_info['components'][1]
            
            ax4.scatter(pc1_loadings, pc2_loadings, alpha=0.7, s=60)
            ax4.set_xlabel(f'PC1 ({pca_info["explained_variance_ratio"][0]:.3f})')
            ax4.set_ylabel(f'PC2 ({pca_info["explained_variance_ratio"][1]:.3f})')
            ax4.set_title('主成分载荷图')
            ax4.grid(True, alpha=0.3)
            
            # 添加原点线
            ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()

def demo_medical_data():
    """
    演示医学数据PCA分析（模拟论文场景）
    """
    print("=" * 60)
    print("医学数据PCA分析演示")
    print("模拟论文中的血肿扩张影响因素分析")
    print("=" * 60)
    
    # 模拟医学数据
    np.random.seed(42)
    n_samples = 100
    
    # 模拟不同类型的医学特征
    age = np.random.normal(62, 15, n_samples)  # 年龄
    systolic_bp = np.random.normal(145, 20, n_samples)  # 收缩压
    diastolic_bp = np.random.normal(85, 15, n_samples)  # 舒张压
    hm_volume = np.random.lognormal(8, 1, n_samples)  # 血肿体积
    ed_volume = np.random.lognormal(7.5, 1.2, n_samples)  # 水肿体积
    
    # 脑区位置比例（模拟论文中的位置特征）
    hm_mca_r = np.random.beta(2, 5, n_samples)  # 血肿-大脑中动脉右侧
    hm_pons_l = np.random.beta(1, 10, n_samples)  # 血肿-脑桥左侧
    ed_mca_r = np.random.beta(2, 5, n_samples)  # 水肿-大脑中动脉右侧
    ed_mca_l = np.random.beta(2, 5, n_samples)  # 水肿-大脑中动脉左侧
    ed_aca_l = np.random.beta(1.5, 6, n_samples)  # 水肿-大脑前动脉左侧
    
    # 添加一些常量特征（模拟论文中提到的问题）
    drinking_history = np.zeros(n_samples)  # 饮酒史（全为0）
    nutrition_therapy = np.ones(n_samples)  # 营养神经治疗（全为1）
    
    # 组合数据
    X = np.column_stack([
        age, systolic_bp, diastolic_bp, hm_volume, ed_volume,
        hm_mca_r, hm_pons_l, ed_mca_r, ed_mca_l, ed_aca_l,
        drinking_history, nutrition_therapy
    ])
    
    feature_names = [
        '年龄', '收缩压', '舒张压', '血肿体积', '水肿体积',
        'HM_MCA_R_Ratio', 'HM_Pons_L_Ratio', 'ED_MCA_R_Ratio', 
        'ED_MCA_L_Ratio', 'ED_ACA_L_Ratio',
        '饮酒史', '营养神经治疗'
    ]
    
    # 执行PCA分析
    pca_analyzer = SimplePCA(variance_threshold=0.0, pca_threshold=0.9)
    
    # 数据预处理
    X_processed = pca_analyzer.preprocess_data(X, feature_names)
    
    # PCA分析
    X_transformed, pca_info = pca_analyzer.fit_pca(X_processed)
    
    # 特征重要性分析
    importance_info = pca_analyzer.get_feature_importance(top_k=7)
    
    # 绘制结果
    pca_analyzer.plot_results(pca_info)
    
    return X_transformed, pca_info, importance_info

def demo_general_usage():
    """
    通用数据PCA分析演示
    """
    print("=" * 60)
    print("通用数据PCA分析演示")
    print("=" * 60)
    
    # 生成示例数据
    np.random.seed(42)
    n_samples, n_features = 150, 10
    
    # 创建相关性数据
    X = np.random.randn(n_samples, n_features)
    # 添加一些特征间的相关性
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)
    X[:, 2] = X[:, 0] - 0.3 * X[:, 1] + 0.4 * np.random.randn(n_samples)
    
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    # 执行PCA分析
    pca_analyzer = SimplePCA(pca_threshold=0.85)
    
    X_processed = pca_analyzer.preprocess_data(X, feature_names)
    X_transformed, pca_info = pca_analyzer.fit_pca(X_processed)
    importance_info = pca_analyzer.get_feature_importance()
    
    pca_analyzer.plot_results(pca_info)
    
    return X_transformed, pca_info

if __name__ == "__main__":
    # 运行医学数据演示（基于论文场景）
    print("运行医学数据分析演示...")
    demo_medical_data()
    
    print("\n" + "="*80 + "\n")
    
    # 运行通用数据演示
    print("运行通用数据分析演示...")
    demo_general_usage() 