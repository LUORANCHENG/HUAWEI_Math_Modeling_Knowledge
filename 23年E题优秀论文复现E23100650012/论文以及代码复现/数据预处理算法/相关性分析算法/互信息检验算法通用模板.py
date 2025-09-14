#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
互信息检验算法通用模板（精简版）
基于论文《出血性脑卒中临床智能诊疗建模》中的相关性分析算法实现

互信息检验：用于衡量两个变量之间的相互依赖性，能够捕捉线性和非线性关系

算法原理：
1. 计算联合概率分布和边际概率分布
2. 使用信息论中的互信息公式计算
3. MI(X,Y) = ∑∑ P(x,y) * log(P(x,y) / (P(x)*P(y)))

适用条件：
1. 能够检测非线性关系
2. 适用于离散和连续变量
3. 对数据分布无特殊要求
4. 互信息值为0表示完全独立
"""

import numpy as np
import pandas as pd
import math
from typing import Union, Tuple, List, Optional
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import warnings

warnings.filterwarnings('ignore')


class MutualInformationAnalysis:
    """互信息检验分析器（精简版）"""
    
    def __init__(self, method: str = 'auto', n_bins: int = 10):
        """
        初始化互信息分析器
        
        Args:
            method: 计算方法 ('auto', 'sklearn', 'manual')
            n_bins: 连续变量离散化的分箱数
        """
        self.name = "Mutual Information Analysis"
        self.description = "互信息检验，用于衡量变量间的相互依赖性（线性和非线性）"
        self.method = method
        self.n_bins = n_bins
    
    def calculate_sklearn(self, x: Union[list, np.ndarray], 
                         y: Union[list, np.ndarray], 
                         task_type: str = 'regression') -> float:
        """
        使用sklearn计算互信息（API优先）
        
        Args:
            x: 第一个变量
            y: 第二个变量  
            task_type: 任务类型 ('regression' 或 'classification')
            
        Returns:
            float: 互信息值
        """
        x_array = np.array(x).reshape(-1, 1)
        y_array = np.array(y)
        
        if task_type == 'regression':
            # 连续变量的互信息
            mi_score = mutual_info_regression(x_array, y_array, random_state=42)[0]
        else:
            # 离散变量的互信息
            mi_score = mutual_info_classif(x_array, y_array, random_state=42)[0]
            
        return mi_score
    
    def calculate_discrete(self, x: Union[list, np.ndarray], 
                          y: Union[list, np.ndarray]) -> float:
        """
        计算离散变量的互信息（基于sklearn.metrics）
        
        Args:
            x: 第一个变量（离散）
            y: 第二个变量（离散）
            
        Returns:
            float: 互信息值
        """
        return mutual_info_score(x, y)
    
    def calculate_manual(self, x: Union[list, np.ndarray], 
                        y: Union[list, np.ndarray]) -> float:
        """
        手动计算互信息（基于论文附录实现）
        
        Args:
            x: 第一个变量
            y: 第二个变量
            
        Returns:
            float: 互信息值
        """
        # 转换为numpy数组
        A = np.array(x)
        B = np.array(y)
        
        # 验证长度一致
        if len(A) != len(B):
            raise ValueError("输入序列长度必须相等")
        
        total = len(A)
        A_ids = set(A)
        B_ids = set(B)
        
        # 计算互信息
        MI = 0
        eps = 1.4e-45  # 防止log(0)
        
        for idA in A_ids:
            for idB in B_ids:
                # 找到A中等于idA的位置
                idAOccur = np.where(A == idA)
                # 找到B中等于idB的位置  
                idBOccur = np.where(B == idB)
                # 找到A和B同时满足条件的位置
                idABOccur = np.intersect1d(idAOccur, idBOccur)
                
                # 计算概率
                px = 1.0 * len(idAOccur[0]) / total  # P(X=idA)
                py = 1.0 * len(idBOccur[0]) / total  # P(Y=idB)
                pxy = 1.0 * len(idABOccur) / total   # P(X=idA, Y=idB)
                
                # 计算互信息项
                if pxy > 0:
                    MI += pxy * math.log(pxy / (px * py + eps) + eps, 2)
        
        return MI
    
    def discretize_continuous(self, data: Union[list, np.ndarray], 
                             n_bins: Optional[int] = None) -> np.ndarray:
        """
        将连续变量离散化
        
        Args:
            data: 连续数据
            n_bins: 分箱数，如果为None则使用self.n_bins
            
        Returns:
            np.ndarray: 离散化后的数据
        """
        if n_bins is None:
            n_bins = self.n_bins
            
        data_array = np.array(data)
        
        # 使用等频分箱
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(data_array, quantiles)
        
        # 处理重复的分箱边界
        bins = np.unique(bins)
        
        # 离散化
        discretized = np.digitize(data_array, bins) - 1
        discretized = np.clip(discretized, 0, len(bins) - 2)
        
        return discretized
    
    def calculate(self, x: Union[list, np.ndarray], 
                 y: Union[list, np.ndarray],
                 continuous: bool = True) -> float:
        """
        计算互信息（统一接口）
        
        Args:
            x: 第一个变量
            y: 第二个变量
            continuous: 是否为连续变量
            
        Returns:
            float: 互信息值
        """
        if self.method == 'sklearn' or (self.method == 'auto' and continuous):
            # 优先使用sklearn API
            task_type = 'regression' if continuous else 'classification'
            return self.calculate_sklearn(x, y, task_type)
        elif not continuous:
            # 离散变量使用sklearn的mutual_info_score
            return self.calculate_discrete(x, y)
        else:
            # 连续变量先离散化再计算
            x_discrete = self.discretize_continuous(x)
            y_discrete = self.discretize_continuous(y)
            return self.calculate_manual(x_discrete, y_discrete)
    
    def calculate_with_stats(self, x: Union[list, np.ndarray], 
                           y: Union[list, np.ndarray],
                           continuous: bool = True) -> Tuple[float, dict]:
        """
        计算互信息并返回统计信息
        
        Args:
            x: 第一个变量
            y: 第二个变量
            continuous: 是否为连续变量
            
        Returns:
            tuple: (互信息值, 统计信息字典)
        """
        mi_score = self.calculate(x, y, continuous)
        
        # 计算统计信息
        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
        
        stats_info = {
            'mutual_information': mi_score,
            'n_samples': len(x_array),
            'method': self.method,
            'continuous': continuous,
            'x_mean': np.mean(x_array),
            'y_mean': np.mean(y_array),
            'x_std': np.std(x_array, ddof=1),
            'y_std': np.std(y_array, ddof=1),
            'interpretation': self._interpret_mi(mi_score),
            'n_bins': self.n_bins if not continuous else None
        }
        
        return mi_score, stats_info
    
    def _interpret_mi(self, mi_score: float) -> str:
        """
        解释互信息值的强度
        
        Args:
            mi_score: 互信息值
            
        Returns:
            str: 依赖性强度描述
        """
        if mi_score >= 2.0:
            strength = "极强依赖"
        elif mi_score >= 1.0:
            strength = "强依赖"
        elif mi_score >= 0.5:
            strength = "中等依赖"
        elif mi_score >= 0.1:
            strength = "弱依赖"
        else:
            strength = "极弱依赖或独立"
        
        return f"{strength} (MI={mi_score:.6f})"


def mutual_info_analysis(x: Union[list, np.ndarray], 
                        y: Union[list, np.ndarray],
                        continuous: bool = True,
                        method: str = 'auto') -> float:
    """
    简化函数：计算互信息（论文风格）
    
    Args:
        x: 第一个变量
        y: 第二个变量
        continuous: 是否为连续变量
        method: 计算方法
        
    Returns:
        float: 互信息值
    """
    mia = MutualInformationAnalysis(method=method)
    return mia.calculate(x, y, continuous)


def mutual_info_paper_implementation(x: Union[list, np.ndarray], 
                                   y: Union[list, np.ndarray]) -> float:
    """
    论文附录中的互信息实现（NMI函数的简化版）
    
    基于论文附录中的NMI函数实现，去除了标准化部分
    
    Args:
        x: 第一个变量（将被视为A）
        y: 第二个变量（将被视为B）
        
    Returns:
        float: 互信息值
    """
    # 转换为numpy数组
    A = np.array(x)
    B = np.array(y)
    
    # 验证长度一致
    if len(A) != len(B):
        raise ValueError("A与B的长度应该相等")
    
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    
    # 计算互信息
    MI = 0
    eps = 1.4e-45  # 防止log(0)，与论文保持一致
    
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A == idA)
            idBOccur = np.where(B == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)
            
            px = 1.0 * len(idAOccur[0]) / total
            py = 1.0 * len(idBOccur[0]) / total
            pxy = 1.0 * len(idABOccur) / total
            
            if pxy > 0:
                MI = MI + pxy * math.log(pxy / (px * py) + eps, 2)
    
    return MI


def batch_mutual_info_analysis(reference: Union[list, np.ndarray], 
                              sequences: List[Union[list, np.ndarray]], 
                              continuous: bool = True) -> List[Tuple[int, float]]:
    """
    批量计算多个序列与参考序列的互信息
    
    Args:
        reference: 参考序列
        sequences: 多个比较序列的列表
        continuous: 是否为连续变量
        
    Returns:
        List[Tuple[int, float]]: [(序列索引, 互信息值), ...] 按互信息值降序排列
    """
    mia = MutualInformationAnalysis()
    
    results = []
    for i, seq in enumerate(sequences):
        mi_score = mia.calculate(reference, seq, continuous)
        results.append((i, mi_score))
    
    # 按互信息值降序排列
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def demo_usage():
    """演示使用方法"""
    print("=" * 50)
    print("互信息检验算法通用模板演示（精简版）")
    print("=" * 50)
    
    # 创建示例数据
    np.random.seed(42)
    
    # 示例1：线性相关数据
    x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y1 = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])  # y = 2x
    
    print("示例1：线性相关数据")
    print(f"X: {x1}")
    print(f"Y: {y1}")
    
    # 初始化分析器
    mia = MutualInformationAnalysis(method='auto')
    
    # 方法1：基本计算（API版本）
    print("\n方法1：基本计算（sklearn API）")
    mi1 = mia.calculate(x1, y1, continuous=True)
    print(f"互信息值: {mi1:.6f}")
    
    # 方法2：带统计信息的计算
    print("\n方法2：带统计信息")
    mi2, stats = mia.calculate_with_stats(x1, y1, continuous=True)
    print(f"互信息值: {mi2:.6f}")
    print(f"样本数: {stats['n_samples']}")
    print(f"计算方法: {stats['method']}")
    print(f"解释: {stats['interpretation']}")
    
    # 方法3：简化函数（论文风格）
    print("\n方法3：简化函数（论文风格）")
    mi3 = mutual_info_analysis(x1, y1, continuous=True)
    print(f"互信息值: {mi3:.6f}")
    
    # 方法4：论文实现版本（离散化后）
    print("\n方法4：论文实现版本（离散化）")
    # 先离散化连续变量
    x1_discrete = mia.discretize_continuous(x1, n_bins=5)
    y1_discrete = mia.discretize_continuous(y1, n_bins=5)
    mi4 = mutual_info_paper_implementation(x1_discrete, y1_discrete)
    print(f"互信息值: {mi4:.6f}")
    
    # 验证结果一致性（前三种方法应该一致）
    print(f"\n结果验证:")
    print(f"sklearn API: {mi1:.6f}")
    print(f"统计版本: {mi2:.6f}")
    print(f"简化函数: {mi3:.6f}")
    print(f"论文版本: {mi4:.6f}")
    
    # 示例2：非线性相关数据
    print("\n" + "="*30)
    print("示例2：非线性相关数据")
    x2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y2 = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])  # y = x²
    print(f"X: {x2}")
    print(f"Y (x²): {y2}")
    
    mi_nonlinear = mutual_info_analysis(x2, y2, continuous=True)
    print(f"互信息值: {mi_nonlinear:.6f}")
    print(f"解释: {mia._interpret_mi(mi_nonlinear)}")
    
    # 示例3：独立数据
    print("\n" + "="*30)
    print("示例3：独立数据")
    x3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y3 = np.random.rand(10)  # 随机数据
    print(f"X: {x3}")
    print(f"Y (随机): {y3}")
    
    mi_independent = mutual_info_analysis(x3, y3, continuous=True)
    print(f"互信息值: {mi_independent:.6f}")
    print(f"解释: {mia._interpret_mi(mi_independent)}")
    
    # 示例4：离散数据
    print("\n" + "="*30)
    print("示例4：离散数据")
    x4 = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    y4 = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0])  # 几乎完全相反
    print(f"X: {x4}")
    print(f"Y: {y4}")
    
    mi_discrete = mutual_info_analysis(x4, y4, continuous=False)
    print(f"互信息值: {mi_discrete:.6f}")
    print(f"解释: {mia._interpret_mi(mi_discrete)}")
    
    # 示例5：批量分析
    print("\n" + "="*30)
    print("示例5：批量互信息分析")
    reference = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sequences = [
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],  # 线性相关
        [1, 4, 9, 16, 25, 36, 49, 64, 81, 100],  # 非线性相关
        np.random.rand(10),  # 随机数据
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]  # 负相关
    ]
    
    batch_results = batch_mutual_info_analysis(reference, sequences)
    print("批量分析结果（按互信息值降序）：")
    for i, (seq_idx, mi_val) in enumerate(batch_results):
        print(f"排名{i+1}: 序列{seq_idx} - 互信息值: {mi_val:.6f}")


if __name__ == "__main__":
    demo_usage() 