#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灰色关联度算法通用模板（精简版）
基于论文《出血性脑卒中临床智能诊疗建模》中的相关性分析算法实现

灰色关联度分析：用于衡量两个序列之间的关联程度，特别适用于小样本、信息不完全的情况

算法原理：
1. 数据预处理（无量纲化）
2. 计算关联系数
3. 计算关联度

适用条件：
1. 适用于小样本数据
2. 适用于信息不完全的情况
3. 能够处理非线性关系
4. 对数据分布无特殊要求

"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from scipy import stats


class GreyRelationalAnalysis:
    """灰色关联度分析器（精简版）"""
    
    def __init__(self, rho: float = 0.5):
        """
        初始化灰色关联度分析器
        
        Args:
            rho: 分辨系数，通常取0.5，用于调节关联系数的敏感性
        """
        self.name = "Grey Relational Analysis"
        self.description = "灰色关联度分析，适用于小样本、信息不完全的关联性分析"
        self.rho = rho
    
    def normalize_data(self, data: Union[list, np.ndarray]) -> np.ndarray:
        """
        数据无量纲化处理（初值化方法）
        
        Args:
            data: 输入数据序列
            
        Returns:
            np.ndarray: 无量纲化后的数据
        """
        data_array = np.array(data, dtype=float)
        
        # 防止除零错误
        if data_array[0] == 0:
            return data_array
        
        # 初值化：每个值除以第一个值
        return data_array / data_array[0]
    
    def calculate(self, reference: Union[list, np.ndarray], 
                 comparison: Union[list, np.ndarray]) -> float:
        """
        计算灰色关联度
        
        Args:
            reference: 参考序列（母序列）
            comparison: 比较序列（子序列）
            
        Returns:
            float: 灰色关联度 (0 <= γ <= 1)
        """
        # 转换为numpy数组
        ref_array = np.array(reference, dtype=float)
        comp_array = np.array(comparison, dtype=float)
        
        # 数据无量纲化
        ref_norm = self.normalize_data(ref_array)
        comp_norm = self.normalize_data(comp_array)
        
        # 计算绝对差值序列
        delta = np.abs(ref_norm - comp_norm)
        
        # 计算最大差和最小差
        delta_max = np.max(delta)
        delta_min = np.min(delta)
        
        # 防止分母为零
        if delta_max == 0:
            return 1.0
        
        # 计算关联系数序列
        xi = (delta_min + self.rho * delta_max) / (delta + self.rho * delta_max)
        
        # 计算关联度（关联系数的平均值）
        gamma = np.mean(xi)
        
        return gamma
    
    def calculate_with_stats(self, reference: Union[list, np.ndarray], 
                           comparison: Union[list, np.ndarray]) -> Tuple[float, dict]:
        """
        计算灰色关联度并返回统计信息
        
        Args:
            reference: 参考序列
            comparison: 比较序列
            
        Returns:
            tuple: (关联度, 统计信息字典)
        """
        gamma = self.calculate(reference, comparison)
        
        # 计算统计信息
        ref_array = np.array(reference, dtype=float)
        comp_array = np.array(comparison, dtype=float)
        
        stats_info = {
            'grey_relational_grade': gamma,
            'n_samples': len(ref_array),
            'rho': self.rho,
            'reference_mean': np.mean(ref_array),
            'comparison_mean': np.mean(comp_array),
            'reference_std': np.std(ref_array, ddof=1),
            'comparison_std': np.std(comp_array, ddof=1),
            'interpretation': self._interpret_correlation(gamma)
        }
        
        return gamma, stats_info
    
    def _interpret_correlation(self, gamma: float) -> str:
        """
        解释灰色关联度的强度
        
        Args:
            gamma: 灰色关联度
            
        Returns:
            str: 关联性强度描述
        """
        if gamma >= 0.9:
            strength = "极强关联"
        elif gamma >= 0.8:
            strength = "强关联"
        elif gamma >= 0.6:
            strength = "中等关联"
        elif gamma >= 0.4:
            strength = "弱关联"
        else:
            strength = "极弱关联"
        
        return f"{strength} (γ={gamma:.4f})"


def grey_relational_analysis(x: Union[list, np.ndarray], 
                           y: Union[list, np.ndarray], 
                           rho: float = 0.5) -> float:
    """
    简化函数：计算灰色关联度（论文风格）
    
    Args:
        x: 参考序列
        y: 比较序列
        rho: 分辨系数，默认0.5
        
    Returns:
        float: 灰色关联度
    """
    gra = GreyRelationalAnalysis(rho=rho)
    return gra.calculate(x, y)


def grey_relational_analysis_optimized(x: Union[list, np.ndarray], 
                                     y: Union[list, np.ndarray], 
                                     rho: float = 0.5) -> float:
    """
    优化版灰色关联度计算（基于论文中的实现逻辑）
    
    这是基于论文附录中grey_relational_analysis函数的优化实现
    
    Args:
        x: 参考序列
        y: 比较序列  
        rho: 分辨系数权重参数
        
    Returns:
        float: 灰色关联度
    """
    # 转换为numpy数组
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # 数据标准化处理
    max_x = np.max(x)
    min_x = np.min(x)
    max_y = np.max(y)
    min_y = np.min(y)
    
    # 防止分母为零
    if max_x == min_x:
        rho_x = np.ones_like(x)
    else:
        rho_x = (max_x - x + rho * (x - min_x)) / (max_x - min_x + rho * (x - min_x))
    
    if max_y == min_y:
        rho_y = np.ones_like(y)
    else:
        rho_y = (max_y - y + rho * (y - min_y)) / (max_y - min_y + rho * (y - min_y))
    
    # 计算灰色关联度
    correlation = np.abs(rho_x - rho_y).mean()
    
    # 转换为关联度形式（值越小关联度越高，需要转换）
    return 1 - correlation


def batch_grey_analysis(reference: Union[list, np.ndarray], 
                       sequences: List[Union[list, np.ndarray]], 
                       rho: float = 0.5) -> List[Tuple[int, float]]:
    """
    批量计算多个序列与参考序列的灰色关联度
    
    Args:
        reference: 参考序列
        sequences: 多个比较序列的列表
        rho: 分辨系数
        
    Returns:
        List[Tuple[int, float]]: [(序列索引, 关联度), ...] 按关联度降序排列
    """
    gra = GreyRelationalAnalysis(rho=rho)
    
    results = []
    for i, seq in enumerate(sequences):
        gamma = gra.calculate(reference, seq)
        results.append((i, gamma))
    
    # 按关联度降序排列
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def demo_usage():
    """演示使用方法"""
    print("=" * 50)
    print("灰色关联度算法通用模板演示（精简版）")
    print("=" * 50)
    
    # 创建示例数据
    np.random.seed(42)
    
    # 示例1：强关联数据
    reference = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sequence1 = [1.1, 2.2, 3.1, 4.2, 5.1, 6.1, 7.2, 8.1, 9.2, 10.1]  # 与参考序列高度相关
    
    print("示例1：强关联数据")
    print(f"参考序列: {reference}")
    print(f"比较序列: {sequence1}")
    
    # 初始化分析器
    gra = GreyRelationalAnalysis(rho=0.5)
    
    # 方法1：基本计算
    print("\n方法1：基本计算")
    gamma1 = gra.calculate(reference, sequence1)
    print(f"灰色关联度: {gamma1:.4f}")
    
    # 方法2：带统计信息的计算
    print("\n方法2：带统计信息")
    gamma2, stats = gra.calculate_with_stats(reference, sequence1)
    print(f"关联度: {gamma2:.4f}")
    print(f"样本数: {stats['n_samples']}")
    print(f"分辨系数: {stats['rho']}")
    print(f"解释: {stats['interpretation']}")
    
    # 方法3：简化函数（论文风格）
    print("\n方法3：简化函数（论文风格）")
    gamma3 = grey_relational_analysis(reference, sequence1)
    print(f"灰色关联度: {gamma3:.4f}")
    
    # 方法4：优化版（基于论文实现）
    print("\n方法4：优化版（基于论文实现）")
    gamma4 = grey_relational_analysis_optimized(reference, sequence1)
    print(f"灰色关联度: {gamma4:.4f}")
    
    # 验证结果一致性
    print(f"\n结果验证:")
    results = [gamma1, gamma2, gamma3]
    print(f"基本计算: {gamma1:.6f}")
    print(f"统计版本: {gamma2:.6f}")
    print(f"简化函数: {gamma3:.6f}")
    print(f"优化版本: {gamma4:.6f}")
    print(f"前三种方法结果一致: {all(abs(r - results[0]) < 1e-10 for r in results)}")
    
    # 示例2：弱关联数据
    print("\n" + "="*30)
    print("示例2：弱关联数据")
    sequence2 = [10, 1, 9, 2, 8, 3, 7, 4, 6, 5]  # 与参考序列弱相关
    print(f"参考序列: {reference}")
    print(f"比较序列: {sequence2}")
    
    gamma_weak = grey_relational_analysis(reference, sequence2)
    print(f"灰色关联度: {gamma_weak:.4f}")
    print(f"解释: {gra._interpret_correlation(gamma_weak)}")
    
    # 示例3：批量分析
    print("\n" + "="*30)
    print("示例3：批量关联度分析")
    sequences = [
        [1.1, 2.2, 3.1, 4.2, 5.1, 6.1, 7.2, 8.1, 9.2, 10.1],  # 强关联
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],  # 强关联（线性）
        [10, 1, 9, 2, 8, 3, 7, 4, 6, 5],  # 弱关联
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 常数序列
    ]
    
    batch_results = batch_grey_analysis(reference, sequences)
    print("批量分析结果（按关联度降序）：")
    for i, (seq_idx, gamma) in enumerate(batch_results):
        print(f"排名{i+1}: 序列{seq_idx} - 关联度: {gamma:.4f}")


if __name__ == "__main__":
    demo_usage() 