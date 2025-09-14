#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肯德尔相关系数通用模板（精简版）
基于论文《出血性脑卒中临床智能诊疗建模》中的相关性分析算法实现

肯德尔相关系数：又称作和谐系数，用于度量两个有序变量之间单调关系强弱的相关系数

论文公式：τ = (nc - nd) / sqrt((n0 - n1)(n0 - n2))
其中：
- nc 表示XY中拥有一致性的元素对数
- nd 表示XY中拥有不一致性的元素对数

适用条件：
1. 更适用于有序的变量
2. 对异常值不敏感
3. 适用于样本量较小的情况
4. 能够检测非线性单调关系

"""

import numpy as np
import pandas as pd
from typing import Union, Tuple


class KendallCorrelation:
    """肯德尔相关系数计算器（API版本）"""
    
    def __init__(self):
        """初始化肯德尔相关系数计算器"""
        self.name = "Kendall Correlation Coefficient (Kendall's Tau)"
        self.description = "用于度量两个有序变量之间单调关系强弱，对异常值不敏感"
    
    def calculate(self, x: Union[list, np.ndarray, pd.Series], 
                 y: Union[list, np.ndarray, pd.Series]) -> float:
        """
        使用API计算肯德尔相关系数
        
        Args:
            x: 第一个变量的数据
            y: 第二个变量的数据
            
        Returns:
            float: 肯德尔相关系数 (-1 <= τ <= 1)
        """
        # 转换为pandas Series以使用API
        x_series = pd.Series(x)
        y_series = pd.Series(y)
        
        # 使用pandas的corr方法计算肯德尔相关系数
        return x_series.corr(y_series, method="kendall")
    
    def calculate_with_stats(self, x: Union[list, np.ndarray, pd.Series], 
                           y: Union[list, np.ndarray, pd.Series]) -> Tuple[float, dict]:
        """
        计算肯德尔相关系数并返回统计信息
        
        Args:
            x: 第一个变量的数据
            y: 第二个变量的数据
            
        Returns:
            tuple: (相关系数, 统计信息字典)
        """
        correlation = self.calculate(x, y)
        
        # 计算统计信息
        x_array = np.array(x)
        y_array = np.array(y)
        
        # 计算一致性和不一致性对数（用于理解）
        n = len(x_array)
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if (x_array[i] - x_array[j]) * (y_array[i] - y_array[j]) > 0:
                    concordant += 1
                elif (x_array[i] - x_array[j]) * (y_array[i] - y_array[j]) < 0:
                    discordant += 1
        
        total_pairs = n * (n - 1) // 2
        
        stats = {
            'correlation': correlation,
            'n_samples': n,
            'total_pairs': total_pairs,
            'concordant_pairs': concordant,
            'discordant_pairs': discordant,
            'tied_pairs': total_pairs - concordant - discordant,
            'x_mean': np.mean(x_array),
            'y_mean': np.mean(y_array),
            'x_std': np.std(x_array, ddof=1),
            'y_std': np.std(y_array, ddof=1),
            'interpretation': self._interpret_correlation(correlation)
        }
        
        return correlation, stats
    
    def _interpret_correlation(self, tau: float) -> str:
        """
        解释肯德尔相关系数的强度
        
        Args:
            tau: 肯德尔相关系数
            
        Returns:
            str: 相关性强度描述
        """
        abs_tau = abs(tau)
        
        if abs_tau >= 0.7:
            strength = "强单调关系"
        elif abs_tau >= 0.5:
            strength = "中等单调关系"
        elif abs_tau >= 0.3:
            strength = "弱单调关系"
        else:
            strength = "几乎无单调关系"
        
        direction = "正" if tau >= 0 else "负"
        
        return f"{direction}{strength} (τ={tau:.4f})"


def kendall_corr(x: Union[list, np.ndarray], 
                y: Union[list, np.ndarray]) -> float:
    """
    简化函数：计算肯德尔相关系数（论文风格）
    
    Args:
        x: 第一个变量的数据
        y: 第二个变量的数据
        
    Returns:
        float: 肯德尔相关系数
    """
    x, y = pd.Series(x), pd.Series(y)
    return x.corr(y, method="kendall")





def demo_usage():
    """演示使用方法"""
    print("=" * 50)
    print("肯德尔相关系数通用模板演示（精简版）")
    print("=" * 50)
    
    # 创建示例数据（有序变量）
    np.random.seed(42)
    
    # 示例1：正相关的有序数据
    x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y1 = [2, 4, 3, 6, 5, 8, 7, 10, 9, 12]  # 大致递增但有些波动
    
    print("示例1：有序变量数据")
    print(f"X: {x1}")
    print(f"Y: {y1}")
    
    # 初始化计算器
    kendall_calc = KendallCorrelation()
    
    # 方法1：基本计算（API版本）
    print("\n方法1：基本计算（API版本）")
    tau_basic = kendall_calc.calculate(x1, y1)
    print(f"肯德尔相关系数: {tau_basic:.4f}")
    
    # 方法2：带统计信息的计算
    print("\n方法2：带统计信息")
    tau_stats, stats = kendall_calc.calculate_with_stats(x1, y1)
    print(f"相关系数: {tau_stats:.4f}")
    print(f"样本数: {stats['n_samples']}")
    print(f"总对数: {stats['total_pairs']}")
    print(f"一致性对数: {stats['concordant_pairs']}")
    print(f"不一致性对数: {stats['discordant_pairs']}")
    print(f"解释: {stats['interpretation']}")
    
    # 方法3：简化函数（论文风格）
    print("\n方法3：简化函数（论文风格）")
    tau_simple = kendall_corr(x1, y1)
    print(f"肯德尔相关系数: {tau_simple:.4f}")
    
    # 验证结果一致性
    print(f"\n结果验证:")
    results = [tau_basic, tau_stats, tau_simple]
    print(f"API版本: {tau_basic:.6f}")
    print(f"统计版本: {tau_stats:.6f}")  
    print(f"简化函数: {tau_simple:.6f}")
    print(f"所有方法结果一致: {all(abs(r - results[0]) < 1e-10 for r in results)}")
    
    # 示例2：负相关数据
    print("\n" + "="*30)
    print("示例2：负相关的有序数据")
    x2 = [1, 2, 3, 4, 5]
    y2 = [10, 8, 6, 4, 2]
    print(f"X: {x2}")
    print(f"Y: {y2}")
    
    tau2 = kendall_corr(x2, y2)
    print(f"肯德尔相关系数: {tau2:.4f}")
    print(f"解释: {kendall_calc._interpret_correlation(tau2)}")


if __name__ == "__main__":
    demo_usage() 