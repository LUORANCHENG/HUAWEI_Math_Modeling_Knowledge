#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
斯皮尔曼相关系数通用模板（精简版）
基于论文《出血性脑卒中临床智能诊疗建模》中的相关性分析算法实现

斯皮尔曼相关系数：又称秩相关系数，用于度量两个变量之间的单调关系强度

适用条件：
1. 当变量不服从正态分布时
2. 适用于分类、等级变量
3. 通过等级排序的方式将数值转化为等级排序
4. 适用于非正态性的数据进行相关性检验

"""

import numpy as np
import pandas as pd
from typing import Union, Tuple


class SpearmanCorrelation:
    """斯皮尔曼相关系数计算器（API版本）"""
    
    def __init__(self):
        """初始化斯皮尔曼相关系数计算器"""
        self.name = "Spearman Correlation Coefficient"
        self.description = "用于分析两个变量之间的单调关系强度，适用于非正态分布数据"
    
    def calculate(self, x: Union[list, np.ndarray, pd.Series], 
                 y: Union[list, np.ndarray, pd.Series]) -> float:
        """
        使用API计算斯皮尔曼相关系数
        
        Args:
            x: 第一个变量的数据
            y: 第二个变量的数据
            
        Returns:
            float: 斯皮尔曼相关系数 (-1 <= r <= 1)
        """
        # 转换为pandas Series以使用API
        x_series = pd.Series(x)
        y_series = pd.Series(y)
        
        # 使用pandas的corr方法计算斯皮尔曼相关系数
        return x_series.corr(y_series, method="spearman")
    
    def calculate_with_stats(self, x: Union[list, np.ndarray, pd.Series], 
                           y: Union[list, np.ndarray, pd.Series]) -> Tuple[float, dict]:
        """
        计算斯皮尔曼相关系数并返回统计信息
        
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
        
        stats = {
            'correlation': correlation,
            'n_samples': len(x_array),
            'x_mean': np.mean(x_array),
            'y_mean': np.mean(y_array),
            'x_std': np.std(x_array, ddof=1),
            'y_std': np.std(y_array, ddof=1),
            'interpretation': self._interpret_correlation(correlation)
        }
        
        return correlation, stats
    
    def _interpret_correlation(self, r: float) -> str:
        """
        解释相关系数的强度
        
        Args:
            r: 相关系数
            
        Returns:
            str: 相关性强度描述
        """
        abs_r = abs(r)
        
        if abs_r >= 0.8:
            strength = "强单调关系"
        elif abs_r >= 0.6:
            strength = "中等单调关系"
        elif abs_r >= 0.3:
            strength = "弱单调关系"
        else:
            strength = "几乎无单调关系"
        
        direction = "正" if r >= 0 else "负"
        
        return f"{direction}{strength} (rs={r:.4f})"


def spearman_corr(x: Union[list, np.ndarray], 
                 y: Union[list, np.ndarray]) -> float:
    """
    简化函数：计算斯皮尔曼相关系数（论文风格）
    
    Args:
        x: 第一个变量的数据
        y: 第二个变量的数据
        
    Returns:
        float: 斯皮尔曼相关系数
    """
    x, y = pd.Series(x), pd.Series(y)
    return x.corr(y, method="spearman")


def demo_usage():
    """演示使用方法"""
    print("=" * 50)
    print("斯皮尔曼相关系数通用模板演示（精简版）")
    print("=" * 50)
    
    # 创建示例数据（非正态分布）
    np.random.seed(42)
    x = np.random.exponential(2, 100)  # 指数分布
    y = np.random.gamma(2, 2, 100)     # 伽马分布
    # 添加一些单调关系
    y = y + 0.5 * np.sort(x) + np.random.normal(0, 0.5, 100)
    
    # 初始化计算器
    spearman_calc = SpearmanCorrelation()
    
    # 方法1：基本计算
    print("方法1：基本计算")
    r_basic = spearman_calc.calculate(x, y)
    print(f"斯皮尔曼相关系数: {r_basic:.4f}")
    
    # 方法2：带统计信息的计算
    print("\n方法2：带统计信息")
    r_stats, stats = spearman_calc.calculate_with_stats(x, y)
    print(f"相关系数: {r_stats:.4f}")
    print(f"样本数: {stats['n_samples']}")
    print(f"解释: {stats['interpretation']}")
    
    # 方法3：简化函数（论文风格）
    print("\n方法3：简化函数（论文风格）")
    r_simple = spearman_corr(x, y)
    print(f"斯皮尔曼相关系数: {r_simple:.4f}")
    
    # 验证结果一致性
    print(f"\n结果验证:")
    results = [r_basic, r_stats, r_simple]
    print(f"基本计算: {r_basic:.6f}")
    print(f"统计版本: {r_stats:.6f}")  
    print(f"简化函数: {r_simple:.6f}")
    print(f"所有方法结果一致: {all(abs(r - results[0]) < 1e-10 for r in results)}")


if __name__ == "__main__":
    demo_usage() 