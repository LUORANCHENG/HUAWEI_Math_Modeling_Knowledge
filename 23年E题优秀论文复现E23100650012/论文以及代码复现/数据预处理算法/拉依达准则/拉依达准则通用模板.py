"""
拉依达准则（3σ准则）通用模板
基于论文《出血性脑卒中临床智能诊疗建模》中的描述实现

拉依达准则原理：
- 对被测变量进行等精度测量，得到 x1, x2, ..., xn
- 计算算术平均值 x̄ 及剩余误差 vi = xi - x̄
- 按照贝塞尔公式计算标准误差 σ
- 如果某个测量值 xj 的剩余误差 |vj| = |xj - x̄| > 3σ，则认为是异常值

贝塞尔公式：σ = √[(1/(n-1)) * Σvi²]
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List


class LaidaRule:
    """拉依达准则（3σ准则）异常值检测器"""
    
    def __init__(self, sigma_threshold: float = 3.0):
        """
        初始化拉依达准则检测器
        
        Args:
            sigma_threshold: σ倍数阈值，默认为3（3σ准则）
        """
        self.sigma_threshold = sigma_threshold
        self.mean_ = None
        self.std_ = None
        self.outlier_indices_ = None
    
    def fit_detect(self, data: Union[np.ndarray, pd.Series, List]) -> Tuple[np.ndarray, np.ndarray]:
        """
        拟合数据并检测异常值
        
        Args:
            data: 输入数据（一维数组）
            
        Returns:
            outlier_indices: 异常值的索引
            outlier_values: 异常值
        """
        # 转换为numpy数组
        data = np.array(data)
        n = len(data)
        
        if n < 2:
            raise ValueError("数据量太少，至少需要2个数据点")
        
        # 计算算术平均值
        self.mean_ = np.mean(data)
        
        # 计算剩余误差 vi = xi - x̄
        residual_errors = data - self.mean_
        
        # 按照贝塞尔公式计算标准误差 σ = √[(1/(n-1)) * Σvi²]
        self.std_ = np.sqrt(np.sum(residual_errors**2) / (n - 1))
        
        # 检测异常值：|vj| = |xj - x̄| > 3σ
        abs_residual_errors = np.abs(residual_errors)
        outlier_mask = abs_residual_errors > (self.sigma_threshold * self.std_)
        
        # 获取异常值索引和值
        self.outlier_indices_ = np.where(outlier_mask)[0]
        outlier_values = data[outlier_mask]
        
        return self.outlier_indices_, outlier_values
    
    def remove_outliers(self, data: Union[np.ndarray, pd.Series, List], 
                       method: str = 'remove') -> np.ndarray:
        """
        处理异常值
        
        Args:
            data: 输入数据
            method: 处理方法
                - 'remove': 删除异常值
                - 'mean': 用均值替换异常值
                - 'median': 用中位数替换异常值
                
        Returns:
            处理后的数据
        """
        data = np.array(data)
        outlier_indices, _ = self.fit_detect(data)
        
        if len(outlier_indices) == 0:
            return data
        
        if method == 'remove':
            # 删除异常值
            return np.delete(data, outlier_indices)
        
        elif method == 'mean':
            # 用均值替换异常值
            clean_data = data.copy()
            clean_data[outlier_indices] = self.mean_
            return clean_data
        
        elif method == 'median':
            # 用中位数替换异常值
            clean_data = data.copy()
            median_value = np.median(data)
            clean_data[outlier_indices] = median_value
            return clean_data
        
        else:
            raise ValueError("method 必须是 'remove', 'mean' 或 'median'")
    
    def get_statistics(self) -> dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        if self.mean_ is None:
            raise ValueError("请先调用 fit_detect 方法")
        
        return {
            'mean': self.mean_,
            'std_bessel': self.std_,
            'threshold': self.sigma_threshold * self.std_,
            'outlier_count': len(self.outlier_indices_) if self.outlier_indices_ is not None else 0
        }


def detect_outliers_simple(data: Union[np.ndarray, pd.Series, List], 
                          sigma_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    简化版拉依达准则异常值检测函数
    
    Args:
        data: 输入数据
        sigma_threshold: σ倍数阈值，默认为3
        
    Returns:
        outlier_indices: 异常值索引
        outlier_values: 异常值
    """
    detector = LaidaRule(sigma_threshold)
    return detector.fit_detect(data)


# 使用示例
if __name__ == "__main__":
    # 示例数据（模拟论文中的医疗数据）
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 50)  # 正常数据
    outliers = np.array([200, 10, 250])  # 人工添加的异常值
    test_data = np.concatenate([normal_data, outliers])
    
    print("=== 拉依达准则（3σ准则）异常值检测示例 ===\n")
    
    # 方法1：使用类
    detector = LaidaRule(sigma_threshold=3.0)
    outlier_indices, outlier_values = detector.fit_detect(test_data)
    
    print(f"原始数据长度: {len(test_data)}")
    print(f"检测到异常值数量: {len(outlier_indices)}")
    print(f"异常值索引: {outlier_indices}")
    print(f"异常值: {outlier_values}")
    
    # 获取统计信息
    stats = detector.get_statistics()
    print(f"\n统计信息:")
    print(f"均值: {stats['mean']:.4f}")
    print(f"标准差（贝塞尔公式）: {stats['std_bessel']:.4f}")
    print(f"异常值阈值: {stats['threshold']:.4f}")
    
    # 处理异常值
    print(f"\n=== 异常值处理结果 ===")
    
    # 删除异常值
    cleaned_data_remove = detector.remove_outliers(test_data, method='remove')
    print(f"删除异常值后数据长度: {len(cleaned_data_remove)}")
    
    # 用均值替换异常值
    cleaned_data_mean = detector.remove_outliers(test_data, method='mean')
    print(f"用均值替换后异常值位置的值: {cleaned_data_mean[outlier_indices]}")
    
    # 方法2：使用简化函数
    print(f"\n=== 简化函数使用示例 ===")
    outlier_idx, outlier_vals = detect_outliers_simple(test_data)
    print(f"简化函数检测结果 - 异常值数量: {len(outlier_idx)}")
    print(f"异常值: {outlier_vals}") 