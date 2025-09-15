#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并查集（Union-Find）通用模板

基于论文《出血性脑卒中临床智能诊疗建模》中的并查集算法实现
用于高相关性变量过滤和分组管理

Algorithm 1 并查集消除高相关变量:
1. 初始化：对于每个元素，创建一个集合，表示自己，初始时每个集合的代表元素是自身
2. 遍历相关系数矩阵，读取元素i和j
3. 如果C[i][j] > 阈值，则合并两个元素所在的集合
"""

import numpy as np
from typing import List, Set, Dict, Union


class UnionFind:
    """
    并查集数据结构
    
    支持两个基本操作：
    - find(x): 查找元素x所属集合的根节点（代表元素）
    - union(x, y): 合并元素x和y所在的集合
    """
    
    def __init__(self, n: int):
        """
        初始化并查集
        
        Args:
            n: 元素个数
        """
        self.parent = list(range(n))  # 父节点数组，初始时每个元素的父节点是自己
        self.rank = [0] * n  # 秩数组，用于按秩合并优化
        self.size = [1] * n  # 集合大小数组
        self.count = n  # 连通分量个数
    
    def find(self, x: int) -> int:
        """
        查找元素x所属集合的根节点（路径压缩优化）
        
        Args:
            x: 待查找的元素
            
        Returns:
            元素x所属集合的根节点
        """
        if self.parent[x] != x:
            # 路径压缩：将路径上的所有节点直接连接到根节点
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        合并元素x和y所在的集合（按秩合并优化）
        
        Args:
            x: 第一个元素
            y: 第二个元素
            
        Returns:
            如果成功合并返回True，如果x和y已在同一集合返回False
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # 已在同一集合
        
        # 按秩合并：将秩小的树连接到秩大的树下
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
            self.size[root_x] += self.size[root_y]
        
        self.count -= 1  # 连通分量减1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """
        判断元素x和y是否在同一集合中
        
        Args:
            x: 第一个元素
            y: 第二个元素
            
        Returns:
            如果在同一集合返回True，否则返回False
        """
        return self.find(x) == self.find(y)
    
    def get_size(self, x: int) -> int:
        """
        获取元素x所在集合的大小
        
        Args:
            x: 元素
            
        Returns:
            元素x所在集合的大小
        """
        return self.size[self.find(x)]
    
    def get_groups(self) -> Dict[int, List[int]]:
        """
        获取所有连通分量（集合）
        
        Returns:
            字典，键为根节点，值为该集合中的所有元素列表
        """
        groups = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        return groups
    
    def get_representatives(self) -> List[int]:
        """
        获取所有集合的代表元素（根节点）
        
        Returns:
            所有集合的代表元素列表
        """
        return list(set(self.find(i) for i in range(len(self.parent))))


def correlation_based_union_find(correlation_matrix: np.ndarray, 
                                threshold: float = 0.8) -> List[int]:
    """
    基于相关性矩阵的并查集特征过滤
    
    根据论文Algorithm 1实现，用于消除高相关变量
    
    Args:
        correlation_matrix: 相关性系数矩阵
        threshold: 相关性阈值，超过此值的变量将被归为同一组
        
    Returns:
        每个组的代表性变量索引列表
    """
    n = correlation_matrix.shape[0]
    uf = UnionFind(n)
    
    # 遍历相关系数矩阵
    for i in range(n):
        for j in range(i + 1, n):  # 只需要遍历上三角矩阵
            # 如果相关系数超过阈值，则合并两个变量
            if abs(correlation_matrix[i][j]) >= threshold:
                uf.union(i, j)
    
    # 获取每个集合的代表元素
    representatives = uf.get_representatives()
    return sorted(representatives)


def filter_features_by_correlation(feature_names: List[str], 
                                  correlation_matrix: np.ndarray,
                                  threshold: float = 0.8) -> List[str]:
    """
    基于相关性过滤特征变量
    
    Args:
        feature_names: 特征变量名称列表
        correlation_matrix: 相关性系数矩阵
        threshold: 相关性阈值
        
    Returns:
        过滤后的特征变量名称列表
    """
    representatives = correlation_based_union_find(correlation_matrix, threshold)
    return [feature_names[i] for i in representatives]


# 使用示例
if __name__ == "__main__":
    # 示例1：基本并查集操作
    print("=== 基本并查集操作示例 ===")
    uf = UnionFind(5)
    
    # 合并操作
    uf.union(0, 1)
    uf.union(2, 3)
    uf.union(1, 2)
    
    # 查询操作
    print(f"元素0和元素3是否连通: {uf.connected(0, 3)}")  # True
    print(f"元素0和元素4是否连通: {uf.connected(0, 4)}")  # False
    print(f"连通分量个数: {uf.count}")  # 2
    print(f"所有连通分量: {uf.get_groups()}")
    
    # 示例2：基于相关性的特征过滤
    print("\n=== 相关性特征过滤示例 ===")
    
    # 模拟相关性矩阵
    correlation_matrix = np.array([
        [1.0, 0.9, 0.1, 0.2],
        [0.9, 1.0, 0.2, 0.1],
        [0.1, 0.2, 1.0, 0.85],
        [0.2, 0.1, 0.85, 1.0]
    ])
    
    feature_names = ['HM_volume', 'HM_ACA_R_Ratio', 'ED_volume', 'ED_ACA_R_Ratio']
    
    # 过滤高相关性特征
    filtered_features = filter_features_by_correlation(
        feature_names, correlation_matrix, threshold=0.8
    )
    
    print(f"原始特征: {feature_names}")
    print(f"过滤后特征: {filtered_features}")
    print(f"特征数量: {len(feature_names)} -> {len(filtered_features)}") 