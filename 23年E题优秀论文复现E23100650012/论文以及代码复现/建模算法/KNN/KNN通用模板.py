#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN算法通用模板
基于23年数学建模竞赛E题论文实现
包含训练、预测、评估三个核心部分
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class KNNTemplate:
    """
    KNN算法通用模板类
    
    基于论文中的KNN算法描述实现，支持：
    1. 分类任务的概率预测
    2. 置信度计算
    3. K折交叉验证
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        """
        初始化KNN模型
        
        Parameters:
        -----------
        n_neighbors : int, default=5
            近邻数K
        weights : str, default='uniform'
            权重计算方式 ('uniform', 'distance')
        metric : str, default='euclidean'
            距离度量方式
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def train(self, X_train, y_train, scale_features=True):
        """
        训练KNN模型
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            训练特征数据
        y_train : array-like, shape (n_samples,)
            训练标签数据
        scale_features : bool, default=True
            是否进行特征标准化
            
        Returns:
        --------
        self : KNNTemplate
            返回训练后的模型实例
        """
        print("=" * 50)
        print("KNN模型训练开始")
        print("=" * 50)
        
        # 特征标准化
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            print(f"✓ 完成特征标准化")
        
        # 创建并训练KNN模型
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric
        )
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        print(f"✓ KNN模型训练完成")
        print(f"  - 近邻数K: {self.n_neighbors}")
        print(f"  - 权重方式: {self.weights}")
        print(f"  - 距离度量: {self.metric}")
        print(f"  - 训练样本数: {len(X_train)}")
        print(f"  - 特征维度: {X_train.shape[1]}")
        
        return self
    
    def predict(self, X_test, return_proba=True):
        """
        使用KNN模型进行预测
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            测试特征数据
        return_proba : bool, default=True
            是否返回概率预测（基于论文中的置信度计算）
            
        Returns:
        --------
        predictions : array-like
            预测结果（类别或概率）
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        print("=" * 50)
        print("KNN模型预测开始")
        print("=" * 50)
        
        # 特征标准化（使用训练时的参数）
        if hasattr(self.scaler, 'mean_'):
            X_test = self.scaler.transform(X_test)
            print(f"✓ 完成特征标准化")
        
        if return_proba:
            # 基于论文的置信度计算方法
            # 样本的置信度为：p_c = (max_c ∑I(y_i = c))/k
            probabilities = self.model.predict_proba(X_test)
            # 返回正类的概率
            if probabilities.shape[1] == 2:
                predictions = probabilities[:, 1]
            else:
                predictions = np.max(probabilities, axis=1)
            
            print(f"✓ 完成概率预测")
            print(f"  - 测试样本数: {len(X_test)}")
            print(f"  - 预测概率范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
            
        else:
            # 返回类别预测
            predictions = self.model.predict(X_test)
            print(f"✓ 完成类别预测")
            print(f"  - 测试样本数: {len(X_test)}")
            print(f"  - 预测类别: {np.unique(predictions)}")
        
        return predictions
    
    def evaluate(self, X_test, y_test, task_type='classification'):
        """
        评估KNN模型性能
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            测试特征数据
        y_test : array-like, shape (n_samples,)
            测试标签数据
        task_type : str, default='classification'
            任务类型 ('classification' 或 'probability_regression')
            
        Returns:
        --------
        metrics : dict
            评估指标字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        print("=" * 50)
        print("KNN模型评估开始")
        print("=" * 50)
        
        metrics = {}
        
        if task_type == 'classification':
            # 分类任务评估
            y_pred = self.predict(X_test, return_proba=False)
            
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            print("分类任务评估结果:")
            print(f"  - 准确率 (Accuracy): {metrics['accuracy']:.4f}")
            print(f"  - 精确率 (Precision): {metrics['precision']:.4f}")
            print(f"  - 召回率 (Recall): {metrics['recall']:.4f}")
            
        elif task_type == 'probability_regression':
            # 概率回归任务评估（基于论文中的评估方式）
            y_pred_proba = self.predict(X_test, return_proba=True)
            
            metrics['mae'] = mean_absolute_error(y_test, y_pred_proba)
            metrics['mse'] = mean_squared_error(y_test, y_pred_proba)
            metrics['r2'] = r2_score(y_test, y_pred_proba)
            
            print("概率预测评估结果:")
            print(f"  - 平均绝对误差 (MAE): {metrics['mae']:.4f}")
            print(f"  - 均方误差 (MSE): {metrics['mse']:.4f}")
            print(f"  - 决定系数 (R²): {metrics['r2']:.4f}")
        
        return metrics
    
    def cross_validate(self, X, y, cv=5, scoring='accuracy'):
        """
        K折交叉验证
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            特征数据
        y : array-like, shape (n_samples,)
            标签数据
        cv : int, default=5
            交叉验证折数
        scoring : str, default='accuracy'
            评分方式
            
        Returns:
        --------
        scores : dict
            交叉验证结果
        """
        print("=" * 50)
        print("K折交叉验证开始")
        print("=" * 50)
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建模型（用于交叉验证）
        model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric
        )
        
        # 执行交叉验证
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring)
        
        scores = {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std()
        }
        
        print(f"✓ {cv}折交叉验证完成")
        print(f"  - 评分方式: {scoring}")
        print(f"  - 平均得分: {scores['mean_score']:.4f} ± {scores['std_score']:.4f}")
        print(f"  - 各折得分: {cv_scores}")
        
        return scores


def demo_usage():
    """
    演示KNN模板的使用方法
    """
    print("KNN算法通用模板使用演示")
    print("=" * 60)
    
    # 生成示例数据
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=200, 
        n_features=10, 
        n_informative=5, 
        n_redundant=2,
        n_classes=2, 
        random_state=42
    )
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 创建KNN模型实例
    knn_model = KNNTemplate(n_neighbors=5, weights='uniform')
    
    # 1. 训练模型
    knn_model.train(X_train, y_train)
    
    # 2. 进行预测
    # 分类预测
    class_predictions = knn_model.predict(X_test, return_proba=False)
    print(f"\n分类预测示例: {class_predictions[:5]}")
    
    # 概率预测（基于论文方法）
    prob_predictions = knn_model.predict(X_test, return_proba=True)
    print(f"概率预测示例: {prob_predictions[:5]}")
    
    # 3. 模型评估
    # 分类评估
    classification_metrics = knn_model.evaluate(X_test, y_test, task_type='classification')
    
    # 概率预测评估（将真实标签作为概率处理）
    prob_metrics = knn_model.evaluate(X_test, y_test.astype(float), task_type='probability_regression')
    
    # 4. 交叉验证
    cv_results = knn_model.cross_validate(X, y, cv=5, scoring='accuracy')
    
    print("\n" + "=" * 60)
    print("演示完成！")


if __name__ == "__main__":
    demo_usage() 