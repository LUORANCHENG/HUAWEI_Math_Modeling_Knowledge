"""
逻辑回归通用模板
基于论文E23100650012中的逻辑回归应用
适用于多分类问题，特别是医学预测场景（如mRS评分预测）
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class LogisticRegressionTemplate:
    """
    逻辑回归通用模板类
    论文中作为Stacking框架的元学习器使用
    """
    
    def __init__(self, max_iter=1000, random_state=42, multi_class='ovr'):
        """
        初始化逻辑回归模型
        
        Args:
            max_iter: 最大迭代次数，适合小样本数据
            random_state: 随机种子，保证结果可重复
            multi_class: 多分类策略，'ovr'适合七分类问题
        """
        self.model = LogisticRegression(
            max_iter=max_iter, 
            random_state=random_state,
            multi_class=multi_class,
            solver='liblinear'  # 适合小样本数据
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def train(self, X_train, y_train, scale_features=True):
        """
        训练逻辑回归模型
        
        Args:
            X_train: 训练特征数据
            y_train: 训练标签数据
            scale_features: 是否进行特征标准化
            
        Returns:
            self: 返回自身，支持链式调用
        """
        print("开始训练逻辑回归模型...")
        
        # 数据预处理：特征标准化（论文中使用min-max标准化）
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train
            
        # 训练模型
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        print(f"模型训练完成！")
        print(f"训练样本数量: {len(X_train)}")
        print(f"特征数量: {X_train.shape[1]}")
        print(f"类别数量: {len(np.unique(y_train))}")
        
        return self
    
    def predict(self, X_test, return_proba=False):
        """
        使用训练好的模型进行预测
        
        Args:
            X_test: 测试特征数据
            return_proba: 是否返回概率预测（论文中需要概率输出）
            
        Returns:
            predictions: 预测结果（类别或概率）
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        # 特征标准化
        try:
            X_test_scaled = self.scaler.transform(X_test)
        except:
            X_test_scaled = X_test
            
        if return_proba:
            # 返回概率预测（用于Stacking集成或医学决策）
            probabilities = self.model.predict_proba(X_test_scaled)
            return probabilities
        else:
            # 返回类别预测
            predictions = self.model.predict(X_test_scaled)
            return predictions
    
    def evaluate(self, X_test, y_test, verbose=True):
        """
        评估模型性能
        基于论文中使用的评价指标：准确率、精确率、召回率
        
        Args:
            X_test: 测试特征数据
            y_test: 测试标签数据
            verbose: 是否打印详细结果
            
        Returns:
            metrics: 评估指标字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        # 获取预测结果
        y_pred = self.predict(X_test)
        y_proba = self.predict(X_test, return_proba=True)
        
        # 计算评价指标
        accuracy = accuracy_score(y_test, y_pred)
        
        # 对于多分类问题，使用macro平均
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
        
        if verbose:
            print("\n=== 模型评估结果 ===")
            print(f"准确率 (Accuracy): {accuracy:.3f}")
            print(f"精确率 (Precision): {precision:.3f}")
            print(f"召回率 (Recall): {recall:.3f}")
            
            print("\n=== 详细分类报告 ===")
            print(classification_report(y_test, y_pred))
            
            print("\n=== 混淆矩阵 ===")
            print(confusion_matrix(y_test, y_pred))
            
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        K折交叉验证（论文中使用K则交叉验证）
        
        Args:
            X: 特征数据
            y: 标签数据
            cv: 交叉验证折数
            
        Returns:
            cv_scores: 交叉验证分数
        """
        print(f"\n开始{cv}折交叉验证...")
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 执行交叉验证
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        
        print(f"交叉验证准确率: {cv_scores}")
        print(f"平均准确率: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return cv_scores

def demo_usage():
    """
    演示如何使用逻辑回归模板
    模拟论文中的七分类问题（mRS评分：0-6）
    """
    print("=== 逻辑回归通用模板演示 ===")
    
    # 生成模拟数据（模拟论文中的医学数据）
    np.random.seed(42)
    n_samples = 100  # 论文中使用前100个患者数据
    n_features = 67  # 论文中处理后的特征数量
    
    # 模拟特征数据
    X = np.random.randn(n_samples, n_features)
    
    # 模拟mRS评分（0-6）
    y = np.random.randint(0, 7, n_samples)
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建并使用逻辑回归模板
    lr_template = LogisticRegressionTemplate()
    
    # 1. 训练模型
    lr_template.train(X_train, y_train)
    
    # 2. 预测
    predictions = lr_template.predict(X_test)
    probabilities = lr_template.predict(X_test, return_proba=True)
    
    print(f"\n预测类别: {predictions}")
    print(f"预测概率形状: {probabilities.shape}")
    
    # 3. 评估模型
    metrics = lr_template.evaluate(X_test, y_test)
    
    # 4. 交叉验证
    cv_scores = lr_template.cross_validate(X, y)
    
    return lr_template, metrics

if __name__ == "__main__":
    # 运行演示
    model, results = demo_usage()
    
    print("\n=== 使用说明 ===")
    print("1. 这个模板专为小样本多分类问题设计")
    print("2. 支持概率预测，适合作为Stacking的元学习器")
    print("3. 内置交叉验证，适合论文中的模型评估需求")
    print("4. 可以直接用于医学预测场景，如mRS评分预测") 