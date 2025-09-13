"""
Stacking集成算法通用模板
基于论文《出血性脑卒中临床智能诊疗建模》中的Stacking实现
"""

import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler


class StackingEnsemble:
    """
    Stacking集成算法实现
    
    基于论文描述的两层集成框架：
    - 第一层：多个基学习器（初级学习器）
    - 第二层：元学习器（meta-learner）
    """
    
    def __init__(self, base_learners=None, meta_learner=None, cv_folds=5, random_state=42):
        """
        初始化Stacking集成模型
        
        Parameters:
        -----------
        base_learners : list, optional
            基学习器列表，默认使用论文中的四个算法
        meta_learner : object, optional  
            元学习器，默认使用逻辑回归
        cv_folds : int, default=5
            交叉验证折数，论文中使用K折交叉验证
        random_state : int, default=42
            随机种子
        """
        # 默认使用论文中提到的基学习器
        if base_learners is None:
            self.base_learners = [
                ('knn', KNeighborsClassifier(n_neighbors=5)),
                ('svc', SVC(kernel='rbf', probability=True, random_state=random_state)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state)),
                ('gbdt', GradientBoostingClassifier(random_state=random_state))
            ]
        else:
            self.base_learners = base_learners
            
        # 默认使用逻辑回归作为元学习器
        if meta_learner is None:
            self.meta_learner = LogisticRegression(random_state=random_state)
        else:
            self.meta_learner = meta_learner
            
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        训练Stacking模型
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练特征
        y : array-like, shape (n_samples,)
            训练标签
        """
        print("开始训练Stacking集成模型...")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 第一阶段：训练基学习器并生成元特征
        print("第一阶段：训练基学习器...")
        meta_features = np.zeros((X_scaled.shape[0], len(self.base_learners)))
        
        # 使用交叉验证生成元特征，避免过拟合
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for i, (name, learner) in enumerate(self.base_learners):
            print(f"  训练基学习器: {name}")
            # 使用交叉验证预测，防止过拟合
            meta_features[:, i] = cross_val_predict(
                learner, X_scaled, y, cv=cv, method='predict_proba'
            )[:, 1] if hasattr(learner, 'predict_proba') else cross_val_predict(
                learner, X_scaled, y, cv=cv
            )
            
            # 在全部数据上训练基学习器，用于后续预测
            learner.fit(X_scaled, y)
        
        # 第二阶段：训练元学习器
        print("第二阶段：训练元学习器...")
        self.meta_learner.fit(meta_features, y)
        
        self.is_fitted = True
        print("Stacking模型训练完成！")
        
    def predict(self, X):
        """
        使用训练好的Stacking模型进行预测
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            测试特征
            
        Returns:
        --------
        predictions : array-like, shape (n_samples,)
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 数据标准化
        X_scaled = self.scaler.transform(X)
        
        # 使用基学习器生成元特征
        meta_features = np.zeros((X_scaled.shape[0], len(self.base_learners)))
        
        for i, (name, learner) in enumerate(self.base_learners):
            if hasattr(learner, 'predict_proba'):
                meta_features[:, i] = learner.predict_proba(X_scaled)[:, 1]
            else:
                meta_features[:, i] = learner.predict(X_scaled)
        
        # 使用元学习器进行最终预测
        predictions = self.meta_learner.predict(meta_features)
        
        return predictions
    
    def predict_proba(self, X):
        """
        预测类别概率
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            测试特征
            
        Returns:
        --------
        probabilities : array-like, shape (n_samples, n_classes)
            预测概率
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        # 数据标准化
        X_scaled = self.scaler.transform(X)
        
        # 使用基学习器生成元特征
        meta_features = np.zeros((X_scaled.shape[0], len(self.base_learners)))
        
        for i, (name, learner) in enumerate(self.base_learners):
            if hasattr(learner, 'predict_proba'):
                meta_features[:, i] = learner.predict_proba(X_scaled)[:, 1]
            else:
                meta_features[:, i] = learner.predict(X_scaled)
        
        # 使用元学习器预测概率
        if hasattr(self.meta_learner, 'predict_proba'):
            probabilities = self.meta_learner.predict_proba(meta_features)
        else:
            # 如果元学习器不支持概率预测，返回预测结果
            predictions = self.meta_learner.predict(meta_features)
            probabilities = np.zeros((len(predictions), 2))
            probabilities[np.arange(len(predictions)), predictions.astype(int)] = 1
        
        return probabilities
    
    def evaluate(self, X, y):
        """
        评估模型性能
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            测试特征
        y : array-like, shape (n_samples,)
            真实标签
            
        Returns:
        --------
        metrics : dict
            包含准确率、精确率、召回率的字典
        """
        predictions = self.predict(X)
        
        # 计算评估指标（论文中使用的指标）
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted')
        recall = recall_score(y, predictions, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
        
        print("=== 模型评估结果 ===")
        print(f"准确率 (Accuracy): {accuracy:.3f}")
        print(f"精确率 (Precision): {precision:.3f}")
        print(f"召回率 (Recall): {recall:.3f}")
        print("\n详细分类报告:")
        print(classification_report(y, predictions))
        
        return metrics


def demo_usage():
    """
    演示Stacking算法的使用方法
    """
    print("=== Stacking集成算法演示 ===\n")
    
    # 生成示例数据（模拟论文中的医疗数据特征）
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("1. 生成示例数据...")
    X, y = make_classification(
        n_samples=1000, 
        n_features=67,  # 论文中处理后的特征数
        n_classes=2,    # 简化为二分类演示
        n_informative=50,
        n_redundant=10,
        random_state=42
    )
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 2. 训练模型
    print("\n2. 初始化并训练Stacking模型...")
    stacking_model = StackingEnsemble(cv_folds=5, random_state=42)
    stacking_model.fit(X_train, y_train)
    
    # 3. 预测
    print("\n3. 进行预测...")
    predictions = stacking_model.predict(X_test)
    probabilities = stacking_model.predict_proba(X_test)
    
    print(f"预测结果示例: {predictions[:10]}")
    print(f"预测概率示例: {probabilities[:5]}")
    
    # 4. 评估
    print("\n4. 评估模型性能...")
    metrics = stacking_model.evaluate(X_test, y_test)
    
    return stacking_model, metrics


if __name__ == "__main__":
    # 运行演示
    model, results = demo_usage()
    
    print(f"\n=== 演示完成 ===")
    print("该模板实现了论文中描述的Stacking集成算法核心功能：")
    print("- 使用K折交叉验证防止过拟合")
    print("- 支持多种基学习器组合")
    print("- 包含完整的训练、预测、评估流程")
    print("- 可根据具体任务调整基学习器和元学习器") 