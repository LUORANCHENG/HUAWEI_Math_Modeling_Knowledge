#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BP神经网络通用模板 (PyTorch版本)

基于论文《草原放牧策略研究》中的BP神经网络实现
论文中BP神经网络最佳配置：
- 隐藏层数：4层
- 神经元数：128-64-64-32
- 批处理大小：64
- 训练轮数：97
- 优化器：RMSProp
- 评价指标：R²、MSE、RMSE、MAPE

作者：根据E22107100039论文复现
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False # 解决负号显示问题

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class BPNet(nn.Module):
    """
    BP神经网络模型类
    
    使用PyTorch实现的多层全连接神经网络
    """
    
    def __init__(self, input_dim, output_dim, hidden_layers=None, dropout_rate=0.2):
        """
        初始化网络结构
        
        Args:
            input_dim (int): 输入特征维度
            output_dim (int): 输出维度
            hidden_layers (list): 隐藏层神经元数量列表
            dropout_rate (float): Dropout比率
        """
        super(BPNet, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 64, 64, 32]  # 论文配置
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # 组合所有层
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)


class BPNeuralNetwork:
    """
    BP神经网络通用模板类
    
    基于PyTorch实现的BP神经网络，适用于回归预测问题
    """
    
    def __init__(self, input_dim, output_dim, hidden_layers=None, 
                 learning_rate=0.001, batch_size=64, epochs=100, device=None):
        """
        初始化BP神经网络
        
        Args:
            input_dim (int): 输入特征维度
            output_dim (int): 输出维度
            hidden_layers (list): 隐藏层神经元数量列表，默认使用论文配置[128, 64, 64, 32]
            learning_rate (float): 学习率，默认0.001
            batch_size (int): 批处理大小，默认64
            epochs (int): 训练轮数，默认100
            device (str): 设备类型，默认自动选择
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers if hidden_layers else [128, 64, 64, 32]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # 设备选择
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 初始化模型和相关组件
        self.model = BPNet(input_dim, output_dim, hidden_layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        
        # 数据标准化器
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        print("=== BP神经网络模型结构 (PyTorch) ===")
        print(self.model)
        print(f"总参数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _prepare_data(self, X, y, is_training=True):
        """准备数据"""
        if is_training:
            X_scaled = self.scaler_x.fit_transform(X)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            y_scaled = self.scaler_y.fit_transform(y)
        else:
            X_scaled = self.scaler_x.transform(X)
            if y is not None:
                if y.ndim == 1:
                    y = y.reshape(-1, 1)
                y_scaled = self.scaler_y.transform(y)
            else:
                y_scaled = None
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device) if y_scaled is not None else None
        
        return X_tensor, y_tensor
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              validation_split=0.2, verbose=1, patience=10):
        """
        训练BP神经网络
        
        Args:
            X_train (array): 训练特征数据
            y_train (array): 训练目标数据
            X_val (array): 验证特征数据，可选
            y_val (array): 验证目标数据，可选
            validation_split (float): 验证集比例，默认0.2
            verbose (int): 训练过程显示详细程度
            patience (int): 早停耐心值
            
        Returns:
            dict: 训练历史记录
        """
        print("=== 开始训练BP神经网络 ===")
        
        # 准备验证数据
        if X_val is None or y_val is None:
            # 自动划分验证集
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=validation_split, random_state=42
            )
        else:
            X_train_split, y_train_split = X_train, y_train
            X_val_split, y_val_split = X_val, y_val
        
        # 数据预处理
        X_train_tensor, y_train_tensor = self._prepare_data(X_train_split, y_train_split, is_training=True)
        X_val_tensor, y_val_tensor = self._prepare_data(X_val_split, y_val_split, is_training=False)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 早停相关变量
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 训练循环
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 打印进度
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], '
                      f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发！在第 {epoch+1} 轮停止训练")
                    # 恢复最佳模型
                    self.model.load_state_dict(self.best_model_state)
                    break
        
        print("=== 训练完成 ===")
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def predict(self, X_test):
        """
        使用训练好的模型进行预测
        
        Args:
            X_test (array): 测试特征数据
            
        Returns:
            array: 预测结果（已逆标准化）
        """
        self.model.eval()
        
        # 数据预处理
        X_test_tensor, _ = self._prepare_data(X_test, None, is_training=False)
        
        # 预测
        with torch.no_grad():
            y_pred_tensor = self.model(X_test_tensor)
            y_pred_scaled = y_pred_tensor.cpu().numpy()
        
        # 逆标准化
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred.flatten() if y_pred.shape[1] == 1 else y_pred
    
    def evaluate(self, X_test, y_test, show_plot=True):
        """
        评估模型性能
        
        Args:
            X_test (array): 测试特征数据
            y_test (array): 测试真实标签
            show_plot (bool): 是否显示预测结果对比图
            
        Returns:
            dict: 包含各项评价指标的字典
        """
        print("=== 模型性能评估 ===")
        
        # 预测
        y_pred = self.predict(X_test)
        
        # 确保维度一致
        if y_test.ndim == 1 and y_pred.ndim == 2:
            y_pred = y_pred.flatten()
        elif y_test.ndim == 2 and y_pred.ndim == 1:
            y_test = y_test.flatten()
        
        # 计算评价指标
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        # 输出结果
        metrics = {
            'R²': r2,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE(%)': mape
        }
        
        print(f"R² (决定系数): {r2:.4f}")
        print(f"MSE (均方误差): {mse:.4f}")
        print(f"RMSE (均方根误差): {rmse:.4f}")
        print(f"MAPE (平均绝对百分比误差): {mape:.2f}%")
        
        # 与论文结果对比
        print("\n=== 论文中BP神经网络性能对比 ===")
        print("论文结果 - R²: 0.477, MSE: 17.581, RMSE: 4.193, MAPE: 7.939%")
        
        # 绘制预测结果对比图
        if show_plot:
            self._plot_prediction_comparison(y_test, y_pred)
        
        return metrics
    
    def _plot_prediction_comparison(self, y_true, y_pred, sample_size=50):
        """绘制预测结果对比图"""
        plt.figure(figsize=(12, 5))
        
        # 如果数据量太大，只显示部分样本
        if len(y_true) > sample_size:
            indices = np.random.choice(len(y_true), sample_size, replace=False)
            y_true_plot = y_true[indices]
            y_pred_plot = y_pred[indices]
        else:
            y_true_plot = y_true
            y_pred_plot = y_pred
        
        # 预测对比折线图
        plt.subplot(1, 2, 1)
        x = range(len(y_true_plot))
        plt.plot(x, y_true_plot, 'b-o', label='真实值', markersize=4)
        plt.plot(x, y_pred_plot, 'r-s', label='预测值', markersize=4)
        plt.xlabel('样本序号')
        plt.ylabel('数值')
        plt.title('BP神经网络预测结果对比 (PyTorch)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 散点图
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测值 vs 真实值')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        """绘制训练历史曲线"""
        if not self.train_losses:
            print("尚未训练模型，无法绘制训练历史")
            return
        
        plt.figure(figsize=(10, 4))
        
        # 损失函数曲线
        plt.plot(self.train_losses, label='训练损失', color='blue')
        plt.plot(self.val_losses, label='验证损失', color='red')
        plt.title('BP神经网络训练历史 (PyTorch)')
        plt.xlabel('训练轮次')
        plt.ylabel('损失值 (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印最终损失
        print(f"最终训练损失: {self.train_losses[-1]:.6f}")
        print(f"最终验证损失: {self.val_losses[-1]:.6f}")
    
    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y,
            'model_config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'hidden_layers': self.hidden_layers
            }
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler_x = checkpoint['scaler_x']
        self.scaler_y = checkpoint['scaler_y']
        print(f"模型已从 {filepath} 加载")


def demo_usage():
    """演示BP神经网络的使用方法"""
    print("=== BP神经网络通用模板演示 (PyTorch版本) ===")
    
    # 生成模拟数据（类似土壤湿度预测问题）
    np.random.seed(42)
    n_samples = 1000
    n_features = 10  # 模拟10个特征（如降水量、蒸发量等）
    
    # 生成特征数据
    X = np.random.randn(n_samples, n_features)
    # 生成目标数据（模拟土壤湿度，4个深度）
    y = np.random.randn(n_samples, 4) + X[:, :4]  # 简单的线性关系加噪声
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建BP神经网络实例
    bp_model = BPNeuralNetwork(
        input_dim=n_features,
        output_dim=y.shape[1],
        hidden_layers=[128, 64, 64, 32],  # 论文配置
        batch_size=64,
        epochs=50  # 演示用较少轮次
    )
    
    # 训练模型
    history = bp_model.train(X_train, y_train, verbose=1)
    
    # 评估模型
    metrics = bp_model.evaluate(X_test, y_test, show_plot=True)
    
    # 绘制训练历史
    bp_model.plot_training_history()
    
    # 使用模型进行新的预测
    new_data = np.random.randn(5, n_features)
    predictions = bp_model.predict(new_data)
    print(f"\n新数据预测结果形状: {predictions.shape}")
    print(f"预测结果示例:\n{predictions[:2]}")
    
    # 保存模型（可选）
    # bp_model.save_model('bp_model.pth')
    
    return bp_model, metrics


if __name__ == "__main__":
    # 运行演示
    model, results = demo_usage()
    
    print("\n=== 演示完成 ===")
    print("PyTorch版本使用说明：")
    print("1. 实例化 BPNeuralNetwork 类")
    print("2. 调用 train() 方法训练模型")
    print("3. 调用 predict() 方法进行预测")
    print("4. 调用 evaluate() 方法评估性能")
    print("5. 支持模型保存和加载功能") 