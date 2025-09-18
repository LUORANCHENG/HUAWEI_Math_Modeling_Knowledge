"""
LSTM神经网络通用模板 - 基于PyTorch实现
根据论文《草原放牧策略研究》中的LSTM模型设计

模板包含三个核心部分：
1. 训练 (Training)
2. 预测 (Prediction) 
3. 评估 (Evaluation)

论文参考参数：
- LSTM层大小: 64
- 隐藏层: 2层 (64, 32神经元)
- 批大小: 32
- 优化器: Adam
- 损失函数: MAE (训练), MSE (验证)
- 激活函数: tanh (LSTM), ReLU (全连接层)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False # 解决负号显示问题

class LSTMModel(nn.Module):
    """
    LSTM神经网络模型
    基于论文中的架构设计
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        # LSTM层 - 论文中使用64个神经元
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层 - 论文中使用2层隐藏层 (64, 32)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, output_size)
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数 - 论文中使用ReLU
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM层 - 论文中提到使用tanh激活函数（LSTM默认）
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc_out(x)
        
        return output

class LSTMTrainer:
    """
    LSTM训练器 - 封装训练、预测、评估功能
    """
    def __init__(self, input_size, output_size=1, hidden_size=64, num_layers=1, 
                 dropout=0.2, learning_rate=0.001, device=None):
        # 设备选择
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        ).to(self.device)
        
        # 优化器 - 论文中使用Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 损失函数 - 论文中训练使用MAE，验证使用MSE
        self.train_criterion = nn.L1Loss()  # MAE
        self.val_criterion = nn.MSELoss()   # MSE
        
        # 数据标准化器
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
    def prepare_data(self, X, y, sequence_length=48, test_size=0.2, val_size=0.2):
        """
        数据预处理 - 论文中使用滑动窗口长度48
        """
        print("准备数据...")
        
        # 创建时间序列数据
        X_sequences, y_sequences = self._create_sequences(X, y, sequence_length)
        
        # 数据标准化
        X_sequences_reshaped = X_sequences.reshape(-1, X_sequences.shape[-1])
        X_scaled = self.scaler_X.fit_transform(X_sequences_reshaped)
        X_sequences = X_scaled.reshape(X_sequences.shape)
        
        y_sequences = self.scaler_y.fit_transform(y_sequences.reshape(-1, 1)).flatten()
        
        # 划分数据集 - 论文中使用80%训练，20%测试
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_sequences, y_sequences, test_size=test_size, random_state=42
        )
        
        # 从训练数据中划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42
        )
        
        # 转换为张量
        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.X_val = torch.FloatTensor(X_val).to(self.device)
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_train = torch.FloatTensor(y_train).to(self.device)
        self.y_val = torch.FloatTensor(y_val).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)
        
        print(f"训练集大小: {self.X_train.shape}")
        print(f"验证集大小: {self.X_val.shape}")
        print(f"测试集大小: {self.X_test.shape}")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def _create_sequences(self, X, y, sequence_length):
        """
        创建序列数据 - 滑动窗口方法
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:i+sequence_length])
            y_sequences.append(y[i+sequence_length])
            
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, epochs=100, batch_size=32, patience=10, verbose=True):
        """
        训练模型
        参数基于论文设置：batch_size=32, 早停patience=10
        """
        print("开始训练...")
        
        # 创建数据加载器
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(self.X_val, self.y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.train_criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = self.val_criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            # 计算平均损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'训练损失: {avg_train_loss:.4f}, '
                      f'验证损失: {avg_val_loss:.4f}')
            
            # 早停
            if patience_counter >= patience:
                print(f"早停触发，在第{epoch+1}轮停止训练")
                self.model.load_state_dict(best_model_state)
                break
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print("训练完成！")
    
    def predict(self, X_input=None):
        """
        预测功能
        """
        self.model.eval()
        
        if X_input is None:
            X_input = self.X_test
        
        with torch.no_grad():
            if isinstance(X_input, np.ndarray):
                # 标准化输入数据
                X_input_reshaped = X_input.reshape(-1, X_input.shape[-1])
                X_scaled = self.scaler_X.transform(X_input_reshaped)
                X_input = X_scaled.reshape(X_input.shape)
                X_input = torch.FloatTensor(X_input).to(self.device)
            
            predictions = self.model(X_input)
            
            # 反标准化
            predictions_np = predictions.cpu().numpy()
            predictions_original = self.scaler_y.inverse_transform(
                predictions_np.reshape(-1, 1)
            ).flatten()
            
        return predictions_original
    
    def evaluate(self, X_test=None, y_test=None, plot_results=True):
        """
        评估模型 - 使用论文中的评价指标：R2, MSE, RMSE, MAPE
        """
        print("评估模型...")
        
        if X_test is None:
            X_test = self.X_test
            y_test = self.y_test
        
        # 获取预测结果
        y_pred = self.predict(X_test)
        
        # 反标准化真实值
        y_true = self.scaler_y.inverse_transform(
            y_test.cpu().numpy().reshape(-1, 1)
        ).flatten()
        
        # 计算评价指标
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mae = mean_absolute_error(y_true, y_pred)
        
        print("=== 模型评估结果 ===")
        print(f"R² Score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        # 绘制结果对比图
        if plot_results:
            self._plot_results(y_true, y_pred)
            self._plot_training_history()
        
        return {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def _plot_results(self, y_true, y_pred, num_samples=50):
        """
        绘制预测结果对比图
        """
        plt.figure(figsize=(15, 5))
        
        # 选择部分样本进行可视化
        indices = np.random.choice(len(y_true), min(num_samples, len(y_true)), replace=False)
        indices = np.sort(indices)
        
        # 子图1：预测vs真实值对比
        plt.subplot(1, 2, 1)
        plt.plot(indices, y_true[indices], 'b-', label='真实值', linewidth=2)
        plt.plot(indices, y_pred[indices], 'r--', label='预测值', linewidth=2)
        plt.xlabel('样本')
        plt.ylabel('值')
        plt.title('预测值 vs 真实值对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：散点图
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('散点图 (理想情况下应在对角线上)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_training_history(self):
        """
        绘制训练历史
        """
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练损失变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses)
        plt.xlabel('Epoch')
        plt.ylabel('验证损失')
        plt.title('验证损失变化')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        保存模型
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        加载模型
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"模型已从 {filepath} 加载")

def demo_usage():
    """
    使用示例 - 模拟时间序列数据
    """
    print("=== LSTM神经网络通用模板使用示例 ===")
    
    # 1. 生成模拟数据（时间序列）
    np.random.seed(42)
    time_steps = 1000
    t = np.linspace(0, 10, time_steps)
    
    # 创建具有趋势和噪声的时间序列
    trend = 0.5 * t
    seasonal = 2 * np.sin(2 * np.pi * t) + np.sin(4 * np.pi * t)
    noise = np.random.normal(0, 0.5, time_steps)
    
    # 多特征输入（模拟多个传感器数据）
    feature1 = trend + seasonal + noise
    feature2 = 1.5 * trend + 0.8 * seasonal + 0.3 * noise
    feature3 = 0.7 * trend + 1.2 * seasonal + 0.4 * noise
    
    X = np.column_stack([feature1, feature2, feature3])
    
    # 目标变量（比如预测下一个时刻的feature1值）
    y = np.roll(feature1, -1)[:-1]  # 下一个时刻的值
    X = X[:-1]  # 对应的输入
    
    print(f"数据形状 - X: {X.shape}, y: {y.shape}")
    
    # 2. 初始化LSTM训练器
    input_size = X.shape[1]  # 特征数量
    trainer = LSTMTrainer(
        input_size=input_size,
        output_size=1,
        hidden_size=64,  # 论文参数
        num_layers=1,
        dropout=0.2,
        learning_rate=0.001
    )
    
    # 3. 准备数据
    trainer.prepare_data(X, y, sequence_length=48)  # 论文中的滑动窗口长度
    
    # 4. 训练模型
    trainer.train(
        epochs=100, 
        batch_size=32,  # 论文参数
        patience=10,    # 论文参数
        verbose=True
    )
    
    # 5. 评估模型
    results = trainer.evaluate(plot_results=True)
    
    # 6. 保存模型
    trainer.save_model('lstm_model.pth')
    
    return trainer, results

if __name__ == "__main__":
    # 运行示例
    trainer, results = demo_usage()
    
    print("\n=== 使用说明 ===")
    print("1. 训练: trainer.train(epochs=100, batch_size=32)")
    print("2. 预测: predictions = trainer.predict(X_input)")
    print("3. 评估: results = trainer.evaluate()")
    print("4. 保存: trainer.save_model('model.pth')")
    print("5. 加载: trainer.load_model('model.pth')") 