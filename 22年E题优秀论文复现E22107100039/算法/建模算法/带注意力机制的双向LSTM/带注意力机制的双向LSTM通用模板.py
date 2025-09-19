"""
带注意力机制的双向LSTM通用模板
基于论文：草原放牧策略研究中的Attention-BiLSTM模型
作者参考的超参数：LSTM层64，隐藏层2层(64,16)，滑动窗口72，batch_size=64，优化器Adam
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class AttentionLayer(nn.Module):
    """注意力机制层"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_size)
        # 计算注意力权重
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # 加权求和
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class AttentionBiLSTM(nn.Module):
    """带注意力机制的双向LSTM模型"""
    def __init__(self, input_size, lstm_hidden_size=64, num_layers=1, 
                 hidden_sizes=[64, 16], output_size=1, dropout=0.2):
        super(AttentionBiLSTM, self).__init__()
        
        # 双向LSTM层
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = AttentionLayer(lstm_hidden_size * 2)  # 双向所以*2
        
        # 全连接层
        layers = []
        prev_size = lstm_hidden_size * 2
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.fc_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        
        # BiLSTM前向传播
        lstm_out, _ = self.bilstm(x)
        
        # 应用注意力机制
        context_vector, attention_weights = self.attention(lstm_out)
        
        # 通过全连接层
        output = self.fc_layers(context_vector)
        
        return output, attention_weights

class AttentionBiLSTMTrainer:
    """Attention-BiLSTM训练器"""
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.train_losses = []
        self.val_losses = []
        
    def create_sequences(self, data, sequence_length=72):
        """创建时间序列数据（滑动窗口）"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length), :-1])  # 除了最后一列作为特征
            y.append(data[i + sequence_length, -1])       # 最后一列作为目标
        return np.array(X), np.array(y)
    
    def prepare_data(self, data, sequence_length=72, test_size=0.2, val_size=0.2):
        """数据预处理"""
        # 标准化
        data_scaled = self.scaler_X.fit_transform(data)
        
        # 创建序列
        X, y = self.create_sequences(data_scaled, sequence_length)
        
        # 标准化目标变量
        y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 划分数据集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, shuffle=False
        )
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train(self, train_data, val_data, epochs=144, batch_size=64, lr=0.001, 
              patience=10, verbose=True):
        """训练模型"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(dataloader)
            self.train_losses.append(train_loss)
            
            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                val_outputs, _ = self.model(X_val)
                val_loss = criterion(val_outputs.squeeze(), y_val).item()
                self.val_losses.append(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # 早停
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.train_losses, self.val_losses
    
    def predict(self, X):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X).to(self.device)
            predictions, attention_weights = self.model(X)
            
            # 反标准化
            predictions = predictions.cpu().numpy()
            predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            return predictions, attention_weights.cpu().numpy()
    
    def evaluate(self, test_data, verbose=True):
        """评估模型"""
        X_test, y_test = test_data
        
        # 获取预测结果
        predictions, attention_weights = self.predict(X_test)
        
        # 反标准化真实值
        y_true = self.scaler_y.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()
        
        # 计算评估指标
        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100
        
        if verbose:
            print("\n=== 模型评估结果 ===")
            print(f"R² Score: {r2:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"MAPE: {mape:.4f}%")
        
        results = {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'predictions': predictions,
            'y_true': y_true,
            'attention_weights': attention_weights
        }
        
        return results
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Model Loss (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, results, n_samples=50):
        """绘制预测结果对比"""
        predictions = results['predictions'][:n_samples]
        y_true = results['y_true'][:n_samples]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(y_true, label='True Values', marker='o', alpha=0.7)
        plt.plot(predictions, label='Predictions', marker='s', alpha=0.7)
        plt.title('Predictions vs True Values')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, predictions, alpha=0.7)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'R² = {results["r2"]:.4f}')
        
        plt.tight_layout()
        plt.show()

# ==================== 使用示例 ====================

def generate_sample_data(n_samples=1000, n_features=5, sequence_length=72):
    """生成示例时间序列数据"""
    np.random.seed(42)
    
    # 生成带有时间趋势的数据
    time = np.arange(n_samples)
    data = np.zeros((n_samples, n_features))
    
    for i in range(n_features-1):
        # 添加季节性和趋势
        seasonal = np.sin(2 * np.pi * time / 12) + np.cos(2 * np.pi * time / 4)
        trend = 0.001 * time
        noise = np.random.normal(0, 0.1, n_samples)
        data[:, i] = seasonal + trend + noise
    
    # 目标变量（与其他特征相关）
    data[:, -1] = (np.sum(data[:, :-1], axis=1) + 
                   np.sin(2 * np.pi * time / 6) + 
                   np.random.normal(0, 0.05, n_samples))
    
    return data

def main():
    """主函数 - 演示完整的训练、预测、评估流程"""
    
    print("=== 带注意力机制的双向LSTM通用模板 ===\n")
    
    # 1. 生成或加载数据
    print("1. 生成示例数据...")
    data = generate_sample_data(n_samples=500, n_features=5)
    print(f"数据形状: {data.shape}")
    
    # 2. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 3. 创建模型
    print("\n2. 创建模型...")
    sequence_length = 72  # 滑动窗口长度（论文中的设置）
    input_size = data.shape[1] - 1  # 除了目标变量的特征数
    
    model = AttentionBiLSTM(
        input_size=input_size,
        lstm_hidden_size=64,        # 论文设置
        hidden_sizes=[64, 16],      # 论文设置
        output_size=1,
        dropout=0.2
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 创建训练器
    trainer = AttentionBiLSTMTrainer(model, device)
    
    # 5. 数据预处理
    print("\n3. 数据预处理...")
    train_data, val_data, test_data = trainer.prepare_data(
        data, 
        sequence_length=sequence_length,
        test_size=0.2,
        val_size=0.2
    )
    
    print(f"训练集大小: {train_data[0].shape}")
    print(f"验证集大小: {val_data[0].shape}")
    print(f"测试集大小: {test_data[0].shape}")
    
    # 6. 训练模型
    print("\n4. 开始训练...")
    train_losses, val_losses = trainer.train(
        train_data, 
        val_data,
        epochs=144,      # 论文设置
        batch_size=64,   # 论文设置
        lr=0.001,
        patience=10,     # 早停耐心值
        verbose=True
    )
    
    # 7. 评估模型
    print("\n5. 模型评估...")
    results = trainer.evaluate(test_data)
    
    # 8. 可视化结果
    print("\n6. 可视化结果...")
    trainer.plot_training_history()
    trainer.plot_predictions(results)
    
    # 9. 展示注意力权重
    print("\n7. 注意力权重分析...")
    attention_weights = results['attention_weights']
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"平均注意力权重分布 (前10个时间步): {np.mean(attention_weights[:, :10, 0], axis=0)}")
    
    print("\n=== 训练和评估完成 ===")
    
    return model, trainer, results

# ==================== 快速使用函数 ====================

def quick_train_and_evaluate(data, sequence_length=72, **kwargs):
    """快速训练和评估函数"""
    
    # 设置默认参数
    default_params = {
        'lstm_hidden_size': 64,
        'hidden_sizes': [64, 16],
        'epochs': 144,
        'batch_size': 64,
        'lr': 0.001,
        'patience': 10
    }
    default_params.update(kwargs)
    
    # 创建模型和训练器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = data.shape[1] - 1
    
    model = AttentionBiLSTM(
        input_size=input_size,
        lstm_hidden_size=default_params['lstm_hidden_size'],
        hidden_sizes=default_params['hidden_sizes'],
        output_size=1
    )
    
    trainer = AttentionBiLSTMTrainer(model, device)
    
    # 数据预处理
    train_data, val_data, test_data = trainer.prepare_data(data, sequence_length)
    
    # 训练
    trainer.train(
        train_data, val_data,
        epochs=default_params['epochs'],
        batch_size=default_params['batch_size'],
        lr=default_params['lr'],
        patience=default_params['patience'],
        verbose=True
    )
    
    # 评估
    results = trainer.evaluate(test_data)
    
    return model, trainer, results

if __name__ == "__main__":
    # 运行主函数
    model, trainer, results = main()
    
    # 模型保存示例
    # torch.save(model.state_dict(), 'attention_bilstm_model.pth')
    
    # 模型加载示例
    # model.load_state_dict(torch.load('attention_bilstm_model.pth')) 