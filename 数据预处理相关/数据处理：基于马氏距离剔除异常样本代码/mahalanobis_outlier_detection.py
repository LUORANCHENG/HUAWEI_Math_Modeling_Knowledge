import numpy as np
import scipy.io
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def mahalanobis_outlier_detection(data, outlier_fraction=0.02):
    """
    基于马氏距离的异常样本检测
    
    参数:
    data: numpy数组，输入数据矩阵，每行为一个样本
    outlier_fraction: float，异常样本的比例阈值，默认为0.02 (2%)
    
    返回:
    inliers: 正常样本的索引列表
    outliers: 异常样本的索引列表
    distances: 所有样本的马氏距离
    threshold: 判定阈值
    """
    
    # 确保输入是numpy数组
    X = np.array(data)
    
    # 计算均值
    mean_vector = np.mean(X, axis=0)
    
    # 获取数据维度
    m, n = X.shape
    
    # 计算每个样本的马氏距离
    distances = []
    
    # 计算协方差矩阵
    cov_matrix = np.cov(X.T)
    
    # 计算协方差矩阵的逆
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # 如果协方差矩阵不可逆，使用伪逆
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
    
    # 计算每个样本与均值的马氏距离
    for i in range(m):
        diff = X[i, :] - mean_vector
        # 马氏距离的平方
        dist_squared = np.dot(np.dot(diff, inv_cov_matrix), diff.T)
        distances.append(dist_squared)
    
    distances = np.array(distances)
    
    # 对距离进行排序，获取排序后的索引
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    
    # 计算阈值
    T = int(np.ceil(m * outlier_fraction))
    threshold = sorted_distances[m - T - 1] if T < m else sorted_distances[0]
    
    # 分类正常样本和异常样本
    inliers = []
    outliers = []
    
    for i in range(len(distances)):
        if distances[i] < threshold:
            inliers.append(i)
        else:
            outliers.append(i)
    
    return inliers, outliers, distances, threshold

def load_matlab_data(filepath):
    """
    加载MATLAB .mat文件
    
    参数:
    filepath: 字符串，.mat文件路径
    
    返回:
    data: numpy数组，加载的数据
    """
    try:
        mat_data = scipy.io.loadmat(filepath)
        # 通常.mat文件中的变量名为'shuju'
        if 'shuju' in mat_data:
            return mat_data['shuju']
        else:
            # 如果没有'shuju'变量，返回第一个非系统变量
            for key in mat_data.keys():
                if not key.startswith('__'):
                    return mat_data[key]
    except Exception as e:
        print(f"加载MATLAB文件时出错: {e}")
        return None

def visualize_results(data, inliers, outliers, distances, threshold):
    """
    可视化检测结果
    
    参数:
    data: 原始数据
    inliers: 正常样本索引
    outliers: 异常样本索引
    distances: 马氏距离数组
    threshold: 阈值
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制马氏距离分布
    plt.subplot(2, 2, 1)
    plt.hist(distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'阈值 = {threshold:.2f}')
    plt.xlabel('马氏距离')
    plt.ylabel('频次')
    plt.title('马氏距离分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制距离排序图
    plt.subplot(2, 2, 2)
    sorted_distances = np.sort(distances)
    plt.plot(range(len(sorted_distances)), sorted_distances, 'b-', linewidth=2)
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'阈值 = {threshold:.2f}')
    plt.xlabel('样本索引（排序后）')
    plt.ylabel('马氏距离')
    plt.title('马氏距离排序图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 如果数据是2D的，绘制散点图
    if data.shape[1] == 2:
        plt.subplot(2, 2, 3)
        plt.scatter(data[inliers, 0], data[inliers, 1], c='blue', alpha=0.6, label='正常样本', s=30)
        plt.scatter(data[outliers, 0], data[outliers, 1], c='red', alpha=0.8, label='异常样本', s=50, marker='x')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('样本分布图（2D）')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 显示统计信息
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = f"""
    检测统计信息:
    
    总样本数: {len(data)}
    正常样本数: {len(inliers)}
    异常样本数: {len(outliers)}
    异常比例: {len(outliers)/len(data)*100:.2f}%
    
    阈值: {threshold:.4f}
    最大距离: {np.max(distances):.4f}
    最小距离: {np.min(distances):.4f}
    平均距离: {np.mean(distances):.4f}
    """
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    """
    主函数：复现原MATLAB代码的功能
    """
    print("正在加载数据...")
    
    # 加载数据
    data_path = '数据预处理相关\数据处理：基于马氏距离剔除异常样本代码\shuju.mat'
    data = load_matlab_data(data_path)
    
    if data is None:
        print("无法加载数据文件")
        return
    
    print(f"数据加载成功，形状: {data.shape}")
    
    # 执行异常检测
    print("正在执行马氏距离异常检测...")
    inliers, outliers, distances, threshold = mahalanobis_outlier_detection(data, outlier_fraction=0.02)
    
    # 打印结果
    print("\n=== 检测结果 ===")
    print(f"总样本数: {len(data)}")
    print(f"正常样本数: {len(inliers)}")
    print(f"异常样本数: {len(outliers)}")
    print(f"异常比例: {len(outliers)/len(data)*100:.2f}%")
    print(f"判定阈值: {threshold:.6f}")
    
    print("\n正常样本行号:")
    for idx in inliers:
        print(f"正常样本行号：{idx + 1}")  # +1 因为MATLAB索引从1开始
    
    print("\n异常样本行号:")
    for idx in outliers:
        print(f"异常样本行号：{idx + 1}")  # +1 因为MATLAB索引从1开始
    
    # 可视化结果
    visualize_results(data, inliers, outliers, distances, threshold)
    
    return inliers, outliers, distances, threshold

if __name__ == "__main__":
    main() 