"""
微分方程建模通用模板

基于论文《草原放牧策略研究》中的改进欧拉法实现
适用于生态系统动态建模和其他需要微分方程组求解的问题
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class ODESystem:
    """微分方程组系统配置类"""
    equations: List[Callable]  # 微分方程组函数列表
    initial_conditions: np.ndarray  # 初始条件
    time_span: Tuple[float, float]  # 时间范围 (t_start, t_end)
    step_size: float = 0.01  # 步长h
    variable_names: Optional[List[str]] = None  # 变量名称


class DifferentialEquationSolver:
    """微分方程建模求解器"""
    
    def __init__(self, system: ODESystem):
        """
        初始化求解器
        
        Args:
            system: ODESystem对象，包含微分方程组配置
        """
        self.system = system
        self.results = None
        self.time_points = None
        
    def improved_euler_step(self, t: float, y: np.ndarray, h: float) -> np.ndarray:
        """
        改进欧拉法单步求解
        
        基于论文公式：
        y_p = y_n + h*f(x_n, y_n)  # 预报值
        y_c = y_n + h*f(x_{n+1}, y_p)  # 校正值
        y_{n+1} = (y_p + y_c) / 2  # 平均化
        
        Args:
            t: 当前时间点
            y: 当前状态向量
            h: 步长
            
        Returns:
            下一时间点的状态向量
        """
        # 计算当前点的导数
        dy_dt = np.array([eq(t, y) for eq in self.system.equations])
        
        # 预报值 (欧拉法)
        y_predict = y + h * dy_dt
        
        # 计算预报点的导数
        dy_dt_predict = np.array([eq(t + h, y_predict) for eq in self.system.equations])
        
        # 校正值
        y_correct = y + h * dy_dt_predict
        
        # 改进欧拉法：取预报值和校正值的平均
        y_next = (y_predict + y_correct) / 2
        
        return y_next
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解微分方程组
        
        Returns:
            time_points: 时间点数组
            results: 解向量数组，形状为 (n_points, n_variables)
        """
        t_start, t_end = self.system.time_span
        h = self.system.step_size
        
        # 生成时间点
        self.time_points = np.arange(t_start, t_end + h, h)
        n_points = len(self.time_points)
        n_vars = len(self.system.initial_conditions)
        
        # 初始化结果数组
        self.results = np.zeros((n_points, n_vars))
        self.results[0] = self.system.initial_conditions
        
        # 改进欧拉法求解
        for i in range(1, n_points):
            t = self.time_points[i-1]
            y = self.results[i-1]
            self.results[i] = self.improved_euler_step(t, y, h)
            
        return self.time_points, self.results
    
    def plot_results(self, save_path: Optional[str] = None):
        """绘制求解结果"""
        if self.results is None:
            raise ValueError("请先调用solve()方法求解方程组")
            
        plt.figure(figsize=(12, 8))
        
        n_vars = self.results.shape[1]
        var_names = self.system.variable_names or [f'y{i+1}' for i in range(n_vars)]
        
        for i in range(n_vars):
            plt.subplot((n_vars + 1) // 2, 2, i + 1)
            plt.plot(self.time_points, self.results[:, i], 'b-', linewidth=2)
            plt.title(f'{var_names[i]}随时间变化')
            plt.xlabel('时间')
            plt.ylabel(var_names[i])
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """将结果转换为DataFrame格式"""
        if self.results is None:
            raise ValueError("请先调用solve()方法求解方程组")
            
        var_names = self.system.variable_names or [f'y{i+1}' for i in range(self.results.shape[1])]
        
        df = pd.DataFrame(self.results, columns=var_names)
        df['time'] = self.time_points
        
        return df[['time'] + var_names]


def grassland_ecosystem_model():
    """
    草原生态系统模型示例
    基于论文中的微分方程组
    """
    
    def soil_moisture_change(t, y):
        """
        土壤湿度变化方程
        ΔW = P - (Et_a + G_d + IC_store)
        """
        W, w = y  # W: 土壤湿度, w: 植被生物量
        
        # 模型参数 (可根据实际数据调整)
        P = 50 + 30 * np.sin(2 * np.pi * t / 365)  # 降水量，考虑季节性变化
        Et_a = 0.1 * W  # 土壤蒸发量，与土壤湿度成正比
        G_d = 0.05 * W  # 土壤水渗透量
        
        # 植被截流量计算 (简化版)
        NDVI = min(0.8, w / 3000)  # 植被指数与生物量相关
        FVC = max(0, (NDVI - 0.014) / (0.0783 - 0.014))  # 植被覆盖度
        LAI = FVC * 3  # 叶面积指数
        IC_max = 0.935 + 0.498 * LAI - 0.00575 * LAI**2
        IC_store = FVC * IC_max * (1 - np.exp(-LAI / IC_max)) if IC_max > 0 else 0
        
        dW_dt = P - (Et_a + G_d + IC_store)
        return dW_dt
    
    def vegetation_biomass_change(t, y):
        """
        植被生物量变化方程
        dw/dt = 0.049*w*(1 - w/4000) - 0.0047*S*w
        """
        W, w = y
        
        # 放牧强度 (可调整)
        S = 2.0  # 单位面积载畜率 (羊/天/公顷)
        
        dw_dt = 0.049 * w * (1 - w / 4000) - 0.0047 * S * w
        return dw_dt
    
    # 配置微分方程组
    system = ODESystem(
        equations=[soil_moisture_change, vegetation_biomass_change],
        initial_conditions=np.array([100.0, 1000.0]),  # 初始土壤湿度和植被生物量
        time_span=(0, 365),  # 模拟一年
        step_size=1.0,  # 1天为步长
        variable_names=['土壤湿度(kg/m²)', '植被生物量(g/m²)']
    )
    
    return system


def simple_predator_prey_model():
    """
    简单的捕食者-被捕食者模型示例
    经典的Lotka-Volterra方程
    """
    
    def prey_change(t, y):
        """被捕食者数量变化"""
        x, y_pop = y
        return 0.1 * x - 0.02 * x * y_pop
    
    def predator_change(t, y):
        """捕食者数量变化"""
        x, y_pop = y
        return -0.1 * y_pop + 0.01 * x * y_pop
    
    system = ODESystem(
        equations=[prey_change, predator_change],
        initial_conditions=np.array([100.0, 10.0]),
        time_span=(0, 50),
        step_size=0.1,
        variable_names=['被捕食者数量', '捕食者数量']
    )
    
    return system


def main():
    """主函数 - 演示使用方法"""
    print("=== 微分方程建模通用模板演示 ===\n")
    
    # 示例1: 草原生态系统模型
    print("1. 草原生态系统模型")
    grassland_system = grassland_ecosystem_model()
    solver1 = DifferentialEquationSolver(grassland_system)
    
    time_points1, results1 = solver1.solve()
    print(f"求解完成，共计算 {len(time_points1)} 个时间点")
    
    # 显示最终结果
    final_soil_moisture = results1[-1, 0]
    final_biomass = results1[-1, 1]
    print(f"一年后土壤湿度: {final_soil_moisture:.2f} kg/m²")
    print(f"一年后植被生物量: {final_biomass:.2f} g/m²")
    
    # 绘制结果
    solver1.plot_results("grassland_model_results.png")
    
    # 保存结果
    df1 = solver1.get_results_dataframe()
    df1.to_csv("grassland_model_results.csv", index=False)
    print("结果已保存到 grassland_model_results.csv\n")
    
    # 示例2: 捕食者-被捕食者模型
    print("2. 捕食者-被捕食者模型")
    predator_prey_system = simple_predator_prey_model()
    solver2 = DifferentialEquationSolver(predator_prey_system)
    
    time_points2, results2 = solver2.solve()
    solver2.plot_results("predator_prey_results.png")
    
    df2 = solver2.get_results_dataframe()
    print("捕食者-被捕食者模型求解完成")
    print(f"最终被捕食者数量: {results2[-1, 0]:.2f}")
    print(f"最终捕食者数量: {results2[-1, 1]:.2f}")


if __name__ == "__main__":
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    main() 