"""
改进欧拉法通用模板
用于求解常微分方程(组)的数值解

"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Callable, Union, Tuple, List

plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False # 解决负号显示问题

def improved_euler_api(f: Callable, t_span: Tuple[float, float], y0: Union[float, List[float]], 
                      t_eval: np.ndarray = None, **kwargs) -> dict:
    """
    使用SciPy API实现改进欧拉法（推荐方式）
    
    参数:
        f: 微分方程函数 dy/dt = f(t, y)
        t_span: 时间区间 (t_start, t_end)
        y0: 初始条件
        t_eval: 求解点（可选）
        **kwargs: 其他参数
    
    返回:
        dict: 包含时间t和解y的字典
    """
    # 使用scipy的RK23方法，它是改进欧拉法的高阶版本
    # 对于简单情况，可以使用较小的tolerances来近似改进欧拉法
    solution = solve_ivp(
        f, t_span, y0, 
        method='RK23',  # 2-3阶Runge-Kutta方法（类似改进欧拉法）
        t_eval=t_eval,
        rtol=1e-6, atol=1e-9,
        **kwargs
    )
    
    return {
        't': solution.t,
        'y': solution.y,
        'success': solution.success,
        'message': solution.message
    }


def improved_euler_manual(f: Callable, t_span: Tuple[float, float], y0: Union[float, np.ndarray], 
                         h: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    手动实现改进欧拉法（基于论文公式）
    
    算法步骤：
    1. 预报步：y_p = y_n + h*f(x_n, y_n)
    2. 校正步：y_c = y_n + h*f(x_{n+1}, y_p)  
    3. 平均化：y_{n+1} = (y_p + y_c) / 2
    
    参数:
        f: 微分方程函数 dy/dt = f(t, y)
        t_span: 时间区间 (t_start, t_end)
        y0: 初始条件
        h: 步长
    
    返回:
        t: 时间数组
        y: 解数组
    """
    t_start, t_end = t_span
    
    # 确保y0是numpy数组
    y0 = np.atleast_1d(y0)
    n_vars = len(y0)
    
    # 生成时间网格
    t = np.arange(t_start, t_end + h, h)
    n_steps = len(t)
    
    # 初始化解数组
    if n_vars == 1:
        y = np.zeros(n_steps)
        y[0] = y0[0]
    else:
        y = np.zeros((n_vars, n_steps))
        y[:, 0] = y0
    
    # 改进欧拉法迭代
    for i in range(n_steps - 1):
        t_n = t[i]
        t_n1 = t[i + 1]
        
        if n_vars == 1:
            y_n = y[i]
            
            # 预报步
            y_p = y_n + h * f(t_n, y_n)
            
            # 校正步  
            y_c = y_n + h * f(t_n1, y_p)
            
            # 平均化
            y[i + 1] = (y_p + y_c) / 2
            
        else:
            y_n = y[:, i]
            
            # 预报步
            y_p = y_n + h * f(t_n, y_n)
            
            # 校正步
            y_c = y_n + h * f(t_n1, y_p)
            
            # 平均化
            y[:, i + 1] = (y_p + y_c) / 2
    
    return t, y


class ImprovedEulerSolver:
    """改进欧拉法求解器类"""
    
    def __init__(self, use_api: bool = True):
        """
        初始化求解器
        
        参数:
            use_api: 是否使用API（推荐True）
        """
        self.use_api = use_api
        
    def solve(self, f: Callable, t_span: Tuple[float, float], y0: Union[float, List[float]], 
              h: float = 0.01, t_eval: np.ndarray = None) -> dict:
        """
        求解微分方程
        
        参数:
            f: 微分方程函数
            t_span: 时间区间
            y0: 初始条件
            h: 步长（仅手动实现时使用）
            t_eval: 求解点（仅API时使用）
        
        返回:
            dict: 求解结果
        """
        if self.use_api:
            return improved_euler_api(f, t_span, y0, t_eval)
        else:
            t, y = improved_euler_manual(f, t_span, y0, h)
            return {'t': t, 'y': y, 'success': True, 'message': 'Manual implementation'}
    
    def plot_solution(self, result: dict, labels: List[str] = None, title: str = "改进欧拉法求解结果"):
        """
        绘制求解结果
        
        参数:
            result: 求解结果
            labels: 变量标签
            title: 图标题
        """
        t = result['t']
        y = result['y']
        
        plt.figure(figsize=(10, 6))
        
        if y.ndim == 1:
            plt.plot(t, y, 'b-', linewidth=2, label=labels[0] if labels else 'y(t)')
        else:
            for i in range(y.shape[0]):
                label = labels[i] if labels and i < len(labels) else f'y{i+1}(t)'
                plt.plot(t, y[i], linewidth=2, label=label)
        
        plt.xlabel('时间 t')
        plt.ylabel('解 y(t)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# 示例使用
if __name__ == "__main__":
    
    # 示例1：单个微分方程 dy/dt = -2*y + 1, y(0) = 0
    def example1(t, y):
        return -2 * y + 1
    
    print("=" * 50)
    print("示例1：单个微分方程")
    print("dy/dt = -2*y + 1, y(0) = 0")
    print("解析解: y(t) = 0.5*(1 - exp(-2*t))")
    
    # 使用API方式
    solver_api = ImprovedEulerSolver(use_api=True)
    t_eval = np.linspace(0, 2, 100)
    result_api = solver_api.solve(example1, (0, 2), [0], t_eval=t_eval)
    
    # 使用手动实现
    solver_manual = ImprovedEulerSolver(use_api=False)
    result_manual = solver_manual.solve(example1, (0, 2), [0], h=0.02)
    
    # 解析解对比
    t_analytical = t_eval
    y_analytical = 0.5 * (1 - np.exp(-2 * t_analytical))
    
    print(f"API方式最终值: {result_api['y'][0, -1]:.6f}")
    print(f"手动实现最终值: {result_manual['y'][-1]:.6f}")
    print(f"解析解最终值: {y_analytical[-1]:.6f}")
    
    
    # 示例2：微分方程组（Lotka-Volterra方程）
    def lotka_volterra(t, y):
        """Lotka-Volterra捕食者-猎物模型"""
        x, y_var = y
        dxdt = x * (1 - 0.5 * y_var)
        dydt = -0.5 * y_var * (1 - x)
        return np.array([dxdt, dydt])
    
    print("\n" + "=" * 50)
    print("示例2：Lotka-Volterra微分方程组")
    print("dx/dt = x*(1 - 0.5*y)")
    print("dy/dt = -0.5*y*(1 - x)")
    
    # 求解
    solver = ImprovedEulerSolver(use_api=True)
    result = solver.solve(lotka_volterra, (0, 10), [1.5, 1.0], t_eval=np.linspace(0, 10, 1000))
    
    print(f"求解成功: {result['success']}")
    if result['success']:
        solver.plot_solution(result, labels=['猎物(x)', '捕食者(y)'], 
                           title="Lotka-Volterra系统-改进欧拉法求解")
    
    
    # 示例3：类似论文中的草原生态系统模型
    def grassland_model(t, y):
        """
        简化的草原生态系统模型
        类似论文中的模型：dw/dt = 0.049*w*(1 - w/4000) - 0.0047*S*w
        """
        w = y[0]  # 植被生物量
        S = 2.0   # 放牧强度（羊/天/公顷）
        
        # 植被生长方程
        dwdt = 0.049 * w * (1 - w/4000) - 0.0047 * S * w
        
        return np.array([dwdt])
    
    print("\n" + "=" * 50)
    print("示例3：草原生态系统模型（类似论文）")
    print("dw/dt = 0.049*w*(1 - w/4000) - 0.0047*S*w")
    print("S = 2.0（轻度放牧）")
    
    # 求解
    solver = ImprovedEulerSolver(use_api=True)
    result = solver.solve(grassland_model, (0, 100), [500], t_eval=np.linspace(0, 100, 500))
    
    if result['success']:
        solver.plot_solution(result, labels=['植被生物量'], 
                           title="草原生态系统-植被生物量变化")
        print(f"初始生物量: {result['y'][0, 0]:.1f}")
        print(f"最终生物量: {result['y'][0, -1]:.1f}") 