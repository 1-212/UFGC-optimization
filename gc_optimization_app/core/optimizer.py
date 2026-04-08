"""GC升温参数优化器模块

本模块实现了贝叶斯优化算法，用于自动优化气相色谱(GC)的升温程序参数。
基于高斯过程回归(GPR)和期望改进(EI)采集函数，结合多保真度优化策略，
实现高效、准确的参数优化。

算法原理：
1. 使用ARD Matérn核函数构建高斯过程模型
2. 通过期望改进(EI)采集函数选择下一个评估点
3. 结合多保真度策略平衡计算效率和优化精度
4. 实现多种停止条件确保优化过程的收敛性

作者: 研究团队
日期: 2026年
"""

import numpy as np
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from config import parameter_dimensions as dimensions, gp_optimizer_config, stopping_criteria_config

def create_ard_matern_kernel():
    """创建自动相关性确定(ARD) Matérn核函数
    
    ARD Matérn核能够为每个参数学习不同的长度尺度，
    从而更好地捕捉参数间的不同重要性。
    采用nu=2.5的Matern核，提供良好的平滑性和灵活性。
    
    Returns:
        kernel: 组合的ARD Matérn核函数
    """
    initial_length_scales = gp_optimizer_config['initial_length_scales']
    
    # 常数核：捕捉全局趋势
    constant_kernel = ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-5, 1e5)
    )
    
    # Matern核：捕捉局部特征和相关性
    matern_kernel = Matern(
        length_scale=initial_length_scales,
        length_scale_bounds=(1e-3, 1e3),
        nu=2.5  # 3/2阶Matern核，平衡平滑性和灵活性
    )
    
    # 白噪声核：处理观测噪声
    white_kernel = WhiteKernel(
        noise_level=1e-5,
        noise_level_bounds=(1e-10, 1e-1)
    )
    
    # 组合核：常数核 * Matern核 + 白噪声核
    kernel = constant_kernel * matern_kernel + white_kernel
    return kernel

def create_gp_regressor():
    """创建高斯过程回归器
    
    基于ARD Matérn核创建GPR模型，用于建模参数与评分之间的关系。
    
    Returns:
        gp: 配置好的高斯过程回归器
    """
    kernel = create_ard_matern_kernel()
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=gp_optimizer_config['n_restarts_optimizer'],
        alpha=gp_optimizer_config['alpha'],
        normalize_y=gp_optimizer_config['normalize_y'],
        random_state=gp_optimizer_config['random_state']
    )
    
    return gp

class BayesianOptimizer:
    """贝叶斯优化器包装类
    
    封装了skopt的Optimizer，实现了参数离散化和评分反馈功能，
    专门用于GC升温参数的优化。
    """
    
    def __init__(self):
        """初始化贝叶斯优化器
        
        配置优化器参数，包括搜索空间、基础估计器、采集函数等。
        """
        self.optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator=create_gp_regressor(),
            acq_func=gp_optimizer_config['acquisition_function'],
            n_initial_points=10,  # 初始采样点数
            random_state=gp_optimizer_config['random_state']
        )
    
    def ask(self):
        """生成候选参数点并转换为离散值
        
        Returns:
            discrete_params: 离散化后的参数值列表
        """
        continuous_params = self.optimizer.ask()
        discrete_params = self._convert_to_discrete_values(continuous_params)
        return discrete_params
    
    def _convert_to_discrete_values(self, params):
        """将连续参数值转换为最接近的离散值
        
        根据配置的步长，将连续参数四舍五入到最接近的离散值，
        同时确保值在参数的有效范围内。
        
        Args:
            params: 连续参数值列表
            
        Returns:
            discrete_params: 离散化后的参数值列表
        """
        from config import parameter_step_sizes, parameter_names
        discrete_params = []
        
        for i, param_value in enumerate(params):
            param_name = parameter_names[i]
            if param_name in parameter_step_sizes:
                step_size = parameter_step_sizes[param_name]
                # 将参数值四舍五入到最接近的步长倍数
                rounded_value = round(param_value / step_size) * step_size
                
                # 确保值在允许范围内
                dimension_low = self.optimizer.space.dimensions[i].low
                dimension_high = self.optimizer.space.dimensions[i].high
                rounded_value = max(dimension_low, min(dimension_high, rounded_value))
                
                # 再次确保值确实是步长的整数倍
                rounded_value = round(rounded_value / step_size) * step_size
                
                discrete_params.append(rounded_value)
            else:
                # 如果没有指定步长，则保留原始值
                discrete_params.append(param_value)
        
        return discrete_params
    
    def tell(self, params, score):
        """反馈评分给优化器
        
        Args:
            params: 评估的参数值
            score: 对应的评分（越高越好）
        """
        # 由于skopt默认最小化目标函数，因此使用负分
        self.optimizer.tell(params, -score)
    
    def get_gp_model(self):
        """获取训练好的高斯过程模型
        
        Returns:
            gp_model: 训练好的高斯过程回归模型
        """
        return self.optimizer.base_estimator_
    
    @property
    def base_estimator_(self):
        """获取基础估计器
        
        Returns:
            base_estimator: 优化器的基础估计器
        """
        return self.optimizer.base_estimator_


class StoppingCriteria:
    """优化停止条件管理
    
    实现多种停止条件，包括最大迭代次数、连续无改进次数、
    达到目标评分和收敛性检查等。
    """
    
    def __init__(self):
        """初始化停止条件
        
        从配置中加载停止条件参数。
        """
        config = stopping_criteria_config
        self.max_iterations = config['max_iterations']
        self.max_no_improvement = config['max_no_improvement']
        self.target_score = config['target_score']
        self.convergence_threshold = config['convergence_threshold']
        self.patience = config['patience']
        
        # 初始化状态变量
        self.iteration = 0
        self.best_score = -np.inf
        self.best_params = None
        self.no_improvement_count = 0
        self.recent_scores = []
    
    def update(self, current_score, current_params):
        """更新停止条件状态
        
        Args:
            current_score: 当前评估的评分
            current_params: 当前评估的参数
        """
        self.iteration += 1
        self.recent_scores.append(current_score)
        
        # 保持最近的评分记录
        if len(self.recent_scores) > self.patience:
            self.recent_scores.pop(0)
        
        # 更新最佳评分和参数
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_params = current_params.copy() if hasattr(current_params, 'copy') else current_params
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
    
    def should_stop(self):
        """判断是否应该停止优化
        
        Returns:
            stop: 是否停止优化
            reasons: 停止的原因列表
        """
        reasons = []
        
        if self.iteration >= self.max_iterations:
            reasons.append(f"✓ 达到最大迭代次数 ({self.iteration}/{self.max_iterations})")
        
        if self.no_improvement_count >= self.max_no_improvement:
            reasons.append(f"✓ 连续 {self.no_improvement_count} 轮无改进")
        
        if self.best_score >= self.target_score:
            reasons.append(f"✓ 达到目标评分 ({self.best_score:.4f} >= {self.target_score:.4f})")
        
        if len(self.recent_scores) >= self.patience:
            score_variance = np.var(self.recent_scores)
            if score_variance < self.convergence_threshold ** 2:
                reasons.append(f"✓ 优化收敛 (方差: {score_variance:.6f})")
        
        return len(reasons) > 0, reasons
    
    def get_status(self):
        """获取优化状态
        
        Returns:
            status: 包含优化状态信息的字典
        """
        return {
            'iteration': self.iteration,
            'best_score': self.best_score,
            'no_improvement_count': self.no_improvement_count,
            'recent_scores': self.recent_scores.copy()
        }
    
    def reset(self):
        """重置停止条件状态"""
        self.iteration = 0
        self.best_score = -np.inf
        self.best_params = None
        self.no_improvement_count = 0
        self.recent_scores = []


class MultiFidelityOptimizer:
    """多保真度贝叶斯优化器
    
    实现多保真度优化策略，通过平衡低保真度（快速）和高保真度（准确）
    评估，提高优化效率。
    """
    
    def __init__(self, dimensions):
        """初始化多保真度优化器
        
        Args:
            dimensions: 参数搜索空间
        """
        from config import multi_fidelity_optimization_config
        self.config = multi_fidelity_optimization_config
        self.dimensions = dimensions
        self.optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator=create_gp_regressor(),
            acq_func='EI',  # 期望改进采集函数
            n_initial_points=10,
            random_state=42
        )
        self.low_fidelity_samples = []
        self.high_fidelity_samples = []
        self.last_suggested_fidelity = 'low'
    
    def ask(self):
        """生成候选参数点
        
        根据当前优化状态，决定使用低保真度还是高保真度评估。
        
        Returns:
            params: 候选参数点
        """
        from config import multi_fidelity_optimization_config
        
        # 初期仅使用低保真度，加速探索
        if len(self.low_fidelity_samples) < multi_fidelity_optimization_config['initial_low_fidelity_iterations']:
            self.last_suggested_fidelity = 'low'
        else:
            # 基于成本比随机选择保真度
            cost_ratio = multi_fidelity_optimization_config['cost_low_fidelity'] / multi_fidelity_optimization_config['cost_high_fidelity']
            if np.random.rand() < cost_ratio:
                self.last_suggested_fidelity = 'high'
            else:
                self.last_suggested_fidelity = 'low'
        
        return self.optimizer.ask()
    
    def tell(self, params, score, fidelity='low'):
        """反馈评分给优化器
        
        Args:
            params: 评估的参数值
            score: 对应的评分
            fidelity: 评估的保真度级别 ('low' 或 'high')
        """
        # 存储样本
        if fidelity == 'low':
            self.low_fidelity_samples.append((params, score))
        else:
            self.high_fidelity_samples.append((params, score))
        
        # 始终以负分反馈给优化器（因为默认最小化）
        self.optimizer.tell(params, -score)
    
    @property
    def base_estimator_(self):
        """获取基础估计器
        
        Returns:
            base_estimator: 优化器的基础估计器
        """
        return self.optimizer.base_estimator_
    
    @property
    def high_fidelity_repeat(self):
        """获取高保真度评估的重复次数
        
        Returns:
            repeat_count: 高保真度评估的重复次数
        """
        return self.config['high_fidelity_repeat']