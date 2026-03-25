"""优化器模块"""

import numpy as np
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from config import PARAM_DIMENSIONS as dimensions, GP_CONFIG, STOPPING_CONFIG

def create_ard_matern_kernel():
    """创建ARD Matérn核"""
    initial_length_scales = GP_CONFIG['initial_length_scales']
    
    constant_kernel = ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-5, 1e5)
    )
    
    matern_kernel = Matern(
        length_scale=initial_length_scales,
        length_scale_bounds=(1e-3, 1e3),
        nu=2.5
    )
    
    white_kernel = WhiteKernel(
        noise_level=1e-5,
        noise_level_bounds=(1e-10, 1e-1)
    )
    
    kernel = constant_kernel * matern_kernel + white_kernel
    return kernel

def create_gp_regressor():
    """创建GP回归器"""
    kernel = create_ard_matern_kernel()
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=GP_CONFIG['n_restarts_optimizer'],
        alpha=GP_CONFIG['alpha'],
        normalize_y=GP_CONFIG['normalize_y'],
        random_state=GP_CONFIG['random_state']
    )
    
    return gp

class BayesianOptimizer:
    """贝叶斯优化器包装"""
    
    def __init__(self):
        self.optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator=create_gp_regressor(),
            acq_func=GP_CONFIG['acq_func'],
            n_initial_points=10,
            random_state=GP_CONFIG['random_state']
        )
    
    def ask(self):
        """生成候选点并将其转换为离散值"""
        continuous_params = self.optimizer.ask()
        discrete_params = self._to_discrete_values(continuous_params)
        return discrete_params
    
    def _to_discrete_values(self, params):
        """将连续参数值转换为最接近的离散值"""
        from config import STEP_SIZES, PARAM_NAMES
        discrete_params = []
        
        for i, param_val in enumerate(params):
            param_name = PARAM_NAMES[i]
            if param_name in STEP_SIZES:
                step_size = STEP_SIZES[param_name]
                # 将参数值四舍五入到最接近的步长倍数
                rounded_val = round(param_val / step_size) * step_size
                
                # 确保值在允许范围内
                dim_low = self.optimizer.space.dimensions[i].low
                dim_high = self.optimizer.space.dimensions[i].high
                rounded_val = max(dim_low, min(dim_high, rounded_val))
                
                # 确保值确实是步长的整数倍
                rounded_val = round(rounded_val / step_size) * step_size
                
                discrete_params.append(rounded_val)
            else:
                # 如果没有指定步长，则保留原始值
                discrete_params.append(param_val)
        
        return discrete_params
    
    def tell(self, params, score):
        """反馈评分"""
        self.optimizer.tell(params, -score)
    
    def get_gp_model(self):
        """获取GP模型"""
        return self.optimizer.base_estimator_


class StoppingCriteria:
    """停止条件管理"""
    
    def __init__(self):
        config = STOPPING_CONFIG
        self.max_iterations = config['max_iterations']
        self.max_no_improvement = config['max_no_improvement']
        self.target_score = config['target_score']
        self.convergence_threshold = config['convergence_threshold']
        self.patience = config['patience']
        
        self.iteration = 0
        self.best_score = -np.inf
        self.best_params = None
        self.no_improvement_count = 0
        self.recent_scores = []
    
    def update(self, current_score, current_params):
        """更新停止条件"""
        self.iteration += 1
        self.recent_scores.append(current_score)
        
        if len(self.recent_scores) > self.patience:
            self.recent_scores.pop(0)
        
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_params = current_params.copy() if hasattr(current_params, 'copy') else current_params
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
    
    def should_stop(self):
        """判断是否应该停止优化"""
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
        """获取优化状态"""
        return {
            'iteration': self.iteration,
            'best_score': self.best_score,
            'no_improvement_count': self.no_improvement_count,
            'recent_scores': self.recent_scores.copy()
        }
    
    def reset(self):
        """重置"""
        self.iteration = 0
        self.best_score = -np.inf
        self.best_params = None
        self.no_improvement_count = 0
        self.recent_scores = []