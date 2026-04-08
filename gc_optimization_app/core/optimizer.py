"""GC temperature program optimizer module"""


import numpy as np
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from config import parameter_dimensions as dimensions, gp_optimizer_config, stopping_criteria_config

def create_ard_matern_kernel():
    """Create Automatic Relevance Determination (ARD) Matérn kernel
    
    ARD Matérn kernel can learn different length scales for each parameter,
    better capturing the different importance between parameters.
    Uses nu=2.5 Matern kernel, providing good smoothness and flexibility.
    
    Returns:
        kernel: Combined ARD Matérn kernel function
    """
    initial_length_scales = gp_optimizer_config['initial_length_scales']
    
    # Constant kernel: capture global trends
    constant_kernel = ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-5, 1e5)
    )
    
    # Matern kernel: capture local features and correlations
    matern_kernel = Matern(
        length_scale=initial_length_scales,
        length_scale_bounds=(1e-3, 1e3),
        nu=2.5  # 3/2 order Matern kernel, balance smoothness and flexibility
    )
    
    # White noise kernel: handle observation noise
    white_kernel = WhiteKernel(
        noise_level=1e-5,
        noise_level_bounds=(1e-10, 1e-1)
    )
    
    # Combined kernel: constant kernel * Matern kernel + white noise kernel
    kernel = constant_kernel * matern_kernel + white_kernel
    return kernel

def create_gp_regressor():
    """Create Gaussian Process Regressor
    
    Create GPR model based on ARD Matérn kernel, used to model the relationship between parameters and scores.
    
    Returns:
        gp: Configured Gaussian Process Regressor
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
    """Bayesian optimizer wrapper class
    
    Encapsulates skopt's Optimizer, implements parameter discretization and score feedback functionality,
    specifically designed for GC temperature program parameter optimization.
    """
    
    def __init__(self):
        """Initialize Bayesian optimizer
        
        Configure optimizer parameters, including search space, base estimator, acquisition function, etc.
        """
        self.optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator=create_gp_regressor(),
            acq_func=gp_optimizer_config['acquisition_function'],
            n_initial_points=10,  # Initial sampling points
            random_state=gp_optimizer_config['random_state']
        )
    
    def ask(self):
        """Generate candidate parameter points and convert to discrete values
        
        Returns:
            discrete_params: List of discretized parameter values
        """
        continuous_params = self.optimizer.ask()
        discrete_params = self._convert_to_discrete_values(continuous_params)
        return discrete_params
    
    def _convert_to_discrete_values(self, params):
        """Convert continuous parameter values to nearest discrete values
        
        Based on configured step sizes, round continuous parameters to the nearest discrete values,
        while ensuring values are within valid parameter ranges.
        
        Args:
            params: List of continuous parameter values
            
        Returns:
            discrete_params: List of discretized parameter values
        """
        from config import parameter_step_sizes, parameter_names
        discrete_params = []
        
        for i, param_value in enumerate(params):
            param_name = parameter_names[i]
            if param_name in parameter_step_sizes:
                step_size = parameter_step_sizes[param_name]
                # Round parameter value to nearest step multiple
                rounded_value = round(param_value / step_size) * step_size
                
                # Ensure value is within allowed range
                dimension_low = self.optimizer.space.dimensions[i].low
                dimension_high = self.optimizer.space.dimensions[i].high
                rounded_value = max(dimension_low, min(dimension_high, rounded_value))
                
                # Ensure value is exactly a multiple of step
                rounded_value = round(rounded_value / step_size) * step_size
                
                discrete_params.append(rounded_value)
            else:
                # If no step specified, keep original value
                discrete_params.append(param_value)
        
        return discrete_params
    
    def tell(self, params, score):
        """Feedback score to optimizer
        
        Args:
            params: Evaluated parameter values
            score: Corresponding score (higher is better)
        """
        # Since skopt minimizes the objective function by default, use negative score
        self.optimizer.tell(params, -score)
    
    def get_gp_model(self):
        """Get trained Gaussian Process model
        
        Returns:
            gp_model: Trained Gaussian Process regression model
        """
        return self.optimizer.base_estimator_
    
    @property
    def base_estimator_(self):
        """Get base estimator
        
        Returns:
            base_estimator: Optimizer's base estimator
        """
        return self.optimizer.base_estimator_


class StoppingCriteria:
    """Optimization stopping criteria management
    
    Implements multiple stopping criteria, including maximum iterations, consecutive no-improvement count,
    reaching target score, and convergence check, etc.
    """
    
    def __init__(self):
        """Initialize stopping criteria
        
        Load stopping criteria parameters from configuration.
        """
        config = stopping_criteria_config
        self.max_iterations = config['max_iterations']
        self.max_no_improvement = config['max_no_improvement']
        self.target_score = config['target_score']
        self.convergence_threshold = config['convergence_threshold']
        self.patience = config['patience']
        
        # Initialize state variables
        self.iteration = 0
        self.best_score = -np.inf
        self.best_params = None
        self.no_improvement_count = 0
        self.recent_scores = []
    
    def update(self, current_score, current_params):
        """Update stopping criteria state
        
        Args:
            current_score: Currently evaluated score
            current_params: Currently evaluated parameters
        """
        self.iteration += 1
        self.recent_scores.append(current_score)
        
        # Keep recent score records
        if len(self.recent_scores) > self.patience:
            self.recent_scores.pop(0)
        
        # Update best score and parameters
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_params = current_params.copy() if hasattr(current_params, 'copy') else current_params
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
    
    def should_stop(self):
        """Determine whether optimization should stop
        
        Returns:
            stop: Whether to stop optimization
            reasons: List of stop reasons
        """
        reasons = []
        
        if self.iteration >= self.max_iterations:
            reasons.append(f"✓ Reached maximum iterations ({self.iteration}/{self.max_iterations})")
        
        if self.no_improvement_count >= self.max_no_improvement:
            reasons.append(f"✓ Consecutive {self.no_improvement_count} rounds without improvement")
        
        if self.best_score >= self.target_score:
            reasons.append(f"✓ Reached target score ({self.best_score:.4f} >= {self.target_score:.4f})")
        
        if len(self.recent_scores) >= self.patience:
            score_variance = np.var(self.recent_scores)
            if score_variance < self.convergence_threshold ** 2:
                reasons.append(f"✓ Optimization converged (variance: {score_variance:.6f})")
        
        return len(reasons) > 0, reasons
    
    def get_status(self):
        """Get optimization status
        
        Returns:
            status: Dictionary containing optimization status information
        """
        return {
            'iteration': self.iteration,
            'best_score': self.best_score,
            'no_improvement_count': self.no_improvement_count,
            'recent_scores': self.recent_scores.copy()
        }
    
    def reset(self):
        """Reset stopping criteria state"""
        self.iteration = 0
        self.best_score = -np.inf
        self.best_params = None
        self.no_improvement_count = 0
        self.recent_scores = []


class MultiFidelityOptimizer:
    """Multi-fidelity Bayesian optimizer
    
    Implements multi-fidelity optimization strategy, by balancing low-fidelity (fast) and high-fidelity (accurate)
    evaluations, to improve optimization efficiency.
    """
    
    def __init__(self, dimensions):
        """Initialize multi-fidelity optimizer
        
        Args:
            dimensions: Parameter search space
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
        """Generate candidate parameter points
        
        Based on current optimization state, decide whether to use low-fidelity or high-fidelity evaluation.
        
        Returns:
            params: Candidate parameter points
        """
        from config import multi_fidelity_optimization_config
        
        # Use only low-fidelity initially to accelerate exploration
        if len(self.low_fidelity_samples) < multi_fidelity_optimization_config['initial_low_fidelity_iterations']:
            self.last_suggested_fidelity = 'low'
        else:
            # Randomly select fidelity based on cost ratio
            cost_ratio = multi_fidelity_optimization_config['cost_low_fidelity'] / multi_fidelity_optimization_config['cost_high_fidelity']
            if np.random.rand() < cost_ratio:
                self.last_suggested_fidelity = 'high'
            else:
                self.last_suggested_fidelity = 'low'
        
        return self.optimizer.ask()
    
    def tell(self, params, score, fidelity='low'):
        """Feedback score to optimizer
        
        Args:
            params: Evaluated parameter values
            score: Corresponding score
            fidelity: Evaluation fidelity level ('low' or 'high')
        """
        # Store samples
        if fidelity == 'low':
            self.low_fidelity_samples.append((params, score))
        else:
            self.high_fidelity_samples.append((params, score))
        
        # Always feedback negative score to optimizer (since it defaults to minimization)
        self.optimizer.tell(params, -score)
    
    @property
    def base_estimator_(self):
        """Get base estimator
        
        Returns:
            base_estimator: Optimizer's base estimator
        """
        return self.optimizer.base_estimator_
    
    @property
    def high_fidelity_repeat(self):
        """Get high-fidelity evaluation repeat count
        
        Returns:
            repeat_count: High-fidelity evaluation repeat count
        """
        return self.config['high_fidelity_repeat']