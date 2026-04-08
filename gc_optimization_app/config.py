"""Configuration file for GC temperature program optimization system"""


import numpy as np
from skopt.space import Real, Integer

# ==================== Parameter Dimensions ====================
# Define optimization parameter search space based on experimental experience and literature
parameter_dimensions = [
    Real(30, 60, name='initial_temperature'),          # Initial temperature: 30-60°C (based on conventional GC methods)
    Real(0, 10, name='initial_hold_time'),             # Initial hold time: 0-10s (ensure solvent focusing)
    Real(0.2, 3, name='ramp_rate_1'),                  # First ramp rate: 0.2-3°C/s (separate low boiling components)
    Real(60, 150, name='target_temperature_1'),        # First target temperature: 60-150°C (light component separation temperature)
    Real(0, 10, name='hold_time_1'),                   # First hold time: 0-10s (ensure complete separation of light components)
    Real(0.5, 8, name='ramp_rate_2'),                  # Second ramp rate: 0.5-8°C/s (separate heavy components)
    Real(240, 260, name='target_temperature_2'),       # Second target temperature: 240-260°C (heavy component separation temperature)
    Real(10, 30, name='hold_time_2')                   # Second hold time: 10-30s (ensure complete elution of heavy components)
]

# Extract parameter names list
parameter_names = [dimension.name for dimension in parameter_dimensions]

# Parameter units definition
parameter_units = {
    'initial_temperature': '°C', 
    'initial_hold_time': 's', 
    'ramp_rate_1': '°C/s', 
    'target_temperature_1': '°C',
    'hold_time_1': 's', 
    'ramp_rate_2': '°C/s', 
    'target_temperature_2': '°C', 
    'hold_time_2': 's'
}

# ==================== Optimization Configuration ====================
# Maximum initial sampling points (based on statistical significance requirements)
max_initial_points = 10
# Maximum analysis time limit (seconds), avoid excessively long analysis time
max_analysis_time = 300
# Minimum analysis time limit (seconds), ensure separation effect
min_analysis_time = 80
# Maximum retry attempts for generating parameter combinations
generate_retry_limit = 10
# Minimum and maximum values for Chromatographic Resolution Factor (CRF)
crf_min = 0.0
crf_max = 1.0

# ==================== Optimization Stopping Criteria Configuration ====================
# Stopping criteria based on convergence and computational efficiency
stopping_criteria_config = {
    'max_iterations': 35,              # Maximum iterations (balance computational cost and optimization effect)
    'max_no_improvement': 15,          # Maximum consecutive no-improvement iterations
    'target_score': 1.0,               # Target score (perfect separation)
    'convergence_threshold': 0.01,     # Convergence threshold (score change less than this value)
    'patience': 5                      # Recent iteration window size for evaluating convergence
}

# ==================== Gaussian Process (GP) Optimizer Configuration ====================
# GP parameter settings based on Bayesian optimization theory
gp_optimizer_config = {
    'acquisition_function': 'EI',      # Acquisition function: Expected Improvement (EI)
    'n_restarts_optimizer': 15,        # Optimizer restart times (avoid local optimum)
    'alpha': 1e-6,                     # Noise variance (regularization parameter)
    'normalize_y': True,               # Whether to normalize target values
    'random_state': 42,                # Random seed (ensure reproducibility)
    'initial_length_scales': [2.5, 0.75, 0.6, 22.5, 2.5, 2.0, 5.0, 2.5],  # Initial length scales for each parameter
}

# ==================== Parameter Step Size Configuration ====================
# Parameter step sizes based on experimental equipment precision and practical operation feasibility
parameter_step_sizes = {
    'initial_temperature': 5,       # Initial temperature step (°C)
    'initial_hold_time': 2,         # Initial hold time step (s)
    'ramp_rate_1': 0.1,             # First ramp rate step (°C/s)
    'target_temperature_1': 10,      # First target temperature step (°C)
    'hold_time_1': 2,                # First hold time step (s)
    'ramp_rate_2': 0.5,              # Second ramp rate step (°C/s)
    'target_temperature_2': 5,       # Second target temperature step (°C)
    'hold_time_2': 5                 # Second hold time step (s)
}

# ==================== Signal Processing Configuration ====================
# Parameter settings based on chromatographic signal processing theory
signal_processing_config = {
    'savitzky_golay_window': 15,         # Savitzky-Golay smoothing window size
    'savitzky_golay_polynomial': 3,      # Savitzky-Golay polynomial order
    'als_lambda': 1e5,                   # Asymmetric least squares baseline correction smoothing parameter
    'als_p': 0.01,                       # Asymmetric least squares baseline correction weight parameter
    'cwt_min_signal_noise_ratio': 2,     # Continuous wavelet transform minimum signal-to-noise ratio
    'cwt_noise_percentage': 45,          # Continuous wavelet transform noise percentage
    'peak_merge_distance': 30,           # Peak merge distance threshold
    'crf_resolution_requirement': 1.5,    # Minimum requirement for chromatographic resolution factor
    'crf_separation_weight': 0.3         # Weight of separation in CRF calculation
}

# ==================== Path Configuration ====================
# Path settings based on project structure
work_directory = "gc_optimization"

data_directory = f"{work_directory}/data"                  # Data storage directory
results_directory = f"{work_directory}/results"            # Results storage directory
checkpoint_directory = f"{work_directory}/checkpoints"      # Checkpoint storage directory
visualization_directory = f"{work_directory}/visualizations"  # Visualization results directory

# ==================== UI Configuration ====================
# UI settings based on user experience and interface aesthetics
ui_config = {
    'window_title': "GC Temperature Program Bayesian Optimization System",
    'window_size': "1400x900",
    'font_family': 'Arial',
    'font_size_normal': 10,
    'font_size_title': 14,
}

# ==================== Initial Points Library ====================
# Initial parameter combination library based on experience and literature, used to accelerate optimization process
initial_points_library = [
    [50.0, 2.0, 1.0, 80.0, 0.0, 3.0, 250.0, 40.0],   # Standard method 1
    [55.0, 2.0, 1.8, 100.0, 0.0, 4.2, 252.0, 45.0],  # Fast analysis method
    [45.0, 4.0, 0.6, 70.0, 2.0, 2.0, 248.0, 50.0],   # High resolution method
    [52.0, 2.0, 1.3, 90.0, 0.0, 3.5, 251.0, 40.0],   # Balanced method
    [48.0, 2.0, 0.8, 75.0, 2.0, 2.5, 249.0, 45.0],   # Conservative method
    [40.0, 2.0, 2.0, 100.0, 0.0, 4.5, 255.0, 40.0],  # High temperature fast method
    [56.0, 4.0, 0.5, 85.0, 2.0, 2.0, 248.0, 50.0],   # Low temperature separation method
    [45.0, 2.0, 1.2, 110.0, 0.0, 3.5, 255.0, 45.0],  # Wide range method
    [44.0, 4.0, 0.8, 75.0, 2.0, 1.5, 250.0, 50.0],   # Slow ramp method
    [58.0, 2.0, 2.5, 130.0, 0.0, 7.5, 260.0, 40.0],  # Ultra-fast method
]

# ==================== Analysis Time Penalty Configuration ====================
# Time penalty settings based on balance between experimental efficiency and separation effect
analysis_time_target = 150.0          # Target analysis time (seconds), balance separation effect and analysis efficiency
analysis_time_weight = 0.5            # Time penalty weight, range: 0.0-1.0 (higher means more emphasis on time)
analysis_time_penalty_type = 'squared'  # Penalty type: 'absolute' (linear) or 'squared' (non-linear, recommended)

# ==================== Multi-fidelity Bayesian Optimization Configuration ====================
# Multi-fidelity settings based on balance between computational efficiency and optimization accuracy
multi_fidelity_optimization_config = {
    # Relative costs of low-fidelity and high-fidelity (for cost-aware acquisition function)
    'cost_low_fidelity': 1.0,
    'cost_high_fidelity': 3.0,
    # Number of repetitions for high-fidelity experiments (improve result reliability)
    'high_fidelity_repeat': 3,
    # Number of initial iterations using only low-fidelity (accelerate exploration)
    'initial_low_fidelity_iterations': 12,
    # If the number of existing low-fidelity samples reaches this value at startup, warm-start HF model
    'low_fidelity_samples_for_warmstart': 10,
}