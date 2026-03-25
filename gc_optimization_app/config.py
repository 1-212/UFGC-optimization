"""全局配置文件"""

import numpy as np
from skopt.space import Real, Integer

# ==================== 参数维度定义 ====================
PARAM_DIMENSIONS = [
    Real(30, 60, name='T_init'),          # 初始温度: 30-60°C 
    Real(0, 10, name='t_hold_0'),         # 初始保持时间: 0-10s
    Real(0.2, 3, name='R1'),              # 第一段升温速率: 0.2-3°C/s
    Real(60, 150, name='T1'),             # 第一段目标温度: 60-150°C
    Real(0, 10, name='t_hold_1'),         # 第一段保持时间: 0-10s
    Real(0.5, 8, name='R2'),              # 第二段升温速率: 0.5-8°C/s
    Real(240, 260, name='T2'),            # 第二段目标温度: 240-260°C
    Real(10, 30, name='t_hold_2')         # 第二段保持时间: 10-30s
]

PARAM_NAMES = [d.name for d in PARAM_DIMENSIONS]

PARAM_UNITS = {
    'T_init': '°C', 't_hold_0': 's', 'R1': '°C/s', 'T1': '°C',
    't_hold_1': 's', 'R2': '°C/s', 'T2': '°C', 't_hold_2': 's'
}

# ==================== 优化配置 ====================
MAX_INITIAL_POINTS = 10
MAX_ANALYSIS_TIME = 300
MIN_ANALYSIS_TIME = 80
GENERATE_RETRY_LIMIT = 10
CRF_MIN = 0.0
CRF_MAX = 1.0

# ==================== 停止条件配置 ====================
STOPPING_CONFIG = {
    'max_iterations': 35,
    'max_no_improvement': 15,
    'target_score': 1.0,
    'convergence_threshold': 0.01,
    'patience': 5
}

# ==================== GP优化器配置 ====================
GP_CONFIG = {
    'acq_func': 'EI',
    'n_restarts_optimizer': 15,
    'alpha': 1e-6,
    'normalize_y': True,
    'random_state': 42,
    'initial_length_scales': [2.5, 0.75, 0.6, 22.5, 2.5, 2.0, 5.0, 2.5],
}

# ==================== 步长配置 ====================
STEP_SIZES = {
    'T_init': 5,       # 初始温度步长
    't_hold_0': 2,     # 初始保持时间步长
    'R1': 0.1,        # 第一段升温速率步长
    'T1': 10,         # 第一段目标温度步长
    't_hold_1': 2,    # 第一段保持时间步长
    'R2': 0.5,        # 第二段升温速率步长
    'T2': 5,          # 第二段目标温度步长
    't_hold_2': 5     # 第二段保持时间步长
}

# ==================== 信号处理配置 ====================
SIGNAL_CONFIG = {
    'sg_window': 15,
    'sg_poly': 3,
    'als_lambda': 1e5,
    'als_p': 0.01,
    'cwt_min_snr': 2,
    'cwt_noise_perc': 45,
    'peak_merge_distance': 30,
    'crf_rs_req': 1.5,
    'crf_separation_weight': 0.3
}

# ==================== 路径配置 ====================
WORK_DIR = "gc_optimization"
DATA_DIR = f"{WORK_DIR}/data"
RESULTS_DIR = f"{WORK_DIR}/results"
CHECKPOINT_DIR = f"{WORK_DIR}/checkpoints"
VIZ_DIR = f"{WORK_DIR}/visualizations"

# ==================== UI配置 ====================
UI_CONFIG = {
    'window_title': "GC升温参数贝叶斯优化",
    'window_size': "1400x900",
    'font_family': 'Arial',
    'font_size_normal': 10,
    'font_size_title': 14,
}

# ==================== 初始点库 ====================
INITIAL_POINTS_LIBRARY = [
    [50.0, 2.0, 1.0, 80.0, 0.0, 3.0, 250.0, 40.0],
    [55.0, 2.0, 1.8, 100.0, 0.0, 4.2, 252.0, 45.0],
    [45.0, 4.0, 0.6, 70.0, 2.0, 2.0, 248.0, 50.0],
    [52.0, 2.0, 1.3, 90.0, 0.0, 3.5, 251.0, 40.0],
    [48.0, 2.0, 0.8, 75.0, 2.0, 2.5, 249.0, 45.0],
    [40.0, 2.0, 2.0, 100.0, 0.0, 4.5, 255.0, 40.0],
    [56.0, 4.0, 0.5, 85.0, 2.0, 2.0, 248.0, 50.0],
    [45.0, 2.0, 1.2, 110.0, 0.0, 3.5, 255.0, 45.0],
    [44.0, 4.0, 0.8, 75.0, 2.0, 1.5, 250.0, 50.0],
    [58.0, 2.0, 2.5, 130.0, 0.0, 7.5, 260.0, 40.0],
]

# ==================== 分析时间惩罚配置 ====================
# 目标分析时间（秒），驱动优化倾向于靠近该值
ANALYSIS_TIME_TARGET = 150.0
# 惩罚权重（越高越重视靠近目标时间），范围建议 0.0 - 1.0
ANALYSIS_TIME_WEIGHT = 0.5
# 惩罚类型：'absolute' 或 'squared'（推荐 'squared'）
ANALYSIS_TIME_PENALTY = 'squared'

# ==================== 多保真度贝叶斯优化配置 ====================
MFBO_CONFIG = {
    # 低保真度和高保真度的相对成本（用于成本感知采集函数）
    'cost_lf': 1.0,
    'cost_hf': 3.0,
    # 高保真度实验的重复次数（HF 每点重复次数）
    'hf_repeat': 3,
    # 初期仅使用低保真度的迭代轮数（热启动前的LF阶段）
    'initial_lf_iterations': 12,
    # 启动时如果已有的低保真度样本数达到该值，则热启动HF模型
    'lf_samples_for_warmstart': 10,
}