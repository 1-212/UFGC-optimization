"""GC升温参数优化系统配置文件

本模块包含系统的所有配置参数，包括参数维度定义、优化设置、信号处理参数等。
这些配置参数基于实验研究和文献调研，旨在实现气相色谱升温程序的最优设计。

作者: 研究团队
日期: 2026年
"""

import numpy as np
from skopt.space import Real, Integer

# ==================== 参数维度定义 ====================
# 定义优化参数的搜索空间，基于实验经验和文献调研
parameter_dimensions = [
    Real(30, 60, name='initial_temperature'),          # 初始温度: 30-60°C (基于常规GC方法)
    Real(0, 10, name='initial_hold_time'),             # 初始保持时间: 0-10s (确保溶剂聚焦)
    Real(0.2, 3, name='ramp_rate_1'),                  # 第一段升温速率: 0.2-3°C/s (分离低沸点组分)
    Real(60, 150, name='target_temperature_1'),        # 第一段目标温度: 60-150°C (轻组分分离温度)
    Real(0, 10, name='hold_time_1'),                   # 第一段保持时间: 0-10s (确保轻组分完全分离)
    Real(0.5, 8, name='ramp_rate_2'),                  # 第二段升温速率: 0.5-8°C/s (分离重组分)
    Real(240, 260, name='target_temperature_2'),       # 第二段目标温度: 240-260°C (重组分分离温度)
    Real(10, 30, name='hold_time_2')                   # 第二段保持时间: 10-30s (确保重组分完全洗脱)
]

# 提取参数名称列表
parameter_names = [dimension.name for dimension in parameter_dimensions]

# 参数单位定义
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

# ==================== 优化配置 ====================
# 最大初始采样点数（基于统计显著性要求）
max_initial_points = 10
# 最大分析时间限制（秒），避免分析时间过长
max_analysis_time = 300
# 最小分析时间限制（秒），确保分离效果
min_analysis_time = 80
# 生成参数组合的最大重试次数
generate_retry_limit = 10
# 色谱分辨率因子(CRF)的最小值和最大值
crf_min = 0.0
crf_max = 1.0

# ==================== 优化停止条件配置 ====================
# 基于收敛性和计算效率设置的停止条件
stopping_criteria_config = {
    'max_iterations': 35,              # 最大迭代次数（平衡计算成本和优化效果）
    'max_no_improvement': 15,          # 连续无改进的最大次数
    'target_score': 1.0,               # 目标评分（完美分离）
    'convergence_threshold': 0.01,     # 收敛阈值（分数变化小于此值）
    'patience': 5                      # 评估收敛的最近迭代窗口大小
}

# ==================== 高斯过程(GP)优化器配置 ====================
# 基于贝叶斯优化理论的GP参数设置
gp_optimizer_config = {
    'acquisition_function': 'EI',      # 采集函数：期望改进(EI)
    'n_restarts_optimizer': 15,        # 优化器重启次数（避免局部最优）
    'alpha': 1e-6,                     # 噪声方差（正则化参数）
    'normalize_y': True,               # 是否对目标值进行归一化
    'random_state': 42,                # 随机种子（确保可重复性）
    'initial_length_scales': [2.5, 0.75, 0.6, 22.5, 2.5, 2.0, 5.0, 2.5],  # 各参数的初始长度尺度
}

# ==================== 参数步长配置 ====================
# 基于实验设备精度和实际操作可行性设置的参数步长
parameter_step_sizes = {
    'initial_temperature': 5,       # 初始温度步长（°C）
    'initial_hold_time': 2,         # 初始保持时间步长（s）
    'ramp_rate_1': 0.1,             # 第一段升温速率步长（°C/s）
    'target_temperature_1': 10,      # 第一段目标温度步长（°C）
    'hold_time_1': 2,                # 第一段保持时间步长（s）
    'ramp_rate_2': 0.5,              # 第二段升温速率步长（°C/s）
    'target_temperature_2': 5,       # 第二段目标温度步长（°C）
    'hold_time_2': 5                 # 第二段保持时间步长（s）
}

# ==================== 信号处理配置 ====================
# 基于色谱信号处理理论的参数设置
signal_processing_config = {
    'savitzky_golay_window': 15,         #  Savitzky-Golay平滑窗口大小
    'savitzky_golay_polynomial': 3,      #  Savitzky-Golay多项式阶数
    'als_lambda': 1e5,                   # 非对称最小二乘基线校正的平滑参数
    'als_p': 0.01,                       # 非对称最小二乘基线校正的权重参数
    'cwt_min_signal_noise_ratio': 2,     # 连续小波变换的最小信噪比
    'cwt_noise_percentage': 45,          # 连续小波变换的噪声百分比
    'peak_merge_distance': 30,           # 峰合并的距离阈值
    'crf_resolution_requirement': 1.5,    # 色谱分辨率因子的最低要求
    'crf_separation_weight': 0.3         # 分离度在CRF计算中的权重
}

# ==================== 路径配置 ====================
# 基于项目结构的路径设置
work_directory = "gc_optimization"

data_directory = f"{work_directory}/data"                  # 数据存储目录
results_directory = f"{work_directory}/results"            # 结果存储目录
checkpoint_directory = f"{work_directory}/checkpoints"      # 检查点存储目录
visualization_directory = f"{work_directory}/visualizations"  # 可视化结果目录

# ==================== UI配置 ====================
# 基于用户体验和界面美观性的UI设置
ui_config = {
    'window_title': "GC升温参数贝叶斯优化系统",
    'window_size': "1400x900",
    'font_family': 'Arial',
    'font_size_normal': 10,
    'font_size_title': 14,
}

# ==================== 初始点库 ====================
# 基于经验和文献的初始参数组合库，用于加速优化过程
initial_points_library = [
    [50.0, 2.0, 1.0, 80.0, 0.0, 3.0, 250.0, 40.0],   # 标准方法1
    [55.0, 2.0, 1.8, 100.0, 0.0, 4.2, 252.0, 45.0],  # 快速分析方法
    [45.0, 4.0, 0.6, 70.0, 2.0, 2.0, 248.0, 50.0],   # 高分离度方法
    [52.0, 2.0, 1.3, 90.0, 0.0, 3.5, 251.0, 40.0],   # 平衡方法
    [48.0, 2.0, 0.8, 75.0, 2.0, 2.5, 249.0, 45.0],   # 保守方法
    [40.0, 2.0, 2.0, 100.0, 0.0, 4.5, 255.0, 40.0],  # 高温快速方法
    [56.0, 4.0, 0.5, 85.0, 2.0, 2.0, 248.0, 50.0],   # 低温分离方法
    [45.0, 2.0, 1.2, 110.0, 0.0, 3.5, 255.0, 45.0],  # 宽范围方法
    [44.0, 4.0, 0.8, 75.0, 2.0, 1.5, 250.0, 50.0],   # 慢升温方法
    [58.0, 2.0, 2.5, 130.0, 0.0, 7.5, 260.0, 40.0],  # 超快速方法
]

# ==================== 分析时间惩罚配置 ====================
# 基于实验效率和分离效果平衡的时间惩罚设置
analysis_time_target = 150.0          # 目标分析时间（秒），平衡分离效果和分析效率
analysis_time_weight = 0.5            # 时间惩罚权重，范围：0.0-1.0（越高越重视时间）
analysis_time_penalty_type = 'squared'  # 惩罚类型：'absolute'（线性）或 'squared'（非线性，推荐）

# ==================== 多保真度贝叶斯优化配置 ====================
# 基于计算效率和优化精度平衡的多保真度设置
multi_fidelity_optimization_config = {
    # 低保真度和高保真度的相对成本（用于成本感知采集函数）
    'cost_low_fidelity': 1.0,
    'cost_high_fidelity': 3.0,
    # 高保真度实验的重复次数（提高结果可靠性）
    'high_fidelity_repeat': 3,
    # 初期仅使用低保真度的迭代轮数（加速探索）
    'initial_low_fidelity_iterations': 12,
    # 启动时如果已有的低保真度样本数达到该值，则热启动HF模型
    'low_fidelity_samples_for_warmstart': 10,
}