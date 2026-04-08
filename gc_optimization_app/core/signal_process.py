"""GC信号处理模块

本模块实现了气相色谱(GC)信号的处理功能，包括基线校正、峰值检测、
分离度计算和综合评分等核心算法。基于现代信号处理技术，
实现了高效、准确的色谱信号分析。

核心功能：
1. ALS基线校正：使用非对称最小二乘算法进行基线校正
2. CWT峰值检测：使用连续小波变换进行多尺度峰值检测
3. 分离度计算：基于半峰宽和保留时间计算色谱峰分离度
4. ECRF评分：计算增强型色谱分辨率因子，评估分离效果

作者: 研究团队
日期: 2026年
"""

import numpy as np
import pywt
from scipy.signal import savgol_filter, find_peaks
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from config import signal_processing_config

def perform_als_baseline_correction(signal, lambda_value=1e5, asymmetry_parameter=0.01, max_iterations=10):
    """使用非对称最小二乘(ALS)算法进行基线校正
    
    基于Eilers和Boelens (2005)的算法，通过迭代优化基线估计，
    适用于色谱信号的基线校正。
    
    Args:
        signal: 输入信号
        lambda_value: 平滑参数，控制基线平滑度
        asymmetry_parameter: 非对称参数，控制基线对信号的跟随程度
        max_iterations: 最大迭代次数
        
    Returns:
        baseline: 估计的基线
    """
    signal_length = len(signal)
    # 创建二阶差分矩阵（用于平滑约束）
    difference_matrix = sparse.diags([1, -2, 1], [0, -1, -2], shape=(signal_length, signal_length-2))
    weights = np.ones(signal_length)
    
    for _ in range(max_iterations):
        # 创建权重矩阵
        weight_matrix = sparse.spdiags(weights, 0, signal_length, signal_length)
        # 构建系统矩阵
        system_matrix = weight_matrix + lambda_value * difference_matrix.dot(difference_matrix.transpose())
        # 求解线性系统
        baseline = spsolve(system_matrix, weights * signal)
        # 更新权重
        previous_weights = weights.copy()
        weights = asymmetry_parameter * (signal > baseline) + (1 - asymmetry_parameter) * (signal < baseline)
        # 检查收敛
        if np.linalg.norm(weights - previous_weights) < 1e-6:
            break
    return baseline

def preprocess_chromatogram_data(dataframe, savitzky_golay_window=None, savitzky_golay_polynomial=None, 
                               als_lambda=None, als_asymmetry=None):
    """预处理色谱数据
    
    对色谱数据进行平滑和基线校正处理，提高后续分析的准确性。
    
    Args:
        dataframe: 包含色谱数据的DataFrame
        savitzky_golay_window: Savitzky-Golay平滑窗口大小
        savitzky_golay_polynomial: Savitzky-Golay多项式阶数
        als_lambda: ALS基线校正的平滑参数
        als_asymmetry: ALS基线校正的非对称参数
        
    Returns:
        processed_dataframe: 处理后的DataFrame
    """
    if savitzky_golay_window is None:
        savitzky_golay_window = signal_processing_config['savitzky_golay_window']
    if savitzky_golay_polynomial is None:
        savitzky_golay_polynomial = signal_processing_config['savitzky_golay_polynomial']
    if als_lambda is None:
        als_lambda = signal_processing_config['als_lambda']
    if als_asymmetry is None:
        als_asymmetry = signal_processing_config['als_p']
    
    processed_dataframe = dataframe.copy()
    # 获取信号列（假设第一列是时间或索引）
    signal_columns = processed_dataframe.columns[1:]
    signal_data = processed_dataframe[signal_columns].values
    
    # 确保窗口大小为奇数
    window_size = savitzky_golay_window + (savitzky_golay_window % 2 == 0)
    
    # 应用Savitzky-Golay平滑
    smoothed_data = savgol_filter(signal_data, window_size, savitzky_golay_polynomial, axis=0)
    
    # 应用ALS基线校正
    baseline_corrected_data = smoothed_data - np.apply_along_axis(
        perform_als_baseline_correction, 0, smoothed_data, als_lambda, als_asymmetry
    )
    
    processed_dataframe[signal_columns] = baseline_corrected_data
    return processed_dataframe

def detect_peaks_using_cwt(signal, minimum_signal_noise_ratio=None, noise_percentage=None, wavelet_widths=None):
    """使用连续小波变换(CWT)检测色谱峰
    
    基于多尺度小波分析，实现对不同宽度色谱峰的有效检测。
    
    Args:
        signal: 输入信号
        minimum_signal_noise_ratio: 最小信噪比阈值
        noise_percentage: 噪声水平百分比
        wavelet_widths: 小波尺度宽度范围
        
    Returns:
        detected_peaks: 检测到的峰位置索引
        cwt_matrix: 连续小波变换矩阵
    """
    if minimum_signal_noise_ratio is None:
        minimum_signal_noise_ratio = signal_processing_config['cwt_min_signal_noise_ratio']
    if noise_percentage is None:
        noise_percentage = signal_processing_config['cwt_noise_percentage']
    
    if wavelet_widths is None:
        signal_length = len(signal)
        min_width = max(1, signal_length // 1000)
        max_width = min(50, signal_length // 20)
        wavelet_widths = np.arange(min_width, max_width + 1)
    
    # 执行连续小波变换
    cwt_matrix = pywt.cwt(signal, wavelet_widths, 'mexh')[0]
    peak_candidates = []
    
    # 在每个尺度上检测峰
    for scale_index, width in enumerate(wavelet_widths):
        scale_signal = cwt_matrix[scale_index, :]
        # 估计噪声水平
        noise_level = np.percentile(np.abs(scale_signal), noise_percentage)
        # 计算阈值
        threshold = minimum_signal_noise_ratio * noise_level
        # 检测峰
        peaks, _ = find_peaks(scale_signal, height=threshold, 
                              distance=max(1, width//2))
        
        # 记录峰候选
        for peak in peaks:
            peak_candidates.append({
                'position': peak,
                'intensity': scale_signal[peak],
                'width': width,
                'scale_index': scale_index
            })
    
    if not peak_candidates:
        return np.array([]), cwt_matrix
    
    # 按位置排序峰候选
    peak_candidates.sort(key=lambda x: x['position'])
    merged_peaks = []
    current_group = [peak_candidates[0]]
    # 计算合并距离
    merge_distance = max(3, np.mean(wavelet_widths))
    
    # 合并相邻的峰候选
    for candidate in peak_candidates[1:]:
        if candidate['position'] - current_group[-1]['position'] <= merge_distance:
            current_group.append(candidate)
        else:
            # 选择组内强度最大的峰
            best_peak = max(current_group, key=lambda x: x['intensity'])
            merged_peaks.append(best_peak['position'])
            current_group = [candidate]
    
    # 处理最后一组
    if current_group:
        best_peak = max(current_group, key=lambda x: x['intensity'])
        merged_peaks.append(best_peak['position'])
    
    # 验证峰位置
    validated_peaks = []
    for peak_position in merged_peaks:
        start_index = max(0, peak_position - 2)
        end_index = min(len(signal), peak_position + 3)
        local_signal = signal[start_index:end_index]
        if len(local_signal) > 0:
            local_max_index = np.argmax(local_signal) + start_index
            if abs(local_max_index - peak_position) <= 2:
                validated_peaks.append(local_max_index)
            else:
                validated_peaks.append(peak_position)
    
    return np.array(sorted(set(validated_peaks))), cwt_matrix

def detect_peak_regions_using_cwt(signal, minimum_signal_noise_ratio=None, noise_percentage=None, merge_distance=None):
    """检测色谱峰区域
    
    基于CWT峰检测结果，识别峰区域并进行合并。
    
    Args:
        signal: 输入信号
        minimum_signal_noise_ratio: 最小信噪比阈值
        noise_percentage: 噪声水平百分比
        merge_distance: 峰区域合并距离
        
    Returns:
        peak_regions: 峰区域列表，每个区域包含起始索引、结束索引和峰位置
        cwt_matrix: 连续小波变换矩阵
    """
    if minimum_signal_noise_ratio is None:
        minimum_signal_noise_ratio = signal_processing_config['cwt_min_signal_noise_ratio']
    if noise_percentage is None:
        noise_percentage = signal_processing_config['cwt_noise_percentage']
    if merge_distance is None:
        merge_distance = signal_processing_config['peak_merge_distance']
    
    # 检测峰位置
    peak_indices, cwt_matrix = detect_peaks_using_cwt(signal, minimum_signal_noise_ratio, noise_percentage)
    
    if len(peak_indices) == 0:
        return [], cwt_matrix
    
    def estimate_peak_width(signal, peak_index):
        """估计峰的半峰宽"""
        peak_height = signal[peak_index]
        half_height = peak_height / 2
        left_index = peak_index
        while left_index > 0 and signal[left_index] > half_height:
            left_index -= 1
        right_index = peak_index
        while right_index < len(signal) - 1 and signal[right_index] > half_height:
            right_index += 1
        return max(5, right_index - left_index)
    
    # 初始化峰区域
    initial_regions = []
    for peak_index in peak_indices:
        peak_width = estimate_peak_width(signal, peak_index)
        half_width = max(peak_width, 10)
        start_index = max(0, peak_index - half_width)
        end_index = min(len(signal) - 1, peak_index + half_width)
        initial_regions.append((start_index, end_index, [peak_index]))
    
    if len(initial_regions) <= 1:
        return initial_regions, cwt_matrix
    
    # 合并重叠的峰区域
    merged_regions = []
    current_start, current_end, current_peaks = initial_regions[0]
    
    for start, end, peaks in initial_regions[1:]:
        if start <= current_end + merge_distance:
            # 合并区域
            current_end = max(current_end, end)
            current_peaks.extend(peaks)
        else:
            # 保存当前区域并开始新区域
            merged_regions.append((current_start, current_end, sorted(current_peaks)))
            current_start, current_end, current_peaks = start, end, peaks
    
    # 添加最后一个区域
    merged_regions.append((current_start, current_end, sorted(current_peaks)))
    return merged_regions, cwt_matrix

def calculate_peak_resolution(signal, peak1_index, peak2_index):
    """计算两个色谱峰之间的分离度
    
    基于半峰宽(FWHM)和保留时间差计算分离度，
    采用行业标准公式：Rs = 1.18 * Δt / (W1 + W2)
    
    Args:
        signal: 输入信号
        peak1_index: 第一个峰的索引
        peak2_index: 第二个峰的索引
        
    Returns:
        resolution: 分离度值
    """
    # 确保峰1在峰2之前
    if peak1_index > peak2_index:
        peak1_index, peak2_index = peak2_index, peak1_index
    
    def calculate_full_width_at_half_maximum(signal, peak_index):
        """计算峰的半峰宽(FWHM)"""
        if peak_index < 0 or peak_index >= len(signal):
            return 10  # 默认值
        peak_height = signal[peak_index]
        half_height = peak_height / 2
        left_index = peak_index
        while left_index > 0 and signal[left_index] > half_height:
            left_index -= 1
        right_index = peak_index
        while right_index < len(signal) - 1 and signal[right_index] > half_height:
            right_index += 1
        fwhm = max(1, right_index - left_index)
        return fwhm
    
    # 计算两个峰的半峰宽
    fwhm1 = calculate_full_width_at_half_maximum(signal, peak1_index)
    fwhm2 = calculate_full_width_at_half_maximum(signal, peak2_index)
    # 计算保留时间差
    retention_time_difference = abs(peak2_index - peak1_index)
    # 计算平均半峰宽
    average_width = (fwhm1 + fwhm2) / 2
    
    if average_width == 0:
        return 0
    
    # 计算分离度
    resolution = 1.18 * retention_time_difference / average_width
    return resolution

def calculate_all_peak_resolutions(signal, peak_regions):
    """计算所有相邻峰对的分离度
    
    分析所有相邻峰对的分离度，并评估分离状态。
    
    Args:
        signal: 输入信号
        peak_regions: 峰区域列表
        
    Returns:
        resolution_data: 包含所有峰对分离度信息的列表
    """
    resolution_data = []
    all_peaks = []
    
    # 收集所有峰的位置
    for start_index, end_index, peak_indices in peak_regions:
        all_peaks.extend(peak_indices)
    
    if len(all_peaks) < 2:
        return resolution_data
    
    # 按位置排序峰
    sorted_peaks = sorted(all_peaks)
    
    # 计算相邻峰对的分离度
    for i in range(len(sorted_peaks) - 1):
        peak1_index = sorted_peaks[i]
        peak2_index = sorted_peaks[i + 1]
        resolution = calculate_peak_resolution(signal, peak1_index, peak2_index)
        
        # 评估分离状态
        if resolution >= 1.5:
            separation_status = '基线分离'
        elif resolution >= 1:
            separation_status = '部分分离'
        else:
            separation_status = '未分离'
        
        # 确定峰所在的区域
        peak1_region = None
        peak2_region = None
        
        for region_index, (start_index, end_index, peak_indices) in enumerate(peak_regions):
            if peak1_index in peak_indices:
                peak1_region = region_index
            if peak2_index in peak_indices:
                peak2_region = region_index
        
        # 判断是否为跨区域峰对
        is_inter_region = peak1_region != peak2_region
        
        resolution_data.append({
            'peak1_index': peak1_index,
            'peak2_index': peak2_index,
            'peak1_region': peak1_region,
            'peak2_region': peak2_region,
            'is_inter_region': is_inter_region,
            'resolution': resolution,
            'separation_status': separation_status
        })
    
    return resolution_data

def calculate_enhanced_chromatographic_resolution_factor(signal, peak_regions, resolution_requirement=None, 
                                                      separation_ratio_weight=None):
    """计算增强型色谱分辨率因子(ECRF)
    
    综合考虑分离度和分离比例，评估色谱分离效果。
    
    Args:
        signal: 输入信号
        peak_regions: 峰区域列表
        resolution_requirement: 要求的分离度阈值
        separation_ratio_weight: 分离比例权重
        
    Returns:
        enhanced_crf: 增强型色谱分辨率因子
    """
    if resolution_requirement is None:
        resolution_requirement = signal_processing_config['crf_resolution_requirement']
    if separation_ratio_weight is None:
        separation_ratio_weight = signal_processing_config['crf_separation_weight']
    
    # 计算所有峰对的分离度
    all_resolutions_data = calculate_all_peak_resolutions(signal, peak_regions)
    all_resolutions = [res['resolution'] for res in all_resolutions_data]
    total_peaks = sum(len(peak_indices) for _, _, peak_indices in peak_regions)
    
    if len(all_resolutions) == 0 or total_peaks < 2:
        return 0.0
    
    # 计算分辨率质量分数
    clipped_resolutions = [np.clip(res, 0, resolution_requirement) for res in all_resolutions]
    clipped_sum = sum(clipped_resolutions)
    max_possible_sum = resolution_requirement * (total_peaks - 1)
    
    if max_possible_sum == 0:
        resolution_quality_score = 0.0
    else:
        resolution_quality_score = clipped_sum / max_possible_sum
    
    # 计算基线分离比例奖励
    baseline_separated_pairs = sum(1 for res in all_resolutions_data if res['resolution'] >= 1.5)
    total_pairs = len(all_resolutions_data)
    
    if total_pairs <= 0:
        separation_ratio_bonus = 1.0
    else:
        baseline_ratio = baseline_separated_pairs / total_pairs
        if baseline_ratio <= 0:
            separation_factor = 0.1
        else:
            separation_factor = np.log(1 + baseline_ratio) / np.log(2)
            separation_factor = np.clip(separation_factor, 0.1, 2.0)
        separation_ratio_bonus = separation_factor * (total_pairs / 100)
    
    # 计算增强型CRF
    enhanced_crf = resolution_quality_score * separation_ratio_bonus
    return enhanced_crf


def calculate_ecrf_comprehensive(signal, peak_regions, analysis_time, current_iteration, total_iterations, 
                               reference_peak_count=1, resolution_requirement=1.5, 
                               minimum_analysis_time=80, maximum_analysis_time=300):
    """
    计算综合ECRF指标，考虑分离效果和分析时间
    
    基于分离度、分离比例和分析时间，计算综合评分指标，
    用于评估GC升温程序的整体性能。
    
    Args:
    - signal: 色谱信号数据
    - peak_regions: 峰区域列表
    - analysis_time: 分析时间（秒）
    - current_iteration: 当前迭代次数
    - total_iterations: 总迭代次数
    - reference_peak_count: 初始方法检测到的峰数量
    - resolution_requirement: 要求的分离度阈值
    - minimum_analysis_time: 最短允许分析时间
    - maximum_analysis_time: 最长允许分析时间
    
    Returns:
    - ecrf: 综合ECRF评分
    """
    # 计算分离度相关数据
    all_resolutions_data = calculate_all_peak_resolutions(signal, peak_regions)
    all_resolutions = [res['resolution'] for res in all_resolutions_data]
    observed_peak_count = sum(len(peak_indices) for _, _, peak_indices in peak_regions)  # 检测到的色谱峰总数
    peak_pair_count = len(all_resolutions)  # 相邻峰对数量
    
    if observed_peak_count < 2 or peak_pair_count == 0:
        return 0.0
    
    # 计算分离效果因子 f_sep = S_res * B_sep
    # S_res: 分辨率评分
    clipped_resolutions = [min(res, resolution_requirement) for res in all_resolutions]
    resolution_score = sum(clipped_resolutions) / (resolution_requirement * (observed_peak_count - 1))
    
    # B_sep: 基线分离比例评分
    baseline_separated_pairs = sum(1 for res in all_resolutions if res >= resolution_requirement)  # 基线分离的峰对数
    total_pairs = peak_pair_count  # 总峰对数
    if total_pairs > 0 and reference_peak_count > 0:
        baseline_separation_score = (np.log(1 + baseline_separated_pairs / total_pairs) / np.log(2)) * (total_pairs / reference_peak_count)
    else:
        baseline_separation_score = 0.0
    
    separation_factor = resolution_score * baseline_separation_score
    
    # 计算时间因子 f_time
    time_factor = 1 - (analysis_time - minimum_analysis_time) / (maximum_analysis_time - minimum_analysis_time)
    # 限制时间因子在合理范围内 [0, 1]
    time_factor = np.clip(time_factor, 0.0, 1.0)
    
    # 计算权重（随迭代次数调整）
    separation_weight = 0.7 + 0.3 * (1 - current_iteration / total_iterations)  # 初期更重视分离效果
    time_weight = 1 - separation_weight
    
    # 计算综合ECRF
    ecrf = separation_weight * separation_factor + time_weight * time_factor
    
    return ecrf