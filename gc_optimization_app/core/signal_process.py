"""信号处理模块"""

import numpy as np
import pywt
from scipy.signal import savgol_filter, find_peaks
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from config import SIGNAL_CONFIG

def als_baseline(y, lam=1e5, p=0.01, max_iter=10):
    """ALS基线校正"""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    
    for i in range(max_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w_prev = w.copy()
        w = p * (y > z) + (1 - p) * (y < z)
        if np.linalg.norm(w - w_prev) < 1e-6:
            break
    return z

def preprocess(df, sg_win=None, sg_poly=None, lam=None, p=None):
    """预处理"""
    if sg_win is None:
        sg_win = SIGNAL_CONFIG['sg_window']
    if sg_poly is None:
        sg_poly = SIGNAL_CONFIG['sg_poly']
    if lam is None:
        lam = SIGNAL_CONFIG['als_lambda']
    if p is None:
        p = SIGNAL_CONFIG['als_p']
    
    df = df.copy()
    cols = df.columns[1:]
    X = df[cols].values
    win = sg_win + (sg_win % 2 == 0)
    X_s = savgol_filter(X, win, sg_poly, axis=0)
    X_b = X_s - np.apply_along_axis(als_baseline, 0, X_s, lam, p)
    df[cols] = X_b
    return df

def cwt_peak_detection(signal, min_snr=None, noise_perc=None, widths=None):
    """CWT峰检测"""
    if min_snr is None:
        min_snr = SIGNAL_CONFIG['cwt_min_snr']
    if noise_perc is None:
        noise_perc = SIGNAL_CONFIG['cwt_noise_perc']
    
    if widths is None:
        signal_length = len(signal)
        min_width = max(1, signal_length // 1000)
        max_width = min(50, signal_length // 20)
        widths = np.arange(min_width, max_width + 1)
    
    cwt_matrix = pywt.cwt(signal, widths, 'mexh')[0]
    peak_candidates = []
    
    for i, width in enumerate(widths):
        scale_signal = cwt_matrix[i, :]
        noise_level = np.percentile(np.abs(scale_signal), noise_perc)
        threshold = min_snr * noise_level
        peaks, properties = find_peaks(scale_signal, height=threshold, 
                                       distance=max(1, width//2))
        
        for peak in peaks:
            peak_candidates.append({
                'position': peak,
                'intensity': scale_signal[peak],
                'width': width,
                'scale_idx': i
            })
    
    if not peak_candidates:
        return np.array([]), cwt_matrix
    
    peak_candidates.sort(key=lambda x: x['position'])
    merged_peaks = []
    current_group = [peak_candidates[0]]
    merge_distance = max(3, np.mean(widths))
    
    for candidate in peak_candidates[1:]:
        if candidate['position'] - current_group[-1]['position'] <= merge_distance:
            current_group.append(candidate)
        else:
            best_peak = max(current_group, key=lambda x: x['intensity'])
            merged_peaks.append(best_peak['position'])
            current_group = [candidate]
    
    if current_group:
        best_peak = max(current_group, key=lambda x: x['intensity'])
        merged_peaks.append(best_peak['position'])
    
    validated_peaks = []
    for peak_pos in merged_peaks:
        start_idx = max(0, peak_pos - 2)
        end_idx = min(len(signal), peak_pos + 3)
        local_signal = signal[start_idx:end_idx]
        if len(local_signal) > 0:
            local_max_idx = np.argmax(local_signal) + start_idx
            if abs(local_max_idx - peak_pos) <= 2:
                validated_peaks.append(local_max_idx)
            else:
                validated_peaks.append(peak_pos)
    
    return np.array(sorted(set(validated_peaks))), cwt_matrix

def detect_peak_regions_cwt(signal, min_snr=None, noise_perc=None, merge_distance=None):
    """检测峰区域"""
    if min_snr is None:
        min_snr = SIGNAL_CONFIG['cwt_min_snr']
    if noise_perc is None:
        noise_perc = SIGNAL_CONFIG['cwt_noise_perc']
    if merge_distance is None:
        merge_distance = SIGNAL_CONFIG['peak_merge_distance']
    
    peak_indices, cwt_matrix = cwt_peak_detection(signal, min_snr, noise_perc)
    
    if len(peak_indices) == 0:
        return [], cwt_matrix
    
    def estimate_peak_width(signal, peak_idx):
        peak_height = signal[peak_idx]
        half_height = peak_height / 2
        left_idx = peak_idx
        while left_idx > 0 and signal[left_idx] > half_height:
            left_idx -= 1
        right_idx = peak_idx
        while right_idx < len(signal) - 1 and signal[right_idx] > half_height:
            right_idx += 1
        return max(5, right_idx - left_idx)
    
    initial_regions = []
    for peak_idx in peak_indices:
        peak_width = estimate_peak_width(signal, peak_idx)
        half_width = max(peak_width, 10)
        start_idx = max(0, peak_idx - half_width)
        end_idx = min(len(signal) - 1, peak_idx + half_width)
        initial_regions.append((start_idx, end_idx, [peak_idx]))
    
    if len(initial_regions) <= 1:
        return initial_regions, cwt_matrix
    
    merged_regions = []
    current_start, current_end, current_peaks = initial_regions[0]
    
    for start, end, peaks in initial_regions[1:]:
        if start <= current_end + merge_distance:
            current_end = max(current_end, end)
            current_peaks.extend(peaks)
        else:
            merged_regions.append((current_start, current_end, sorted(current_peaks)))
            current_start, current_end, current_peaks = start, end, peaks
    
    merged_regions.append((current_start, current_end, sorted(current_peaks)))
    return merged_regions, cwt_matrix

def calculate_peak_resolution(signal, peak1_idx, peak2_idx):
    """计算分离度"""
    if peak1_idx > peak2_idx:
        peak1_idx, peak2_idx = peak2_idx, peak1_idx
    
    def calculate_fwhm(signal, peak_idx):
        if peak_idx < 0 or peak_idx >= len(signal):
            return 10
        peak_height = signal[peak_idx]
        half_height = peak_height / 2
        left_idx = peak_idx
        while left_idx > 0 and signal[left_idx] > half_height:
            left_idx -= 1
        right_idx = peak_idx
        while right_idx < len(signal) - 1 and signal[right_idx] > half_height:
            right_idx += 1
        fwhm = max(1, right_idx - left_idx)
        return fwhm
    
    fwhm1 = calculate_fwhm(signal, peak1_idx)
    fwhm2 = calculate_fwhm(signal, peak2_idx)
    delta_rt = abs(peak2_idx - peak1_idx)
    avg_width = (fwhm1 + fwhm2) / 2
    
    if avg_width == 0:
        return 0
    
    resolution = 1.18 * delta_rt / avg_width
    return resolution

def calculate_all_resolutions(signal, regions):
    """计算所有分离度"""
    resolutions = []
    all_peaks = []
    for start_idx, end_idx, peak_indices in regions:
        all_peaks.extend(peak_indices)
    
    if len(all_peaks) < 2:
        return resolutions
    
    sorted_peaks = sorted(all_peaks)
    
    for i in range(len(sorted_peaks) - 1):
        peak1_idx = sorted_peaks[i]
        peak2_idx = sorted_peaks[i + 1]
        resolution = calculate_peak_resolution(signal, peak1_idx, peak2_idx)
        
        if resolution >= 1.5:
            status = '基线分离'
        elif resolution >= 1:
            status = '部分分离'
        else:
            status = '未分离'
        
        peak1_region = None
        peak2_region = None
        
        for region_idx, (start_idx, end_idx, peak_indices) in enumerate(regions):
            if peak1_idx in peak_indices:
                peak1_region = region_idx
            if peak2_idx in peak_indices:
                peak2_region = region_idx
        
        is_inter_region = peak1_region != peak2_region
        
        resolutions.append({
            'peak1_idx': peak1_idx,
            'peak2_idx': peak2_idx,
            'peak1_region': peak1_region,
            'peak2_region': peak2_region,
            'is_inter_region': is_inter_region,
            'resolution': resolution,
            'separation_status': status
        })
    
    return resolutions

def calculate_enhanced_crf(signal, regions, rs_req=None, separation_ratio_weight=None):
    """计算Enhanced CRF"""
    if rs_req is None:
        rs_req = SIGNAL_CONFIG['crf_rs_req']
    if separation_ratio_weight is None:
        separation_ratio_weight = SIGNAL_CONFIG['crf_separation_weight']
    
    all_resolutions_data = calculate_all_resolutions(signal, regions)
    all_resolutions = [res['resolution'] for res in all_resolutions_data]
    total_peaks = sum(len(peak_indices) for _, _, peak_indices in regions)
    
    if len(all_resolutions) == 0 or total_peaks < 2:
        return 0.0
    
    clipped_resolutions = [np.clip(res, 0, rs_req) for res in all_resolutions]
    clipped_sum = sum(clipped_resolutions)
    max_possible_sum = rs_req * (total_peaks - 1)
    
    if max_possible_sum == 0:
        resolution_quality_score = 0.0
    else:
        resolution_quality_score = clipped_sum / max_possible_sum
    
    baseline_pairs = sum(1 for res in all_resolutions_data if res['resolution'] >= 1.5)
    total_pairs = len(all_resolutions_data)
    
    if total_pairs <= 0:
        separation_ratio_bonus = 1.0
    else:
        baseline_ratio = baseline_pairs / total_pairs
        if baseline_ratio <= 0:
            separation_factor = 0.1
        else:
            separation_factor = np.log(1 + baseline_ratio) / np.log(2)
            separation_factor = np.clip(separation_factor, 0.1, 2.0)
        separation_ratio_bonus = separation_factor * (total_pairs / 100)
    
    enhanced_crf = resolution_quality_score * separation_ratio_bonus
    return enhanced_crf


def calculate_new_ecrf(signal, regions, analysis_time, iteration_num, total_iterations, rs_req=1.5, t_min=80, t_max=300):
    """
    计算新的ECRF综合指标
    
    参数:
    - signal: 信号数据
    - regions: 峰区域
    - analysis_time: 分析时间
    - iteration_num: 当前迭代次数
    - total_iterations: 总迭代次数
    - rs_req: 要求的分离度 (默认1.5)
    - t_min: 最短允许分析时间
    - t_max: 最长允许分析时间
    """
    # 计算分离度相关数据
    all_resolutions_data = calculate_all_resolutions(signal, regions)
    all_resolutions = [res['resolution'] for res in all_resolutions_data]
    n_obs = sum(len(peak_indices) for _, _, peak_indices in regions)  # 检测到的色谱峰总数
    n_pairs = len(all_resolutions)  # 相邻峰对数量
    
    if n_obs < 2 or n_pairs == 0:
        return 0.0
    
    # 计算 f_sep = S_res * B_sep
    # S_res = sum(min(Rs_i, Rs_req)) / (Rs_req * (n_obs - 1))
    clipped_resolutions = [min(res, rs_req) for res in all_resolutions]
    S_res = sum(clipped_resolutions) / (rs_req * (n_obs - 1))
    
    # B_sep = log(1 + n_base / n_total) / log(2)
    n_base = sum(1 for res in all_resolutions if res >= rs_req)  # 基线分离的峰对数
    n_total = n_pairs  # 总峰对数
    if n_total > 0:
        B_sep = np.log(1 + n_base / n_total) / np.log(2)
    else:
        B_sep = 0.0
    
    f_sep = S_res * B_sep
    
    # 计算 f_time = 1 - (t - T_min) / (T_max - T_min)
    f_time = 1 - (analysis_time - t_min) / (t_max - t_min)
    # 限制f_time在合理范围内 [0, 1]
    f_time = np.clip(f_time, 0.0, 1.0)
    
    # 计算权重
    w_sep = 0.7 + 0.3 * (1 - iteration_num / total_iterations)  # 权重随迭代次数变化
    w_time = 1 - w_sep
    
    # 计算ECRF
    ecrf = w_sep * f_sep + w_time * f_time
    
    return ecrf