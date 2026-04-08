"""GC色谱数据ASC文件解析模块

本模块实现了ASC格式色谱数据文件的解析和处理功能，
包括文件读取、数据预处理、峰值检测和CRF计算等功能。
基于标准的ASC文件格式，提取色谱信号数据并进行分析。

作者: 研究团队
日期: 2026年
"""

import os
import pandas as pd
from core.signal_process import (
    preprocess_chromatogram_data, 
    detect_peak_regions_using_cwt, 
    calculate_enhanced_chromatographic_resolution_factor
)
from config import signal_processing_config

def process_asc_files(asc_file_paths, use_half_data=True):
    """处理ASC文件
    
    解析ASC格式的色谱数据文件，提取信号数据并转换为DataFrame格式。
    
    Args:
        asc_file_paths: ASC文件路径列表
        use_half_data: 是否使用一半数据（用于加速处理）
        
    Returns:
        pd.DataFrame: 包含色谱数据的DataFrame
    """
    all_signal_data = {}
    max_data_points = 0
    
    for file_path in asc_file_paths:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # 跳过前13行（文件头信息）
            data_lines = lines[13:]
            file_data = []
            
            for line in data_lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            value = float(parts[1])
                            file_data.append(value)
                        except ValueError:
                            continue
                    elif len(parts) == 1:
                        try:
                            value = float(parts[0])
                            file_data.append(value)
                        except ValueError:
                            continue
            
            all_signal_data[filename] = file_data
            max_data_points = max(max_data_points, len(file_data))
        except Exception as error:
            print(f"处理文件 {filename} 时出错: {error}")
    
    if not all_signal_data:
        return None
    
    # 可选：只使用一半数据以加速处理
    if use_half_data:
        max_data_points = max_data_points // 2
    
    # 生成时间值（假设采样间隔为0.01秒）
    time_values = [i * 0.01 for i in range(max_data_points)]
    dataframe_data = {'时间': time_values}
    
    # 构建DataFrame
    for filename, data in all_signal_data.items():
        column_name = filename.replace('.asc', '')
        if use_half_data:
            data = data[:max_data_points]
        # 填充缺失数据
        padded_data = data + [None] * (max_data_points - len(data))
        dataframe_data[column_name] = padded_data
    
    dataframe = pd.DataFrame(dataframe_data)
    return dataframe

def calculate_crf_from_asc_data(dataframe):
    """从ASC数据计算色谱分辨率因子(CRF)
    
    对ASC文件解析得到的数据进行预处理和峰值分析，计算CRF值。
    
    Args:
        dataframe: 包含色谱数据的DataFrame
        
    Returns:
        float: 平均CRF值
    """
    # 提取信号列（排除时间列）
    signal_columns = [col for col in dataframe.columns if col != '时间']
    
    if not signal_columns:
        return 0.0
    
    crf_values = []
    
    # 对每个信号列计算CRF
    for column in signal_columns:
        signal = dataframe[column].dropna().values
        if len(signal) > 10:  # 确保信号长度足够
            # 预处理信号
            preprocessed_signal = preprocess_chromatogram_data(
                pd.DataFrame({'time': range(len(signal)), column: signal}),
                savitzky_golay_window=signal_processing_config['savitzky_golay_window'],
                savitzky_golay_polynomial=signal_processing_config['savitzky_golay_polynomial'],
                als_lambda=signal_processing_config['als_lambda'],
                als_asymmetry=signal_processing_config['als_p']
            )[column].values
            
            # 检测峰区域
            peak_regions, _ = detect_peak_regions_using_cwt(
                preprocessed_signal,
                minimum_signal_noise_ratio=signal_processing_config['cwt_min_signal_noise_ratio'],
                noise_percentage=signal_processing_config['cwt_noise_percentage'],
                merge_distance=signal_processing_config['peak_merge_distance']
            )
            
            # 计算增强型CRF
            crf = calculate_enhanced_chromatographic_resolution_factor(
                preprocessed_signal, peak_regions,
                resolution_requirement=signal_processing_config['crf_resolution_requirement'],
                separation_ratio_weight=signal_processing_config['crf_separation_weight']
            )
            crf_values.append(crf)
    
    if crf_values:
        return sum(crf_values) / len(crf_values)
    else:
        return 0.0


def extract_signals_and_regions_from_asc_data(dataframe):
    """从ASC数据提取信号和峰区域
    
    用于新ECRF计算的信号和峰区域提取。
    
    Args:
        dataframe: 包含色谱数据的DataFrame
        
    Returns:
        tuple: (预处理后的信号, 峰区域列表)
    """
    # 提取信号列（排除时间列）
    signal_columns = [col for col in dataframe.columns if col != '时间']
    
    if not signal_columns:
        return None, []
    
    # 使用第一个有效信号进行分析
    for column in signal_columns:
        signal = dataframe[column].dropna().values
        if len(signal) > 10:  # 确保信号长度足够
            # 预处理信号
            preprocessed_signal = preprocess_chromatogram_data(
                pd.DataFrame({'time': range(len(signal)), column: signal}),
                savitzky_golay_window=signal_processing_config['savitzky_golay_window'],
                savitzky_golay_polynomial=signal_processing_config['savitzky_golay_polynomial'],
                als_lambda=signal_processing_config['als_lambda'],
                als_asymmetry=signal_processing_config['als_p']
            )[column].values
            
            # 检测峰区域
            peak_regions, _ = detect_peak_regions_using_cwt(
                preprocessed_signal,
                minimum_signal_noise_ratio=signal_processing_config['cwt_min_signal_noise_ratio'],
                noise_percentage=signal_processing_config['cwt_noise_percentage'],
                merge_distance=signal_processing_config['peak_merge_distance']
            )
            
            return preprocessed_signal, peak_regions
    
    return None, []