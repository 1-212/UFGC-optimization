"""ASC文件解析模块"""

import os
import pandas as pd
from core.signal_process import preprocess, detect_peak_regions_cwt, calculate_enhanced_crf
from config import SIGNAL_CONFIG

def process_asc_files(asc_files, use_half=True):
    """处理ASC文件"""
    all_data = {}
    max_rows = 0
    
    for filepath in asc_files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
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
            
            all_data[filename] = file_data
            max_rows = max(max_rows, len(file_data))
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
    
    if not all_data:
        return None
    
    if use_half:
        max_rows = max_rows // 2
    
    time_values = [i * 0.01 for i in range(max_rows)]
    df_data = {'时间': time_values}
    
    for filename, data in all_data.items():
        column_name = filename.replace('.asc', '')
        if use_half:
            data = data[:max_rows]
        padded_data = data + [None] * (max_rows - len(data))
        df_data[column_name] = padded_data
    
    df = pd.DataFrame(df_data)
    return df

def calculate_crf_from_asc_data(df):
    """从ASC数据计算CRF"""
    data_cols = [col for col in df.columns if col != '时间']
    
    if not data_cols:
        return 0.0
    
    crf_values = []
    
    for col in data_cols:
        signal = df[col].dropna().values
        if len(signal) > 10:
            preprocessed_signal = preprocess(
                pd.DataFrame({'time': range(len(signal)), col: signal}),
                sg_win=SIGNAL_CONFIG['sg_window'],
                sg_poly=SIGNAL_CONFIG['sg_poly'],
                lam=SIGNAL_CONFIG['als_lambda'],
                p=SIGNAL_CONFIG['als_p']
            )[col].values
            
            regions, _ = detect_peak_regions_cwt(
                preprocessed_signal,
                min_snr=SIGNAL_CONFIG['cwt_min_snr'],
                noise_perc=SIGNAL_CONFIG['cwt_noise_perc'],
                merge_distance=SIGNAL_CONFIG['peak_merge_distance']
            )
            
            crf = calculate_enhanced_crf(
                preprocessed_signal, regions,
                rs_req=SIGNAL_CONFIG['crf_rs_req'],
                separation_ratio_weight=SIGNAL_CONFIG['crf_separation_weight']
            )
            crf_values.append(crf)
    
    if crf_values:
        return sum(crf_values) / len(crf_values)
    else:
        return 0.0


def calculate_signals_and_regions_from_asc_data(df):
    """从ASC数据计算信号和峰区域，用于新ECRF计算"""
    data_cols = [col for col in df.columns if col != '时间']
    
    if not data_cols:
        return None, []
    
    # 使用第一个有效信号进行分析
    for col in data_cols:
        signal = df[col].dropna().values
        if len(signal) > 10:
            preprocessed_signal = preprocess(
                pd.DataFrame({'time': range(len(signal)), col: signal}),
                sg_win=SIGNAL_CONFIG['sg_window'],
                sg_poly=SIGNAL_CONFIG['sg_poly'],
                lam=SIGNAL_CONFIG['als_lambda'],
                p=SIGNAL_CONFIG['als_p']
            )[col].values
            
            regions, _ = detect_peak_regions_cwt(
                preprocessed_signal,
                min_snr=SIGNAL_CONFIG['cwt_min_snr'],
                noise_perc=SIGNAL_CONFIG['cwt_noise_perc'],
                merge_distance=SIGNAL_CONFIG['peak_merge_distance']
            )
            
            return preprocessed_signal, regions
    
    return None, []