"""GC chromatographic data ASC file parsing module"""


import os
import pandas as pd
from core.signal_process import (
    preprocess_chromatogram_data, 
    detect_peak_regions_using_cwt, 
    calculate_enhanced_chromatographic_resolution_factor
)
from config import signal_processing_config

def process_asc_files(asc_file_paths, use_half_data=True):
    """Process ASC files
    
    Parse ASC format chromatographic data files, extract signal data and convert to DataFrame format.
    
    Args:
        asc_file_paths: List of ASC file paths
        use_half_data: Whether to use half data (for faster processing)
        
    Returns:
        pd.DataFrame: DataFrame containing chromatographic data
    """
    all_signal_data = {}
    max_data_points = 0
    
    for file_path in asc_file_paths:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # Skip first 13 lines (file header information)
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
            print(f"Error processing file {filename}: {error}")
    
    if not all_signal_data:
        return None
    
    # Optional: Use only half data for faster processing
    if use_half_data:
        max_data_points = max_data_points // 2
    
    # Generate time values (assuming 0.01 second sampling interval)
    time_values = [i * 0.01 for i in range(max_data_points)]
    dataframe_data = {'time': time_values}
    
    # Build DataFrame
    for filename, data in all_signal_data.items():
        column_name = filename.replace('.asc', '')
        if use_half_data:
            data = data[:max_data_points]
        # Pad missing data
        padded_data = data + [None] * (max_data_points - len(data))
        dataframe_data[column_name] = padded_data
    
    dataframe = pd.DataFrame(dataframe_data)
    return dataframe

def calculate_crf_from_asc_data(dataframe):
    """Calculate Chromatographic Resolution Factor (CRF) from ASC data
    
    Preprocess and perform peak analysis on data parsed from ASC files, calculate CRF values.
    
    Args:
        dataframe: DataFrame containing chromatographic data
        
    Returns:
        float: Average CRF value
    """
    # Extract signal columns (exclude time column)
    signal_columns = [col for col in dataframe.columns if col != 'time']
    
    if not signal_columns:
        return 0.0
    
    crf_values = []
    
    # Calculate CRF for each signal column
    for column in signal_columns:
        signal = dataframe[column].dropna().values
        if len(signal) > 10:  # Ensure signal length is sufficient
            # Preprocess signal
            preprocessed_signal = preprocess_chromatogram_data(
                pd.DataFrame({'time': range(len(signal)), column: signal}),
                savitzky_golay_window=signal_processing_config['savitzky_golay_window'],
                savitzky_golay_polynomial=signal_processing_config['savitzky_golay_polynomial'],
                als_lambda=signal_processing_config['als_lambda'],
                als_asymmetry=signal_processing_config['als_p']
            )[column].values
            
            # Detect peak regions
            peak_regions, _ = detect_peak_regions_using_cwt(
                preprocessed_signal,
                minimum_signal_noise_ratio=signal_processing_config['cwt_min_signal_noise_ratio'],
                noise_percentage=signal_processing_config['cwt_noise_percentage'],
                merge_distance=signal_processing_config['peak_merge_distance']
            )
            
            # Calculate enhanced CRF
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
    """Extract signals and peak regions from ASC data
    
    Extract signals and peak regions for new ECRF calculation.
    
    Args:
        dataframe: DataFrame containing chromatographic data
        
    Returns:
        tuple: (Preprocessed signal, Peak regions list)
    """
    # Extract signal columns (exclude time column)
    signal_columns = [col for col in dataframe.columns if col != 'time']
    
    if not signal_columns:
        return None, []
    
    # Use first valid signal for analysis
    for column in signal_columns:
        signal = dataframe[column].dropna().values
        if len(signal) > 10:  # Ensure signal length is sufficient
            # Preprocess signal
            preprocessed_signal = preprocess_chromatogram_data(
                pd.DataFrame({'time': range(len(signal)), column: signal}),
                savitzky_golay_window=signal_processing_config['savitzky_golay_window'],
                savitzky_golay_polynomial=signal_processing_config['savitzky_golay_polynomial'],
                als_lambda=signal_processing_config['als_lambda'],
                als_asymmetry=signal_processing_config['als_p']
            )[column].values
            
            # Detect peak regions
            peak_regions, _ = detect_peak_regions_using_cwt(
                preprocessed_signal,
                minimum_signal_noise_ratio=signal_processing_config['cwt_min_signal_noise_ratio'],
                noise_percentage=signal_processing_config['cwt_noise_percentage'],
                merge_distance=signal_processing_config['peak_merge_distance']
            )
            
            return preprocessed_signal, peak_regions
    
    return None, []