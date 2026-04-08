"""Signal processing module for GC temperature program optimization system"""


import numpy as np
import pywt
from scipy.signal import savgol_filter, find_peaks
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from config import signal_processing_config

def perform_als_baseline_correction(signal, lambda_value=1e5, asymmetry_parameter=0.01, max_iterations=10):
    """Perform baseline correction using Asymmetric Least Squares (ALS) algorithm
    
    Based on the algorithm by Eilers and Boelens (2005), iteratively optimizes baseline estimation,
    suitable for chromatographic signal baseline correction.
    
    Args:
        signal: Input signal
        lambda_value: Smoothing parameter, controls baseline smoothness
        asymmetry_parameter: Asymmetry parameter, controls how closely baseline follows signal
        max_iterations: Maximum number of iterations
        
    Returns:
        baseline: Estimated baseline
    """
    signal_length = len(signal)
    # Create second-order difference matrix (for smoothing constraint)
    difference_matrix = sparse.diags([1, -2, 1], [0, -1, -2], shape=(signal_length, signal_length-2))
    weights = np.ones(signal_length)
    
    for _ in range(max_iterations):
        # Create weight matrix
        weight_matrix = sparse.spdiags(weights, 0, signal_length, signal_length)
        # Build system matrix
        system_matrix = weight_matrix + lambda_value * difference_matrix.dot(difference_matrix.transpose())
        # Solve linear system
        baseline = spsolve(system_matrix, weights * signal)
        # Update weights
        previous_weights = weights.copy()
        weights = asymmetry_parameter * (signal > baseline) + (1 - asymmetry_parameter) * (signal < baseline)
        # Check convergence
        if np.linalg.norm(weights - previous_weights) < 1e-6:
            break
    return baseline

def preprocess_chromatogram_data(dataframe, savitzky_golay_window=None, savitzky_golay_polynomial=None, 
                               als_lambda=None, als_asymmetry=None):
    """Preprocess chromatogram data
    
    Apply smoothing and baseline correction to chromatographic data to improve accuracy of subsequent analysis.
    
    Args:
        dataframe: DataFrame containing chromatographic data
        savitzky_golay_window: Savitzky-Golay smoothing window size
        savitzky_golay_polynomial: Savitzky-Golay polynomial order
        als_lambda: ALS baseline correction smoothing parameter
        als_asymmetry: ALS baseline correction asymmetry parameter
        
    Returns:
        processed_dataframe: Processed DataFrame
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
    # Get signal columns (assuming first column is time or index)
    signal_columns = processed_dataframe.columns[1:]
    signal_data = processed_dataframe[signal_columns].values
    
    # Ensure window size is odd
    window_size = savitzky_golay_window + (savitzky_golay_window % 2 == 0)
    
    # Apply Savitzky-Golay smoothing
    smoothed_data = savgol_filter(signal_data, window_size, savitzky_golay_polynomial, axis=0)
    
    # Apply ALS baseline correction
    baseline_corrected_data = smoothed_data - np.apply_along_axis(
        perform_als_baseline_correction, 0, smoothed_data, als_lambda, als_asymmetry
    )
    
    processed_dataframe[signal_columns] = baseline_corrected_data
    return processed_dataframe

def detect_peaks_using_cwt(signal, minimum_signal_noise_ratio=None, noise_percentage=None, wavelet_widths=None):
    """Detect chromatographic peaks using Continuous Wavelet Transform (CWT)
    
    Based on multi-scale wavelet analysis, effectively detects chromatographic peaks of different widths.
    
    Args:
        signal: Input signal
        minimum_signal_noise_ratio: Minimum signal-to-noise ratio threshold
        noise_percentage: Noise level percentage
        wavelet_widths: Wavelet scale width range
        
    Returns:
        detected_peaks: Detected peak position indices
        cwt_matrix: Continuous wavelet transform matrix
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
    
    # Perform continuous wavelet transform
    cwt_matrix = pywt.cwt(signal, wavelet_widths, 'mexh')[0]
    peak_candidates = []
    
    # Detect peaks at each scale
    for scale_index, width in enumerate(wavelet_widths):
        scale_signal = cwt_matrix[scale_index, :]
        # Estimate noise level
        noise_level = np.percentile(np.abs(scale_signal), noise_percentage)
        # Calculate threshold
        threshold = minimum_signal_noise_ratio * noise_level
        # Detect peaks
        peaks, _ = find_peaks(scale_signal, height=threshold, 
                              distance=max(1, width//2))
        
        # Record peak candidates
        for peak in peaks:
            peak_candidates.append({
                'position': peak,
                'intensity': scale_signal[peak],
                'width': width,
                'scale_index': scale_index
            })
    
    if not peak_candidates:
        return np.array([]), cwt_matrix
    
    # Sort peak candidates by position
    peak_candidates.sort(key=lambda x: x['position'])
    merged_peaks = []
    current_group = [peak_candidates[0]]
    # Calculate merge distance
    merge_distance = max(3, np.mean(wavelet_widths))
    
    # Merge adjacent peak candidates
    for candidate in peak_candidates[1:]:
        if candidate['position'] - current_group[-1]['position'] <= merge_distance:
            current_group.append(candidate)
        else:
            # Select peak with maximum intensity in group
            best_peak = max(current_group, key=lambda x: x['intensity'])
            merged_peaks.append(best_peak['position'])
            current_group = [candidate]
    
    # Process last group
    if current_group:
        best_peak = max(current_group, key=lambda x: x['intensity'])
        merged_peaks.append(best_peak['position'])
    
    # Validate peak positions
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
    """Detect chromatographic peak regions
    
    Based on CWT peak detection results, identify and merge peak regions.
    
    Args:
        signal: Input signal
        minimum_signal_noise_ratio: Minimum signal-to-noise ratio threshold
        noise_percentage: Noise level percentage
        merge_distance: Peak region merge distance
        
    Returns:
        peak_regions: List of peak regions, each containing start index, end index, and peak positions
        cwt_matrix: Continuous wavelet transform matrix
    """
    if minimum_signal_noise_ratio is None:
        minimum_signal_noise_ratio = signal_processing_config['cwt_min_signal_noise_ratio']
    if noise_percentage is None:
        noise_percentage = signal_processing_config['cwt_noise_percentage']
    if merge_distance is None:
        merge_distance = signal_processing_config['peak_merge_distance']
    
    # Detect peak positions
    peak_indices, cwt_matrix = detect_peaks_using_cwt(signal, minimum_signal_noise_ratio, noise_percentage)
    
    if len(peak_indices) == 0:
        return [], cwt_matrix
    
    def estimate_peak_width(signal, peak_index):
        """Estimate peak full width at half maximum"""
        peak_height = signal[peak_index]
        half_height = peak_height / 2
        left_index = peak_index
        while left_index > 0 and signal[left_index] > half_height:
            left_index -= 1
        right_index = peak_index
        while right_index < len(signal) - 1 and signal[right_index] > half_height:
            right_index += 1
        return max(5, right_index - left_index)
    
    # Initialize peak regions
    initial_regions = []
    for peak_index in peak_indices:
        peak_width = estimate_peak_width(signal, peak_index)
        half_width = max(peak_width, 10)
        start_index = max(0, peak_index - half_width)
        end_index = min(len(signal) - 1, peak_index + half_width)
        initial_regions.append((start_index, end_index, [peak_index]))
    
    if len(initial_regions) <= 1:
        return initial_regions, cwt_matrix
    
    # Merge overlapping peak regions
    merged_regions = []
    current_start, current_end, current_peaks = initial_regions[0]
    
    for start, end, peaks in initial_regions[1:]:
        if start <= current_end + merge_distance:
            # Merge regions
            current_end = max(current_end, end)
            current_peaks.extend(peaks)
        else:
            # Save current region and start new region
            merged_regions.append((current_start, current_end, sorted(current_peaks)))
            current_start, current_end, current_peaks = start, end, peaks
    
    # Add last region
    merged_regions.append((current_start, current_end, sorted(current_peaks)))
    return merged_regions, cwt_matrix

def calculate_peak_resolution(signal, peak1_index, peak2_index):
    """Calculate resolution between two chromatographic peaks
    
    Calculate resolution based on full width at half maximum (FWHM) and retention time difference,
    using industry standard formula: Rs = 1.18 * Δt / (W1 + W2)
    
    Args:
        signal: Input signal
        peak1_index: Index of first peak
        peak2_index: Index of second peak
        
    Returns:
        resolution: Resolution value
    """
    # Ensure peak1 is before peak2
    if peak1_index > peak2_index:
        peak1_index, peak2_index = peak2_index, peak1_index
    
    def calculate_full_width_at_half_maximum(signal, peak_index):
        """Calculate peak full width at half maximum (FWHM)"""
        if peak_index < 0 or peak_index >= len(signal):
            return 10  # Default value
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
    
    # Calculate FWHM for both peaks
    fwhm1 = calculate_full_width_at_half_maximum(signal, peak1_index)
    fwhm2 = calculate_full_width_at_half_maximum(signal, peak2_index)
    # Calculate retention time difference
    retention_time_difference = abs(peak2_index - peak1_index)
    # Calculate average width
    average_width = (fwhm1 + fwhm2) / 2
    
    if average_width == 0:
        return 0
    
    # Calculate resolution
    resolution = 1.18 * retention_time_difference / average_width
    return resolution

def calculate_all_peak_resolutions(signal, peak_regions):
    """Calculate resolutions for all adjacent peak pairs
    
    Analyze resolutions for all adjacent peak pairs and evaluate separation status.
    
    Args:
        signal: Input signal
        peak_regions: List of peak regions
        
    Returns:
        resolution_data: List containing resolution information for all peak pairs
    """
    resolution_data = []
    all_peaks = []
    
    # Collect all peak positions
    for start_index, end_index, peak_indices in peak_regions:
        all_peaks.extend(peak_indices)
    
    if len(all_peaks) < 2:
        return resolution_data
    
    # Sort peaks by position
    sorted_peaks = sorted(all_peaks)
    
    # Calculate resolution for adjacent peak pairs
    for i in range(len(sorted_peaks) - 1):
        peak1_index = sorted_peaks[i]
        peak2_index = sorted_peaks[i + 1]
        resolution = calculate_peak_resolution(signal, peak1_index, peak2_index)
        
        # Evaluate separation status
        if resolution >= 1.5:
            separation_status = 'Baseline separation'
        elif resolution >= 1:
            separation_status = 'Partial separation'
        else:
            separation_status = 'No separation'
        
        # Determine regions where peaks are located
        peak1_region = None
        peak2_region = None
        
        for region_index, (start_index, end_index, peak_indices) in enumerate(peak_regions):
            if peak1_index in peak_indices:
                peak1_region = region_index
            if peak2_index in peak_indices:
                peak2_region = region_index
        
        # Determine if peaks are in different regions
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
    """Calculate Enhanced Chromatographic Resolution Factor (ECRF)
    
    Comprehensive evaluation of chromatographic separation performance considering resolution and separation ratio.
    
    Args:
        signal: Input signal
        peak_regions: List of peak regions
        resolution_requirement: Required resolution threshold
        separation_ratio_weight: Separation ratio weight
        
    Returns:
        enhanced_crf: Enhanced chromatographic resolution factor
    """
    if resolution_requirement is None:
        resolution_requirement = signal_processing_config['crf_resolution_requirement']
    if separation_ratio_weight is None:
        separation_ratio_weight = signal_processing_config['crf_separation_weight']
    
    # Calculate resolutions for all peak pairs
    all_resolutions_data = calculate_all_peak_resolutions(signal, peak_regions)
    all_resolutions = [res['resolution'] for res in all_resolutions_data]
    total_peaks = sum(len(peak_indices) for _, _, peak_indices in peak_regions)
    
    if len(all_resolutions) == 0 or total_peaks < 2:
        return 0.0
    
    # Calculate resolution quality score
    clipped_resolutions = [np.clip(res, 0, resolution_requirement) for res in all_resolutions]
    clipped_sum = sum(clipped_resolutions)
    max_possible_sum = resolution_requirement * (total_peaks - 1)
    
    if max_possible_sum == 0:
        resolution_quality_score = 0.0
    else:
        resolution_quality_score = clipped_sum / max_possible_sum
    
    # Calculate baseline separation ratio bonus
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
    
    # Calculate enhanced CRF
    enhanced_crf = resolution_quality_score * separation_ratio_bonus
    return enhanced_crf


def calculate_ecrf_comprehensive(signal, peak_regions, analysis_time, current_iteration, total_iterations, 
                               reference_peak_count=1, resolution_requirement=1.5, 
                               minimum_analysis_time=80, maximum_analysis_time=300):
    """
    Calculate comprehensive ECRF metric considering separation performance and analysis time
    
    Based on resolution, separation ratio, and analysis time, calculate comprehensive score metric
    for evaluating overall performance of GC temperature programs.
    
    Args:
    - signal: Chromatographic signal data
    - peak_regions: List of peak regions
    - analysis_time: Analysis time (seconds)
    - current_iteration: Current iteration count
    - total_iterations: Total iteration count
    - reference_peak_count: Number of peaks detected by initial method
    - resolution_requirement: Required resolution threshold
    - minimum_analysis_time: Minimum allowed analysis time
    - maximum_analysis_time: Maximum allowed analysis time
    
    Returns:
    - ecrf: Comprehensive ECRF score
    """
    # Calculate resolution-related data
    all_resolutions_data = calculate_all_peak_resolutions(signal, peak_regions)
    all_resolutions = [res['resolution'] for res in all_resolutions_data]
    observed_peak_count = sum(len(peak_indices) for _, _, peak_indices in peak_regions)  # Total number of detected chromatographic peaks
    peak_pair_count = len(all_resolutions)  # Number of adjacent peak pairs
    
    if observed_peak_count < 2 or peak_pair_count == 0:
        return 0.0
    
    # Calculate separation factor f_sep = S_res * B_sep
    # S_res: Resolution score
    clipped_resolutions = [min(res, resolution_requirement) for res in all_resolutions]
    resolution_score = sum(clipped_resolutions) / (resolution_requirement * (observed_peak_count - 1))
    
    # B_sep: Baseline separation ratio score
    baseline_separated_pairs = sum(1 for res in all_resolutions if res >= resolution_requirement)  # Number of baseline-separated peak pairs
    total_pairs = peak_pair_count  # Total number of peak pairs
    if total_pairs > 0 and reference_peak_count > 0:
        baseline_separation_score = (np.log(1 + baseline_separated_pairs / total_pairs) / np.log(2)) * (total_pairs / reference_peak_count)
    else:
        baseline_separation_score = 0.0
    
    separation_factor = resolution_score * baseline_separation_score
    
    # Calculate time factor f_time
    time_factor = 1 - (analysis_time - minimum_analysis_time) / (maximum_analysis_time - minimum_analysis_time)
    # Limit time factor to reasonable range [0, 1]
    time_factor = np.clip(time_factor, 0.0, 1.0)
    
    # Calculate weights (adjusted with iteration count)
    separation_weight = 0.7 + 0.3 * (1 - current_iteration / total_iterations)  # More emphasis on separation in early iterations
    time_weight = 1 - separation_weight
    
    # Calculate comprehensive ECRF
    ecrf = separation_weight * separation_factor + time_weight * time_factor
    
    return ecrf