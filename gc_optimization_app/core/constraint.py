"""Constraint handling module for GC temperature program optimization system"""


import numpy as np
from config import (
    parameter_dimensions as dimensions,
    parameter_names,
    min_analysis_time,
    max_analysis_time,
    initial_points_library
)

class ConstraintHandler:
    """GC temperature program constraint handler
    
    Responsible for handling constraint conditions for GC temperature program parameters,
    ensuring parameters meet physical and operational requirements.
    """
    
    def __init__(self):
        """Initialize constraint handler"""
        self.max_analysis_time = max_analysis_time
        self.min_analysis_time = min_analysis_time
    
    def calculate_analysis_time(self, parameters):
        """Calculate analysis time
        
        Calculate total analysis time based on temperature program parameters,
        including initial hold, ramp phases, and subsequent hold times.
        
        Args:
            parameters: Temperature program parameter list
            
        Returns:
            analysis_time: Total analysis time (seconds)
        """
        # Ensure parameters are scalar values, handle numpy array cases
        parameter_values = []
        for param in parameters:
            # Check if param is a numpy array or other sequence type
            if hasattr(param, '__len__') and hasattr(param, '__getitem__'):
                # If it's a length-1 array, extract element; otherwise take first element
                if hasattr(param, 'item'):  # numpy scalars have .item() method
                    scalar_value = param.item()
                elif len(param) == 1:
                    scalar_value = param[0]
                else:
                    scalar_value = param[0]  # Take first element
            else:
                scalar_value = float(param)
            parameter_values.append(scalar_value)
        
        initial_temperature, initial_hold_time, ramp_rate_1, target_temperature_1, \
        hold_time_1, ramp_rate_2, target_temperature_2, hold_time_2 = parameter_values
        
        # Check if ramp rates are positive
        if ramp_rate_1 <= 0 or ramp_rate_2 <= 0:
            return float('inf')
        
        # Calculate time for each phase
        initial_hold = initial_hold_time
        ramp1_time = (target_temperature_1 - initial_temperature) / ramp_rate_1
        hold1_time = hold_time_1
        ramp2_time = (target_temperature_2 - target_temperature_1) / ramp_rate_2
        hold2_time = hold_time_2
        
        # Calculate total analysis time
        total_time = (initial_hold + ramp1_time + hold1_time + 
                      ramp2_time + hold2_time)
        
        return total_time
    
    def is_feasible(self, parameters):
        """Check parameter feasibility
        
        Verify whether parameters meet temperature and time constraints.
        
        Args:
            parameters: Temperature program parameter list
            
        Returns:
            feasible: Whether feasible
            message: Feasibility check result message
        """
        # Ensure parameters are scalar values, handle numpy array cases
        parameter_values = []
        for param in parameters:
            # Check if param is a numpy array or other sequence type
            if hasattr(param, '__len__') and hasattr(param, '__getitem__'):
                # If it's a length-1 array, extract element; otherwise take first element
                if hasattr(param, 'item'):  # numpy scalars have .item() method
                    scalar_value = param.item()
                elif len(param) == 1:
                    scalar_value = param[0]
                else:
                    scalar_value = param[0]  # Take first element
            else:
                scalar_value = float(param)
            parameter_values.append(scalar_value)
        
        initial_temperature, initial_hold_time, ramp_rate_1, target_temperature_1, \
        hold_time_1, ramp_rate_2, target_temperature_2, hold_time_2 = parameter_values
        
        # Check temperature constraints
        if target_temperature_1 <= initial_temperature:
            return False, "Target temperature T1 must be higher than initial temperature T_init"
        if target_temperature_2 <= target_temperature_1:
            return False, "Target temperature T2 must be higher than T1"
        
        # Check ramp rate constraints
        if ramp_rate_1 <= 0 or ramp_rate_2 <= 0:
            return False, "Ramp rates must be positive"
        
        # Check analysis time constraints
        analysis_time = self.calculate_analysis_time(parameters)
        if analysis_time < min_analysis_time or analysis_time > max_analysis_time:
            return False, f"Analysis time {analysis_time:.1f}s is outside range [{min_analysis_time}, {max_analysis_time}]"
        
        return True, "Parameters are feasible"
    
    def project_to_feasible_region(self, parameters, max_iterations=20):
        """Project parameters to feasible region
        
        Through iterative adjustment, adjust infeasible parameters to feasible parameters
        that satisfy all constraints.
        
        Args:
            parameters: Original parameter list
            max_iterations: Maximum number of iterations
            
        Returns:
            feasible_parameters: Parameters projected to feasible region
        """
        # 确保参数是标量值，处理numpy数组情况
        parameter_values = []
        for param in parameters:
            # 检查是否为numpy数组或其他序列类型
            if hasattr(param, '__len__') and hasattr(param, '__getitem__'):
                # 如果是长度为1的数组，提取元素；否则取第一个元素
                if hasattr(param, 'item'):  # numpy标量有.item()方法
                    scalar_value = param.item()
                elif len(param) == 1:
                    scalar_value = float(param[0])
                else:
                    scalar_value = float(param[0])  # 取第一个元素
            else:
                scalar_value = float(param)
            parameter_values.append(scalar_value)
        
        initial_temperature, initial_hold_time, ramp_rate_1, target_temperature_1, \
        hold_time_1, ramp_rate_2, target_temperature_2, hold_time_2 = parameter_values
        
        # Correct temperature constraints
        initial_temperature = np.clip(initial_temperature, dimensions[0].low, dimensions[0].high)
        target_temperature_1 = np.clip(target_temperature_1, max(initial_temperature + 5, dimensions[3].low), dimensions[3].high)
        target_temperature_2 = np.clip(target_temperature_2, max(target_temperature_1 + 5, dimensions[6].low), dimensions[6].high)
        
        # Correct ramp rates
        ramp_rate_1 = np.clip(ramp_rate_1, dimensions[2].low, dimensions[2].high)
        ramp_rate_2 = np.clip(ramp_rate_2, dimensions[5].low, dimensions[5].high)
        
        # Iteratively adjust to meet time constraints
        for iteration in range(max_iterations):
            current_time = self.calculate_analysis_time(
                [initial_temperature, initial_hold_time, ramp_rate_1, target_temperature_1, 
                 hold_time_1, ramp_rate_2, target_temperature_2, hold_time_2]
            )
            
            if min_analysis_time <= current_time <= max_analysis_time:
                break
            
            if current_time > max_analysis_time:
                # Increase ramp rates to reduce analysis time
                ramp_rate_1 = min(ramp_rate_1 * 1.10, dimensions[2].high)
                ramp_rate_2 = min(ramp_rate_2 * 1.10, dimensions[5].high)
                
                # If ramp rates are near upper limit, reduce hold times
                if ramp_rate_1 >= dimensions[2].high * 0.95:
                    initial_hold_time = max(initial_hold_time * 0.95, dimensions[1].low)
                    hold_time_1 = max(hold_time_1 * 0.95, dimensions[4].low)
                    hold_time_2 = max(hold_time_2 * 0.95, dimensions[7].low)
            else:
                # Decrease ramp rates to increase analysis time
                ramp_rate_1 = max(ramp_rate_1 * 0.90, dimensions[2].low)
                ramp_rate_2 = max(ramp_rate_2 * 0.90, dimensions[5].low)
                
                # If ramp rates are near lower limit, increase hold times
                if ramp_rate_1 <= dimensions[2].low * 1.05:
                    initial_hold_time = min(initial_hold_time * 1.05, dimensions[1].high)
                    hold_time_1 = min(hold_time_1 * 1.05, dimensions[4].high)
                    hold_time_2 = min(hold_time_2 * 1.05, dimensions[7].high)
        
        # Ensure all parameters are within valid ranges
        projected_parameters = [
            np.clip(initial_temperature, dimensions[0].low, dimensions[0].high),
            np.clip(initial_hold_time, dimensions[1].low, dimensions[1].high),
            np.clip(ramp_rate_1, dimensions[2].low, dimensions[2].high),
            np.clip(target_temperature_1, dimensions[3].low, dimensions[3].high),
            np.clip(hold_time_1, dimensions[4].low, dimensions[4].high),
            np.clip(ramp_rate_2, dimensions[5].low, dimensions[5].high),
            np.clip(target_temperature_2, dimensions[6].low, dimensions[6].high),
            np.clip(hold_time_2, dimensions[7].low, dimensions[7].high),
        ]
        
        # Ensure parameter values meet step size requirements
        from config import parameter_step_sizes
        adjusted_parameters = []
        for i, param_value in enumerate(projected_parameters):
            param_name = parameter_names[i]
            if param_name in parameter_step_sizes:
                step_size = parameter_step_sizes[param_name]
                # Round parameter value to nearest multiple of step size
                adjusted_value = round(param_value / step_size) * step_size
                # Ensure within range again
                dim_low = dimensions[i].low
                dim_high = dimensions[i].high
                adjusted_value = max(dim_low, min(dim_high, adjusted_value))
                adjusted_parameters.append(adjusted_value)
            else:
                adjusted_parameters.append(param_value)
        
        return adjusted_parameters
    
    def generate_random_feasible_point(self):
        """Generate random feasible point
        
        Generate random parameter point that satisfies all constraints.
        
        Returns:
            feasible_parameters: Feasible parameter point
        """
        for _ in range(100):
            # Generate random parameters
            raw_parameters = [np.random.uniform(dim.low, dim.high) for dim in dimensions]
            # Ensure parameters meet step size requirements
            from config import parameter_step_sizes
            parameters = []
            for i, param_value in enumerate(raw_parameters):
                param_name = parameter_names[i]
                if param_name in parameter_step_sizes:
                    step_size = parameter_step_sizes[param_name]
                    # Round parameter value to nearest multiple of step size
                    adjusted_value = round(param_value / step_size) * step_size
                    # Ensure within range
                    dim_low = dimensions[i].low
                    dim_high = dimensions[i].high
                    adjusted_value = max(dim_low, min(dim_high, adjusted_value))
                    parameters.append(adjusted_value)
                else:
                    parameters.append(param_value)
            
            # Check feasibility
            is_feasible, _ = self.is_feasible(parameters)
            if is_feasible:
                return parameters
        
        # If random generation fails, use heuristic method
        return self.generate_heuristic_feasible_point()
    
    def generate_heuristic_feasible_point(self):
        """Generate feasible point using heuristic method
        
        Based on experience and chromatographic theory, generate reasonable feasible parameter point.
        
        Returns:
            feasible_parameters: Feasible parameter point
        """
        # Generate reasonable temperature values
        initial_temperature = np.random.uniform(45, 55)  # Common initial temperature range
        target_temperature_1 = np.random.uniform(85, 110)  # Light component separation temperature
        target_temperature_2 = np.random.uniform(245, 258)  # Heavy component separation temperature
        
        # Target analysis time (midpoint)
        target_time = (min_analysis_time + max_analysis_time) / 2
        
        # Generate reasonable hold times
        initial_hold_time = np.random.uniform(2, 3)  # Initial hold time
        hold_time_1 = np.random.uniform(0, 2)  # First hold time
        hold_time_2 = np.random.uniform(21, 25)  # Second hold time
        
        # Calculate total hold time and available ramp time
        total_hold_time = initial_hold_time + hold_time_1 + hold_time_2
        available_ramp_time = target_time - total_hold_time
        
        # Allocate ramp time proportionally to temperature ranges
        ramp1_time = (target_temperature_1 - initial_temperature) / (target_temperature_2 - initial_temperature) * available_ramp_time
        ramp2_time = (target_temperature_2 - target_temperature_1) / (target_temperature_2 - initial_temperature) * available_ramp_time
        
        # Calculate ramp rates
        ramp_rate_1 = (target_temperature_1 - initial_temperature) / max(ramp1_time, 0.1)
        ramp_rate_2 = (target_temperature_2 - target_temperature_1) / max(ramp2_time, 0.1)
        
        # Ensure ramp rates are within valid ranges
        ramp_rate_1 = np.clip(ramp_rate_1, dimensions[2].low, dimensions[2].high)
        ramp_rate_2 = np.clip(ramp_rate_2, dimensions[5].low, dimensions[5].high)
        
        # Build parameter list
        parameters = [initial_temperature, initial_hold_time, ramp_rate_1, target_temperature_1, 
                     hold_time_1, ramp_rate_2, target_temperature_2, hold_time_2]
        
        # Project to feasible region
        return self.project_to_feasible_region(parameters)


# Global constraint handler instance
constraint_handler = ConstraintHandler()


def get_initial_parameter_points():
    """Get initial parameter points
    
    Get initial parameter points from the initial point library after feasibility processing.
    
    Returns:
        initial_points: List of processed initial parameter points
    """
    return [
        constraint_handler.project_to_feasible_region(point) 
        for point in initial_points_library
    ]