"""GC升温参数约束处理模块

本模块实现了GC升温参数的约束处理功能，包括可行性检查、
分析时间计算、参数投影到可行域以及可行点生成等功能。
基于色谱理论和实验要求，确保生成的参数满足物理和操作约束。

核心功能：
1. 分析时间计算：根据升温程序参数计算总分析时间
2. 可行性检查：验证参数是否满足温度和时间约束
3. 可行域投影：将不可行参数调整为可行参数
4. 可行点生成：生成满足约束的随机或启发式参数点

作者: 研究团队
日期: 2026年
"""

import numpy as np
from config import (
    parameter_dimensions as dimensions,
    parameter_names,
    min_analysis_time,
    max_analysis_time,
    initial_points_library
)

class ConstraintHandler:
    """GC升温参数约束处理器
    
    负责处理GC升温参数的约束条件，确保参数满足物理和操作要求。
    """
    
    def __init__(self):
        """初始化约束处理器"""
        self.max_analysis_time = max_analysis_time
        self.min_analysis_time = min_analysis_time
    
    def calculate_analysis_time(self, parameters):
        """计算分析时间
        
        根据升温程序参数计算总分析时间，包括初始保持、升温阶段和后续保持时间。
        
        Args:
            parameters: 升温程序参数列表
            
        Returns:
            analysis_time: 总分析时间（秒）
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
                    scalar_value = param[0]
                else:
                    scalar_value = param[0]  # 取第一个元素
            else:
                scalar_value = float(param)
            parameter_values.append(scalar_value)
        
        initial_temperature, initial_hold_time, ramp_rate_1, target_temperature_1, \
        hold_time_1, ramp_rate_2, target_temperature_2, hold_time_2 = parameter_values
        
        # 检查升温速率是否为正
        if ramp_rate_1 <= 0 or ramp_rate_2 <= 0:
            return float('inf')
        
        # 计算各阶段时间
        initial_hold = initial_hold_time
        ramp1_time = (target_temperature_1 - initial_temperature) / ramp_rate_1
        hold1_time = hold_time_1
        ramp2_time = (target_temperature_2 - target_temperature_1) / ramp_rate_2
        hold2_time = hold_time_2
        
        # 计算总分析时间
        total_time = (initial_hold + ramp1_time + hold1_time + 
                      ramp2_time + hold2_time)
        
        return total_time
    
    def is_feasible(self, parameters):
        """检查参数可行性
        
        验证参数是否满足温度和时间约束条件。
        
        Args:
            parameters: 升温程序参数列表
            
        Returns:
            feasible: 是否可行
            message: 可行性检查结果消息
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
                    scalar_value = param[0]
                else:
                    scalar_value = param[0]  # 取第一个元素
            else:
                scalar_value = float(param)
            parameter_values.append(scalar_value)
        
        initial_temperature, initial_hold_time, ramp_rate_1, target_temperature_1, \
        hold_time_1, ramp_rate_2, target_temperature_2, hold_time_2 = parameter_values
        
        # 检查温度约束
        if target_temperature_1 <= initial_temperature:
            return False, "目标温度T1必须高于初始温度T_init"
        if target_temperature_2 <= target_temperature_1:
            return False, "目标温度T2必须高于T1"
        
        # 检查升温速率约束
        if ramp_rate_1 <= 0 or ramp_rate_2 <= 0:
            return False, "升温速率必须为正数"
        
        # 检查分析时间约束
        analysis_time = self.calculate_analysis_time(parameters)
        if analysis_time < min_analysis_time or analysis_time > max_analysis_time:
            return False, f"分析时间{analysis_time:.1f}s超出范围[{min_analysis_time}, {max_analysis_time}]"
        
        return True, "参数可行"
    
    def project_to_feasible_region(self, parameters, max_iterations=20):
        """将参数投影到可行域
        
        通过迭代调整，将不可行参数调整为满足所有约束的可行参数。
        
        Args:
            parameters: 原始参数列表
            max_iterations: 最大迭代次数
            
        Returns:
            feasible_parameters: 投影到可行域的参数
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
        
        # 修正温度约束
        initial_temperature = np.clip(initial_temperature, dimensions[0].low, dimensions[0].high)
        target_temperature_1 = np.clip(target_temperature_1, max(initial_temperature + 5, dimensions[3].low), dimensions[3].high)
        target_temperature_2 = np.clip(target_temperature_2, max(target_temperature_1 + 5, dimensions[6].low), dimensions[6].high)
        
        # 修正升温速率
        ramp_rate_1 = np.clip(ramp_rate_1, dimensions[2].low, dimensions[2].high)
        ramp_rate_2 = np.clip(ramp_rate_2, dimensions[5].low, dimensions[5].high)
        
        # 迭代调整以满足时间约束
        for iteration in range(max_iterations):
            current_time = self.calculate_analysis_time(
                [initial_temperature, initial_hold_time, ramp_rate_1, target_temperature_1, 
                 hold_time_1, ramp_rate_2, target_temperature_2, hold_time_2]
            )
            
            if min_analysis_time <= current_time <= max_analysis_time:
                break
            
            if current_time > max_analysis_time:
                # 增加升温速率以减少分析时间
                ramp_rate_1 = min(ramp_rate_1 * 1.10, dimensions[2].high)
                ramp_rate_2 = min(ramp_rate_2 * 1.10, dimensions[5].high)
                
                # 如果升温速率已接近上限，减少保持时间
                if ramp_rate_1 >= dimensions[2].high * 0.95:
                    initial_hold_time = max(initial_hold_time * 0.95, dimensions[1].low)
                    hold_time_1 = max(hold_time_1 * 0.95, dimensions[4].low)
                    hold_time_2 = max(hold_time_2 * 0.95, dimensions[7].low)
            else:
                # 减少升温速率以增加分析时间
                ramp_rate_1 = max(ramp_rate_1 * 0.90, dimensions[2].low)
                ramp_rate_2 = max(ramp_rate_2 * 0.90, dimensions[5].low)
                
                # 如果升温速率已接近下限，增加保持时间
                if ramp_rate_1 <= dimensions[2].low * 1.05:
                    initial_hold_time = min(initial_hold_time * 1.05, dimensions[1].high)
                    hold_time_1 = min(hold_time_1 * 1.05, dimensions[4].high)
                    hold_time_2 = min(hold_time_2 * 1.05, dimensions[7].high)
        
        # 确保所有参数在有效范围内
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
        
        # 确保参数值符合步长要求
        from config import parameter_step_sizes
        adjusted_parameters = []
        for i, param_value in enumerate(projected_parameters):
            param_name = parameter_names[i]
            if param_name in parameter_step_sizes:
                step_size = parameter_step_sizes[param_name]
                # 将参数值四舍五入到最接近的步长倍数
                adjusted_value = round(param_value / step_size) * step_size
                # 再次确保在范围内
                dim_low = dimensions[i].low
                dim_high = dimensions[i].high
                adjusted_value = max(dim_low, min(dim_high, adjusted_value))
                adjusted_parameters.append(adjusted_value)
            else:
                adjusted_parameters.append(param_value)
        
        return adjusted_parameters
    
    def generate_random_feasible_point(self):
        """生成随机可行点
        
        生成满足所有约束的随机参数点。
        
        Returns:
            feasible_parameters: 可行的参数点
        """
        for _ in range(100):
            # 生成随机参数
            raw_parameters = [np.random.uniform(dim.low, dim.high) for dim in dimensions]
            # 确保参数符合步长要求
            from config import parameter_step_sizes
            parameters = []
            for i, param_value in enumerate(raw_parameters):
                param_name = parameter_names[i]
                if param_name in parameter_step_sizes:
                    step_size = parameter_step_sizes[param_name]
                    # 将参数值四舍五入到最接近的步长倍数
                    adjusted_value = round(param_value / step_size) * step_size
                    # 确保在范围内
                    dim_low = dimensions[i].low
                    dim_high = dimensions[i].high
                    adjusted_value = max(dim_low, min(dim_high, adjusted_value))
                    parameters.append(adjusted_value)
                else:
                    parameters.append(param_value)
            
            # 检查可行性
            is_feasible, _ = self.is_feasible(parameters)
            if is_feasible:
                return parameters
        
        # 如果随机生成失败，使用启发式方法
        return self.generate_heuristic_feasible_point()
    
    def generate_heuristic_feasible_point(self):
        """使用启发式方法生成可行点
        
        基于经验和色谱理论，生成合理的可行参数点。
        
        Returns:
            feasible_parameters: 可行的参数点
        """
        # 生成合理的温度值
        initial_temperature = np.random.uniform(45, 55)  # 常见初始温度范围
        target_temperature_1 = np.random.uniform(85, 110)  # 轻组分分离温度
        target_temperature_2 = np.random.uniform(245, 258)  # 重组分分离温度
        
        # 目标分析时间（中间值）
        target_time = (min_analysis_time + max_analysis_time) / 2
        
        # 生成合理的保持时间
        initial_hold_time = np.random.uniform(2, 3)  # 初始保持时间
        hold_time_1 = np.random.uniform(0, 2)  # 第一段保持时间
        hold_time_2 = np.random.uniform(21, 25)  # 第二段保持时间
        
        # 计算总保持时间和可用的升温时间
        total_hold_time = initial_hold_time + hold_time_1 + hold_time_2
        available_ramp_time = target_time - total_hold_time
        
        # 按温度范围比例分配升温时间
        ramp1_time = (target_temperature_1 - initial_temperature) / (target_temperature_2 - initial_temperature) * available_ramp_time
        ramp2_time = (target_temperature_2 - target_temperature_1) / (target_temperature_2 - initial_temperature) * available_ramp_time
        
        # 计算升温速率
        ramp_rate_1 = (target_temperature_1 - initial_temperature) / max(ramp1_time, 0.1)
        ramp_rate_2 = (target_temperature_2 - target_temperature_1) / max(ramp2_time, 0.1)
        
        # 确保升温速率在有效范围内
        ramp_rate_1 = np.clip(ramp_rate_1, dimensions[2].low, dimensions[2].high)
        ramp_rate_2 = np.clip(ramp_rate_2, dimensions[5].low, dimensions[5].high)
        
        # 构建参数列表
        parameters = [initial_temperature, initial_hold_time, ramp_rate_1, target_temperature_1, 
                     hold_time_1, ramp_rate_2, target_temperature_2, hold_time_2]
        
        # 投影到可行域
        return self.project_to_feasible_region(parameters)


# 全局约束处理器实例
constraint_handler = ConstraintHandler()

def get_initial_parameter_points():
    """获取初始参数点
    
    从初始点库中获取经过可行性处理的初始参数点。
    
    Returns:
        initial_points: 处理后的初始参数点列表
    """
    return [
        constraint_handler.project_to_feasible_region(point) 
        for point in initial_points_library
    ]