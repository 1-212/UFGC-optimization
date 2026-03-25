"""约束处理模块"""

import numpy as np
from config import (
    PARAM_DIMENSIONS as dimensions,
    PARAM_NAMES,
    MIN_ANALYSIS_TIME,
    MAX_ANALYSIS_TIME,
    INITIAL_POINTS_LIBRARY
)

class ConstraintHandler:
    """约束处理器"""
    
    def __init__(self):
        self.max_analysis_time = MAX_ANALYSIS_TIME
        self.min_analysis_time = MIN_ANALYSIS_TIME
    
    def calculate_analysis_time(self, params):
        """计算分析时间"""
        # 确保参数是标量值，处理numpy数组情况
        values = []
        for p in params:
            # 检查是否为numpy数组或其他序列类型
            if hasattr(p, '__len__') and hasattr(p, '__getitem__'):
                # 如果是长度为1的数组，提取元素；否则取第一个元素
                if hasattr(p, 'item'):  # numpy标量有.item()方法
                    scalar_val = p.item()
                elif len(p) == 1:
                    scalar_val = p[0]
                else:
                    scalar_val = p[0]  # 取第一个元素
            else:
                scalar_val = float(p)
            values.append(scalar_val)
        
        T_init, t_hold_0, R1, T1, t_hold_1, R2, T2, t_hold_2 = values
        
        if R1 <= 0 or R2 <= 0:
            return float('inf')
        
        initial_hold = t_hold_0
        ramp1_time = (T1 - T_init) / R1
        hold1_time = t_hold_1
        ramp2_time = (T2 - T1) / R2
        hold2_time = t_hold_2
        
        total_time = (initial_hold + ramp1_time + hold1_time + 
                      ramp2_time + hold2_time)
        
        return total_time
    
    def is_feasible(self, params):
        """检查参数可行性"""
        # 确保参数是标量值，处理numpy数组情况
        values = []
        for p in params:
            # 检查是否为numpy数组或其他序列类型
            if hasattr(p, '__len__') and hasattr(p, '__getitem__'):
                # 如果是长度为1的数组，提取元素；否则取第一个元素
                if hasattr(p, 'item'):  # numpy标量有.item()方法
                    scalar_val = p.item()
                elif len(p) == 1:
                    scalar_val = p[0]
                else:
                    scalar_val = p[0]  # 取第一个元素
            else:
                scalar_val = float(p)
            values.append(scalar_val)
        
        T_init, t_hold_0, R1, T1, t_hold_1, R2, T2, t_hold_2 = values
        
        if T1 <= T_init:
            return False, "T1必须高于T_init"
        if T2 <= T1:
            return False, "T2必须高于T1"
        if R1 <= 0 or R2 <= 0:
            return False, "升温速率必须为正"
        
        analysis_time = self.calculate_analysis_time(params)
        if analysis_time < MIN_ANALYSIS_TIME or analysis_time > MAX_ANALYSIS_TIME:
            return False, f"分析时间{analysis_time:.1f}s超出范围[{MIN_ANALYSIS_TIME}, {MAX_ANALYSIS_TIME}]"
        
        return True, "可行"
    
    def project_to_feasible(self, params, max_iterations=20):
        """投影到可行域"""
        # 确保参数是标量值，处理numpy数组情况
        values = []
        for p in params:
            # 检查是否为numpy数组或其他序列类型
            if hasattr(p, '__len__') and hasattr(p, '__getitem__'):
                # 如果是长度为1的数组，提取元素；否则取第一个元素
                if hasattr(p, 'item'):  # numpy标量有.item()方法
                    scalar_val = p.item()
                elif len(p) == 1:
                    scalar_val = float(p[0])
                else:
                    scalar_val = float(p[0])  # 取第一个元素
            else:
                scalar_val = float(p)
            values.append(scalar_val)
        
        T_init, t_hold_0, R1, T1, t_hold_1, R2, T2, t_hold_2 = values
        
        # 修正温度约束
        T_init = np.clip(T_init, dimensions[0].low, dimensions[0].high)
        T1 = np.clip(T1, max(T_init + 5, dimensions[3].low), dimensions[3].high)
        T2 = np.clip(T2, max(T1 + 5, dimensions[6].low), dimensions[6].high)
        
        # 修正升温速率
        R1 = np.clip(R1, dimensions[2].low, dimensions[2].high)
        R2 = np.clip(R2, dimensions[5].low, dimensions[5].high)
        
        # 迭代调整以满足时间约束
        for iteration in range(max_iterations):
            current_time = self.calculate_analysis_time(
                [T_init, t_hold_0, R1, T1, t_hold_1, R2, T2, t_hold_2]
            )
            
            if MIN_ANALYSIS_TIME <= current_time <= MAX_ANALYSIS_TIME:
                break
            
            if current_time > MAX_ANALYSIS_TIME:
                R1 = min(R1 * 1.10, dimensions[2].high)
                R2 = min(R2 * 1.10, dimensions[5].high)
                
                if R1 >= dimensions[2].high * 0.95:
                    t_hold_0 = max(t_hold_0 * 0.95, dimensions[1].low)
                    t_hold_1 = max(t_hold_1 * 0.95, dimensions[4].low)
                    t_hold_2 = max(t_hold_2 * 0.95, dimensions[7].low)
            else:
                R1 = max(R1 * 0.90, dimensions[2].low)
                R2 = max(R2 * 0.90, dimensions[5].low)
                
                if R1 <= dimensions[2].low * 1.05:
                    t_hold_0 = min(t_hold_0 * 1.05, dimensions[1].high)
                    t_hold_1 = min(t_hold_1 * 1.05, dimensions[4].high)
                    t_hold_2 = min(t_hold_2 * 1.05, dimensions[7].high)
        
        params_projected = [
            np.clip(T_init, dimensions[0].low, dimensions[0].high),
            np.clip(t_hold_0, dimensions[1].low, dimensions[1].high),
            np.clip(R1, dimensions[2].low, dimensions[2].high),
            np.clip(T1, dimensions[3].low, dimensions[3].high),
            np.clip(t_hold_1, dimensions[4].low, dimensions[4].high),
            np.clip(R2, dimensions[5].low, dimensions[5].high),
            np.clip(T2, dimensions[6].low, dimensions[6].high),
            np.clip(t_hold_2, dimensions[7].low, dimensions[7].high),
        ]
        
        # 确保参数值符合步长要求
        from config import STEP_SIZES, PARAM_NAMES
        params_adjusted = []
        for i, param_val in enumerate(params_projected):
            param_name = PARAM_NAMES[i]
            if param_name in STEP_SIZES:
                step_size = STEP_SIZES[param_name]
                # 将参数值四舍五入到最接近的步长倍数
                adjusted_val = round(param_val / step_size) * step_size
                # 再次确保在范围内
                dim_low = dimensions[i].low
                dim_high = dimensions[i].high
                adjusted_val = max(dim_low, min(dim_high, adjusted_val))
                params_adjusted.append(adjusted_val)
            else:
                params_adjusted.append(param_val)
        
        return params_adjusted
    
    def generate_feasible_random(self):
        """生成随机可行点"""
        for _ in range(100):
            raw_params = [np.random.uniform(dim.low, dim.high) for dim in dimensions]
            # 确保参数符合步长要求
            from config import STEP_SIZES, PARAM_NAMES
            params = []
            for i, param_val in enumerate(raw_params):
                param_name = PARAM_NAMES[i]
                if param_name in STEP_SIZES:
                    step_size = STEP_SIZES[param_name]
                    # 将参数值四舍五入到最接近的步长倍数
                    adjusted_val = round(param_val / step_size) * step_size
                    # 确保在范围内
                    dim_low = dimensions[i].low
                    dim_high = dimensions[i].high
                    adjusted_val = max(dim_low, min(dim_high, adjusted_val))
                    params.append(adjusted_val)
                else:
                    params.append(param_val)
            
            is_feasible, _ = self.is_feasible(params)
            if is_feasible:
                return params
        
        return self.generate_feasible_heuristic()
    
    def generate_feasible_heuristic(self):
        """使用启发式方法生成可行点"""
        T_init = np.random.uniform(45, 55)
        T1 = np.random.uniform(85, 110)
        T2 = np.random.uniform(245, 258)
        
        target_time = (MIN_ANALYSIS_TIME + MAX_ANALYSIS_TIME) / 2
        
        t_hold_0 = np.random.uniform(2, 3)
        t_hold_1 = np.random.uniform(0, 2)
        t_hold_2 = np.random.uniform(21, 25)
        
        hold_time = t_hold_0 + t_hold_1 + t_hold_2
        ramp_time = target_time - hold_time
        
        ramp1_time = (T1 - T_init) / (T2 - T_init) * ramp_time
        ramp2_time = (T2 - T1) / (T2 - T_init) * ramp_time
        
        R1 = (T1 - T_init) / max(ramp1_time, 0.1)
        R2 = (T2 - T1) / max(ramp2_time, 0.1)
        
        R1 = np.clip(R1, dimensions[2].low, dimensions[2].high)
        R2 = np.clip(R2, dimensions[5].low, dimensions[5].high)
        
        params = [T_init, t_hold_0, R1, T1, t_hold_1, R2, T2, t_hold_2]
        
        return self.project_to_feasible(params)


# 全局实例
constraint_handler = ConstraintHandler()

def get_initial_points():
    """获取初始点"""
    return [
        constraint_handler.project_to_feasible(p) 
        for p in INITIAL_POINTS_LIBRARY
    ]