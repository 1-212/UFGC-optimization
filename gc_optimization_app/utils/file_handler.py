"""GC优化系统文件处理模块

本模块实现了文件处理功能，包括检查点保存与加载、结果导出和数据备份等功能。
负责系统数据的持久化存储和管理，确保优化过程的可恢复性和结果的可追溯性。

作者: 研究团队
日期: 2026年
"""

import os
import json
import pickle
from datetime import datetime
import pandas as pd
from config import checkpoint_directory, results_directory, data_directory

class FileHandler:
    """GC优化系统文件处理器
    
    负责系统数据的文件操作，包括检查点管理、结果导出和数据备份等功能。
    """
    
    @staticmethod
    def save_checkpoint(state, experiment_data):
        """保存检查点
        
        将优化状态和实验数据保存到检查点文件，以便后续恢复。
        
        Args:
            state: 优化状态字典
            experiment_data: 实验数据列表
        """
        os.makedirs(checkpoint_directory, exist_ok=True)
        
        state_file = os.path.join(checkpoint_directory, "state.json")
        data_file = os.path.join(checkpoint_directory, "experiments.pkl")
        
        with open(state_file, 'w', encoding='utf-8') as file:
            json.dump(state, file, indent=2, ensure_ascii=False)
        
        with open(data_file, 'wb') as file:
            pickle.dump(experiment_data, file)
        
        print(f"OK 检查点已保存")
    
    @staticmethod
    def load_checkpoint():
        """加载检查点
        
        从检查点文件加载优化状态和实验数据。
        
        Returns:
            tuple: (状态字典, 实验数据列表)，如果加载失败返回(None, None)
        """
        state_file = os.path.join(checkpoint_directory, "state.json")
        data_file = os.path.join(checkpoint_directory, "experiments.pkl")
        
        if not (os.path.exists(state_file) and os.path.exists(data_file)):
            return None, None
        
        try:
            with open(state_file, 'r', encoding='utf-8') as file:
                state = json.load(file)
            
            with open(data_file, 'rb') as file:
                experiment_data = pickle.load(file)
            
            return state, experiment_data
        except Exception as error:
            print(f"❌ 加载检查点失败: {error}")
            return None, None
    
    @staticmethod
    def export_results(experiment_data, parameter_names, parameter_units):
        """导出结果为CSV
        
        将实验数据导出为CSV文件，便于后续分析和查看。
        
        Args:
            experiment_data: 实验数据列表
            parameter_names: 参数名称列表
            parameter_units: 参数单位字典
            
        Returns:
            str: 导出文件的路径
        """
        os.makedirs(results_directory, exist_ok=True)
        
        dataframe = pd.DataFrame(experiment_data)
        
        params_dataframe = pd.DataFrame(
            dataframe['params'].tolist(),
            columns=parameter_names,
            index=dataframe.index
        )
        
        export_dataframe = pd.concat([
            dataframe[['round', 'is_initial', 'score', 'analysis_time', 'timestamp']],
            params_dataframe
        ], axis=1)
        
        export_dataframe['类型'] = export_dataframe['is_initial'].apply(lambda x: '初始点' if x else '优化点')
        
        filename = os.path.join(
            results_directory, 
            f"优化结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        export_dataframe.to_csv(filename, index=False, encoding='utf-8-sig')
        
        return filename
    
    @staticmethod
    def backup_data(experiment_data):
        """备份数据
        
        将实验数据备份到文件，防止数据丢失。
        
        Args:
            experiment_data: 实验数据列表
            
        Returns:
            str: 备份文件的路径
        """
        os.makedirs(results_directory, exist_ok=True)
        
        backup_file = os.path.join(
            results_directory, 
            f"备份_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        
        with open(backup_file, 'wb') as file:
            pickle.dump(experiment_data, file)
        
        return backup_file