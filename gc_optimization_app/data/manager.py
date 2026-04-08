"""GC实验数据管理模块

本模块实现了实验数据的管理功能，包括数据存储、检索和分析。
用于跟踪和管理GC升温参数优化过程中的实验数据，
提供数据访问和查询功能。

作者: 研究团队
日期: 2026年
"""

class ExperimentDataManager:
    """GC实验数据管理器
    
    负责管理和组织GC升温参数优化过程中的实验数据，
    提供数据存储、检索和分析功能。
    """
    
    def __init__(self, max_in_memory=100):
        """初始化数据管理器
        
        Args:
            max_in_memory: 内存中存储的最大实验数据量
        """
        self.experiment_data = []
        self.max_in_memory = max_in_memory
    
    def add_experiment(self, experiment):
        """添加实验数据
        
        Args:
            experiment: 实验数据字典
        """
        self.experiment_data.append(experiment)
    
    def get_all_experiments(self):
        """获取所有实验数据
        
        Returns:
            list: 实验数据列表的副本
        """
        return self.experiment_data.copy()
    
    def get_all(self):
        """获取所有实验数据（兼容旧接口）
        
        Returns:
            list: 实验数据列表的副本
        """
        return self.get_all_experiments()
    
    def __len__(self):
        """返回实验数据总量
        
        Returns:
            int: 实验数据数量
        """
        return len(self.experiment_data)
    
    def __getitem__(self, index):
        """按索引访问实验数据
        
        Args:
            index: 实验数据索引
            
        Returns:
            dict: 实验数据字典
        """
        return self.experiment_data[index]
    
    def __iter__(self):
        """实验数据迭代器
        
        Returns:
            iterator: 实验数据迭代器
        """
        return iter(self.experiment_data)
    
    def clear_data(self):
        """清空所有实验数据"""
        self.experiment_data = []
    
    def get_latest_experiments(self, count=5):
        """获取最近的实验数据
        
        Args:
            count: 要获取的实验数据数量
            
        Returns:
            list: 最近的实验数据列表
        """
        return self.experiment_data[-count:] if len(self.experiment_data) > 0 else []
    
    def get_experiment_by_round(self, round_number):
        """按轮次获取实验数据
        
        Args:
            round_number: 实验轮次
            
        Returns:
            dict: 对应轮次的实验数据，若不存在则返回None
        """
        for experiment in self.experiment_data:
            if experiment['round'] == round_number:
                return experiment
        return None
    
    def get_best_experiment(self):
        """获取最佳实验数据
        
        Returns:
            dict: 评分最高的实验数据，若没有数据则返回None
        """
        if not self.experiment_data:
            return None
        return max(self.experiment_data, key=lambda x: x['score'])
