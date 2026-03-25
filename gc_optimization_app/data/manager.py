"""数据管理模块"""

class ExperimentDataManager:
    """实验数据管理器"""
    
    def __init__(self, max_in_memory=100):
        self.data = []
        self.max_in_memory = max_in_memory
    
    def append(self, experiment):
        """添加实验数据"""
        self.data.append(experiment)
    
    def get_all(self):
        """获取所有数据"""
        return self.data.copy()
    
    def __len__(self):
        """返回总数据量"""
        return len(self.data)
    
    def __getitem__(self, index):
        """按索引访问"""
        return self.data[index]
    
    def __iter__(self):
        """迭代器"""
        return iter(self.data)
    
    def clear(self):
        """清空数据"""
        self.data = []
    
    def get_latest(self, n=5):
        """获取最近n条数据"""
        return self.data[-n:] if len(self.data) > 0 else []
    
    def get_by_round(self, round_num):
        """按轮次获取"""
        for exp in self.data:
            if exp['round'] == round_num:
                return exp
        return None
    
    def get_best(self):
        """获取最佳实验"""
        if not self.data:
            return None
        return max(self.data, key=lambda x: x['score'])
