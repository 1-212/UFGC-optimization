"""GC experiment data management module"""


class ExperimentDataManager:
    """GC experiment data manager
    
    Responsible for managing and organizing experimental data during GC temperature program optimization,
    providing data storage, retrieval, and analysis functionality.
    """
    
    def __init__(self, max_in_memory=100):
        """Initialize data manager
        
        Args:
            max_in_memory: Maximum number of experiment data to store in memory
        """
        self.experiment_data = []
        self.max_in_memory = max_in_memory
    
    def add_experiment(self, experiment):
        """Add experiment data
        
        Args:
            experiment: Experiment data dictionary
        """
        self.experiment_data.append(experiment)
    
    def get_all_experiments(self):
        """Get all experiment data
        
        Returns:
            list: Copy of experiment data list
        """
        return self.experiment_data.copy()
    
    def get_all(self):
        """Get all experiment data (compatible with old interface)
        
        Returns:
            list: Copy of experiment data list
        """
        return self.get_all_experiments()
    
    def __len__(self):
        """Return total experiment data count
        
        Returns:
            int: Number of experiment data
        """
        return len(self.experiment_data)
    
    def __getitem__(self, index):
        """Access experiment data by index
        
        Args:
            index: Experiment data index
            
        Returns:
            dict: Experiment data dictionary
        """
        return self.experiment_data[index]
    
    def __iter__(self):
        """Experiment data iterator
        
        Returns:
            iterator: Experiment data iterator
        """
        return iter(self.experiment_data)
    
    def clear_data(self):
        """Clear all experiment data"""
        self.experiment_data = []
    
    def get_latest_experiments(self, count=5):
        """Get latest experiment data
        
        Args:
            count: Number of experiment data to retrieve
            
        Returns:
            list: List of latest experiment data
        """
        return self.experiment_data[-count:] if len(self.experiment_data) > 0 else []
    
    def get_experiment_by_round(self, round_number):
        """Get experiment data by round
        
        Args:
            round_number: Experiment round number
            
        Returns:
            dict: Experiment data for the specified round, None if not found
        """
        for experiment in self.experiment_data:
            if experiment['round'] == round_number:
                return experiment
        return None
    
    def get_best_experiment(self):
        """Get best experiment data
        
        Returns:
            dict: Experiment data with highest score, None if no data
        """
        if not self.experiment_data:
            return None
        return max(self.experiment_data, key=lambda x: x['score'])
