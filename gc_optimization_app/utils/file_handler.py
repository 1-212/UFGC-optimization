"""GC optimization system file handling module"""


import os
import json
import pickle
from datetime import datetime
import pandas as pd
from config import checkpoint_directory, results_directory, data_directory

class FileHandler:
    """GC optimization system file handler
    
    Responsible for file operations for system data, including checkpoint management, result export, and data backup.
    """
    
    @staticmethod
    def save_checkpoint(state, experiment_data):
        """Save checkpoint
        
        Save optimization state and experiment data to checkpoint files for later recovery.
        
        Args:
            state: Optimization state dictionary
            experiment_data: Experiment data list
        """
        os.makedirs(checkpoint_directory, exist_ok=True)
        
        state_file = os.path.join(checkpoint_directory, "state.json")
        data_file = os.path.join(checkpoint_directory, "experiments.pkl")
        
        with open(state_file, 'w', encoding='utf-8') as file:
            json.dump(state, file, indent=2, ensure_ascii=False)
        
        with open(data_file, 'wb') as file:
            pickle.dump(experiment_data, file)
        
        print(f"OK Checkpoint saved")
    
    @staticmethod
    def load_checkpoint():
        """Load checkpoint
        
        Load optimization state and experiment data from checkpoint files.
        
        Returns:
            tuple: (State dictionary, Experiment data list), returns (None, None) if loading fails
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
            print(f"❌ Failed to load checkpoint: {error}")
            return None, None
    
    @staticmethod
    def export_results(experiment_data, parameter_names, parameter_units):
        """Export results to CSV
        
        Export experiment data to CSV file for easy analysis and review.
        
        Args:
            experiment_data: Experiment data list
            parameter_names: Parameter names list
            parameter_units: Parameter units dictionary
            
        Returns:
            str: Export file path
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
        
        export_dataframe['type'] = export_dataframe['is_initial'].apply(lambda x: 'initial' if x else 'optimized')
        
        filename = os.path.join(
            results_directory, 
            f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        export_dataframe.to_csv(filename, index=False, encoding='utf-8-sig')
        
        return filename
    
    @staticmethod
    def backup_data(experiment_data):
        """Backup data
        
        Backup experiment data to file to prevent data loss.
        
        Args:
            experiment_data: Experiment data list
            
        Returns:
            str: Backup file path
        """
        os.makedirs(results_directory, exist_ok=True)
        
        backup_file = os.path.join(
            results_directory, 
            f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        
        with open(backup_file, 'wb') as file:
            pickle.dump(experiment_data, file)
        
        return backup_file