"""文件处理模块"""

import os
import json
import pickle
from datetime import datetime
import pandas as pd
from config import CHECKPOINT_DIR, RESULTS_DIR, DATA_DIR

class FileHandler:
    """文件处理器"""
    
    @staticmethod
    def save_checkpoint(state, experiment_data):
        """保存检查点"""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        state_file = os.path.join(CHECKPOINT_DIR, "state.json")
        data_file = os.path.join(CHECKPOINT_DIR, "experiments.pkl")
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        with open(data_file, 'wb') as f:
            pickle.dump(experiment_data, f)
        
        print(f"✓ 检查点已保存")
    
    @staticmethod
    def load_checkpoint():
        """加载检查点"""
        state_file = os.path.join(CHECKPOINT_DIR, "state.json")
        data_file = os.path.join(CHECKPOINT_DIR, "experiments.pkl")
        
        if not (os.path.exists(state_file) and os.path.exists(data_file)):
            return None, None
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            with open(data_file, 'rb') as f:
                experiment_data = pickle.load(f)
            
            return state, experiment_data
        except Exception as e:
            print(f"❌ 加载检查点失败: {e}")
            return None, None
    
    @staticmethod
    def export_results(experiment_data, param_names, param_units):
        """导出结果为CSV"""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        df = pd.DataFrame(experiment_data)
        
        params_df = pd.DataFrame(
            df['params'].tolist(),
            columns=param_names,
            index=df.index
        )
        
        export_df = pd.concat([
            df[['round', 'is_initial', 'score', 'analysis_time', 'timestamp']],
            params_df
        ], axis=1)
        
        export_df['类型'] = export_df['is_initial'].apply(lambda x: '初始点' if x else '优化点')
        
        filename = os.path.join(
            RESULTS_DIR, 
            f"优化结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        export_df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        return filename
    
    @staticmethod
    def backup_data(experiment_data):
        """备份数据"""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        backup_file = os.path.join(
            RESULTS_DIR, 
            f"备份_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        
        with open(backup_file, 'wb') as f:
            pickle.dump(experiment_data, f)
        
        return backup_file