"""日志模块"""

import logging
import os
from config import CHECKPOINT_DIR

class Logger:
    """日志记录器"""
    
    def __init__(self, name="gc_optimization"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        handler = logging.FileHandler(
            os.path.join(CHECKPOINT_DIR, "app.log"),
            encoding='utf-8'
        )
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def debug(self, message):
        self.logger.debug(message)

# 全局日志实例
logger = Logger()
