"""GC优化系统日志模块

本模块实现了系统日志记录功能，用于记录系统运行状态、错误信息和调试信息。
通过统一的日志接口，确保系统运行过程的可追溯性和问题排查能力。

作者: 研究团队
日期: 2026年
"""

import logging
import os
from config import checkpoint_directory

class Logger:
    """GC优化系统日志记录器
    
    负责系统日志的配置和记录，支持不同级别的日志输出。
    """
    
    def __init__(self, name="gc_optimization"):
        """初始化日志记录器
        
        Args:
            name: 日志记录器名称
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        os.makedirs(checkpoint_directory, exist_ok=True)
        
        file_handler = logging.FileHandler(
            os.path.join(checkpoint_directory, "app.log"),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def info(self, message):
        """记录信息级日志
        
        Args:
            message: 日志消息
        """
        self.logger.info(message)
    
    def warning(self, message):
        """记录警告级日志
        
        Args:
            message: 日志消息
        """
        self.logger.warning(message)
    
    def error(self, message):
        """记录错误级日志
        
        Args:
            message: 日志消息
        """
        self.logger.error(message)
    
    def debug(self, message):
        """记录调试级日志
        
        Args:
            message: 日志消息
        """
        self.logger.debug(message)

# 全局日志实例
logger = Logger()
