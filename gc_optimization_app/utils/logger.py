"""GC optimization system logger module"""


import logging
import os
from config import checkpoint_directory

class Logger:
    """GC optimization system logger
    
    Responsible for system log configuration and recording, supporting different levels of log output.
    """
    
    def __init__(self, name="gc_optimization"):
        """Initialize logger
        
        Args:
            name: Logger name
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
        """Record info level log
        
        Args:
            message: Log message
        """
        self.logger.info(message)
    
    def warning(self, message):
        """Record warning level log
        
        Args:
            message: Log message
        """
        self.logger.warning(message)
    
    def error(self, message):
        """Record error level log
        
        Args:
            message: Log message
        """
        self.logger.error(message)
    
    def debug(self, message):
        """Record debug level log
        
        Args:
            message: Log message
        """
        self.logger.debug(message)

# Global logger instance
logger = Logger()
