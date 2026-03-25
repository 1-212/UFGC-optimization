"""应用入口"""

import tkinter as tk
from ui.main_window import main
from utils.logger import logger

if __name__ == "__main__":
    try:
        logger.info("="*50)
        logger.info("应用启动")
        logger.info("="*50)
        main()
    except Exception as e:
        logger.error(f"应用崩溃: {e}")
        raise
