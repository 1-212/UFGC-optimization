"""Main entry point for GC temperature program optimization system"""


import tkinter as tk
from ui.main_window import main  # Import main window startup function
from utils.logger import logger  # Import logger

if __name__ == "__main__":
    try:
        # Log application startup information
        logger.info("=" * 50)
        logger.info("GC temperature program optimization system started")
        logger.info("System initializing...")
        logger.info("=" * 50)
        
        # Start main application window
        main()
    except Exception as error:
        # Log application crash information and re-raise exception
        logger.error(f"Application crashed: {error}")
        raise
