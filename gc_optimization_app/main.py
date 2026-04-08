"""GC升温参数优化系统主入口

本模块作为气相色谱(GC)升温参数贝叶斯优化系统的启动点，负责初始化并运行整个应用。
系统采用贝叶斯优化算法，结合信号处理技术，自动优化GC升温程序参数，
以提高色谱分离效果和分析效率。

作者: 研究团队
日期: 2026年
"""

import tkinter as tk
from ui.main_window import main  # 导入主窗口启动函数
from utils.logger import logger  # 导入日志记录器

if __name__ == "__main__":
    try:
        # 记录应用启动信息
        logger.info("=" * 50)
        logger.info("GC升温参数优化系统启动")
        logger.info("系统初始化中...")
        logger.info("=" * 50)
        
        # 启动主应用窗口
        main()
    except Exception as error:
        # 记录应用崩溃信息并重新抛出异常
        logger.error(f"应用崩溃: {error}")
        raise
