# GC Optimization Project

## 项目简介

GC Optimization是一个用于优化和分析的Python应用程序，主要用于处理和优化GC（Gas Chromatography）相关的数据。该项目提供了一套完整的工具，包括数据解析、优化算法、可视化分析等功能，旨在帮助研究人员更高效地进行GC数据的分析和处理。

## 项目结构

```
gc_optimization/
├── checkpoints/        # 实验检查点和日志
├── visualizations/     # 可视化结果

gc_optimization_app/
├── core/               # 核心功能模块
│   ├── constraint.py   # 约束条件处理
│   ├── optimizer.py    # 优化算法实现
│   └── signal_process.py # 信号处理
├── data/               # 数据处理模块
│   ├── asc_parser.py   # ASC文件解析
│   └── manager.py      # 数据管理
├── tests/              # 测试文件
├── ui/                 # 用户界面
│   ├── main_window.py  # 主窗口
│   └── panels.py       # 界面面板
├── utils/              # 工具函数
│   ├── file_handler.py # 文件处理
│   └── logger.py       # 日志记录
├── visualization/      # 可视化模块
│   └── plotter.py      # 绘图功能
├── config.py           # 配置文件
├── main.py             # 主入口
└── requirements.txt    # 依赖项
```

## 安装方法

### 1. 克隆项目

```bash
git clone [项目仓库地址]
cd GC
```

### 2. 安装依赖

```bash
pip install -r gc_optimization_app/requirements.txt
```

## 使用方法

### 命令行运行

```bash
python gc_optimization_app/main.py
```

### 功能说明

1. **数据解析**：支持ASC格式的GC数据文件解析
2. **参数优化**：提供贝叶斯优化算法，用于优化GC分析参数
3. **可视化分析**：生成收敛曲线、时间分布、参数敏感性分析等可视化结果
4. **用户界面**：提供图形化界面，方便用户操作和分析

## 核心功能

- **信号处理**：对GC信号进行预处理和分析
- **约束优化**：在满足约束条件的情况下进行参数优化
- **数据管理**：高效管理和处理GC数据
- **可视化**：生成多种类型的可视化图表，辅助分析决策

## 实验结果

项目生成的可视化结果存储在`gc_optimization/visualizations/`目录中，包括：

- 收敛曲线：展示优化过程的收敛情况
- 时间分布分析：分析优化过程中的时间消耗
- 参数敏感性分析：评估不同参数对优化结果的影响
- 高维参数空间分析：可视化高维参数空间中的优化过程

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request，共同改进项目。

## 联系方式

如有问题或建议，请联系项目维护者。
