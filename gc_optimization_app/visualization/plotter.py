"""可视化模块"""

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import griddata
from config import PARAM_NAMES, PARAM_UNITS, MAX_ANALYSIS_TIME, MIN_ANALYSIS_TIME

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti']
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

class VisualizationManager:
    """可视化管理器"""
    
    def __init__(self, work_dir="gc_optimization"):
        self.work_dir = work_dir
        self.viz_dir = os.path.join(work_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        self.gp_model = None
    
    def set_gp_model(self, gp):
        """设置GP模型"""
        self.gp_model = gp
    
    def save_figure(self, fig, name, dpi=300):
        """保存图表"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        png_path = os.path.join(self.viz_dir, f"{name}_{timestamp}.png")
        pdf_path = os.path.join(self.viz_dir, f"{name}_{timestamp}.pdf")
        
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', format='png')
        fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight', format='pdf')
        
        print(f"✓ 图表已保存: {png_path}")
        return png_path, pdf_path
    
    def plot_convergence_curve(self, experiment_data):
        """绘制收敛曲线"""
        if len(experiment_data) < 1:
            return None
        
        df = pd.DataFrame(experiment_data)
        
        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
        
        ax.plot(df.index + 1, df['score'], 'o-', linewidth=2.5, 
               markersize=8, label='当前评分', color='#1f77b4', alpha=0.7)
        
        best_scores = df['score'].cummax()
        ax.plot(df.index + 1, best_scores, 's-', linewidth=3, 
               markersize=8, label='最佳评分（累积）', color='#d62728')
        
        initial_mask = df['is_initial']
        if initial_mask.sum() > 0:
            ax.scatter(df[initial_mask].index + 1, df[initial_mask]['score'], 
                      color='#2ca02c', s=150, marker='^', label='初始点', 
                      zorder=5, edgecolors='black', linewidth=1.5)
        
        optimized_mask = ~initial_mask
        if optimized_mask.sum() > 0:
            ax.scatter(df[optimized_mask].index + 1, df[optimized_mask]['score'], 
                      color='#ff7f0e', s=150, marker='o', label='优化点', 
                      zorder=5, edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
        ax.set_ylabel('CRF 评分', fontsize=12, fontweight='bold')
        ax.set_title('优化收敛曲线', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
        
        plt.tight_layout()
        return fig
    
    def plot_analysis_time_distribution(self, experiment_data):
        """绘制分析时间分布"""
        if len(experiment_data) < 1:
            return None
        
        df = pd.DataFrame(experiment_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
        
        ax1.plot(df.index + 1, df['analysis_time'], 'o-', linewidth=2.5, 
                markersize=8, color='#1f77b4', label='分析时间')
        ax1.axhline(y=MAX_ANALYSIS_TIME, color='#d62728', linestyle='--', 
                   linewidth=2.5, label=f'最大限制 ({MAX_ANALYSIS_TIME}s)')
        ax1.axhline(y=MIN_ANALYSIS_TIME, color='#2ca02c', linestyle='--', 
                   linewidth=2.5, label=f'最小限制 ({MIN_ANALYSIS_TIME}s)')
        ax1.fill_between(df.index + 1, MIN_ANALYSIS_TIME, MAX_ANALYSIS_TIME, 
                        alpha=0.1, color='gray', label='有效范围')
        
        ax1.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
        ax1.set_ylabel('分析时间 (秒)', fontsize=12, fontweight='bold')
        ax1.set_title('分析时间变化趋势', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10)
        
        initial_times = df[df['is_initial']]['analysis_time']
        optimized_times = df[~df['is_initial']]['analysis_time']
        
        bins = np.linspace(df['analysis_time'].min()-10, df['analysis_time'].max()+10, 15)
        
        ax2.hist(initial_times, bins=bins, alpha=0.6, label='初始点', 
                color='#2ca02c', edgecolor='black', linewidth=1.2)
        ax2.hist(optimized_times, bins=bins, alpha=0.6, label='优化点', 
                color='#ff7f0e', edgecolor='black', linewidth=1.2)
        
        ax2.set_xlabel('分析时间 (秒)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('频率', fontsize=12, fontweight='bold')
        ax2.set_title('分析时间分布', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_parameter_sensitivity(self, experiment_data):
        """绘制参数敏感性"""
        if len(experiment_data) < 3:
            return None
        
        df = pd.DataFrame(experiment_data)
        params_array = np.array(df['params'].tolist())
        scores = df['score'].values
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 14), dpi=100)
        axes = axes.flatten()
        
        for idx in range(8):
            ax = axes[idx]
            
            param_name = PARAM_NAMES[idx]
            param_unit = PARAM_UNITS[param_name]
            param_values = params_array[:, idx]
            
            sorted_indices = np.argsort(param_values)
            sorted_params = param_values[sorted_indices]
            sorted_scores = scores[sorted_indices]
            
            scatter = ax.scatter(sorted_params, sorted_scores, 
                               c=df.index, cmap='viridis', s=120, 
                               alpha=0.7, edgecolors='black', linewidth=1.5)
            
            if len(sorted_params) > 2:
                z = np.polyfit(sorted_params, sorted_scores, 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(sorted_params.min(), sorted_params.max(), 100)
                ax.plot(x_smooth, p(x_smooth), '--', color='#d62728', 
                       linewidth=2.5, alpha=0.8, label='趋势')
            
            ax.set_xlabel(f'{param_name} ({param_unit})', fontsize=11, fontweight='bold')
            ax.set_ylabel('CRF 评分', fontsize=11, fontweight='bold')
            ax.set_title(f'敏感性分析: {param_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim([0, 1.05])
        
        fig.delaxes(axes[8])
        cbar = fig.colorbar(scatter, ax=axes[8], pad=0.1)
        cbar.set_label('迭代次数', fontsize=11, fontweight='bold')
        
        plt.suptitle('参数敏感性分析', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig
    
    def plot_high_dimensional_space(self, experiment_data):
        """绘制高维参数空间"""
        if len(experiment_data) < 2:
            return None
        
        df = pd.DataFrame(experiment_data)
        params_array = np.array(df['params'].tolist())
        scores = df['score'].values
        
        key_pairs = [
            (0, 2, 'T_init', 'R1'),
            (3, 6, 'T1', 'T2'),
            (2, 5, 'R1', 'R2')
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
        
        for ax_idx, (param1_idx, param2_idx, param1_name, param2_name) in enumerate(key_pairs):
            ax = axes[ax_idx]
            
            param1_values = params_array[:, param1_idx]
            param2_values = params_array[:, param2_idx]
            
            grid_x = np.linspace(param1_values.min(), param1_values.max(), 50)
            grid_y = np.linspace(param2_values.min(), param2_values.max(), 50)
            grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
            
            grid_Z = griddata(
                (param1_values, param2_values), scores,
                (grid_X, grid_Y), method='cubic'
            )
            
            contour = ax.contourf(grid_X, grid_Y, grid_Z, levels=15, cmap='RdYlGn', alpha=0.8)
            ax.contour(grid_X, grid_Y, grid_Z, levels=8, colors='black', alpha=0.3, linewidths=0.5)
            
            scatter = ax.scatter(param1_values, param2_values, c=scores, 
                               cmap='RdYlGn', s=150, edgecolors='black', 
                               linewidth=1.5, vmin=0, vmax=1, zorder=5)
            
            ax.set_xlabel(f'{param1_name} ({PARAM_UNITS[param1_name]})', 
                         fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{param2_name} ({PARAM_UNITS[param2_name]})', 
                         fontsize=12, fontweight='bold')
            ax.set_title(f'{param1_name} vs {param2_name}', fontsize=13, fontweight='bold')
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('CRF 评分', fontsize=10, fontweight='bold')
        
        plt.suptitle('高维参数空间探索', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        return fig