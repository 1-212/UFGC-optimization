"""GC optimization system visualization module"""


import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import griddata
from config import parameter_names, parameter_units, max_analysis_time, min_analysis_time

# Set font settings
try:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

class VisualizationManager:
    """GC optimization system visualization manager
    
    Responsible for visualizing system results and generating various analysis charts.
    """
    
    def __init__(self, work_dir="gc_optimization"):
        """Initialize visualization manager
        
        Args:
            work_dir: Working directory
        """
        self.work_dir = work_dir
        self.visualization_directory = os.path.join(work_dir, "visualizations")
        os.makedirs(self.visualization_directory, exist_ok=True)
        self.gp_model = None
    
    def set_gp_model(self, gp):
        """Set GP model
        
        Args:
            gp: Gaussian process model
        """
        self.gp_model = gp
    
    def save_figure(self, fig, name, dpi=300):
        """Save figure
        
        Args:
            fig: Figure object
            name: Figure name
            dpi: Resolution
            
        Returns:
            tuple: (PNG path, PDF path)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        png_path = os.path.join(self.visualization_directory, f"{name}_{timestamp}.png")
        pdf_path = os.path.join(self.visualization_directory, f"{name}_{timestamp}.pdf")
        
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', format='png')
        fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight', format='pdf')
        
        print(f"OK Figure saved: {png_path}")
        return png_path, pdf_path
    
    def plot_convergence_curve(self, experiment_data):
        """Plot convergence curve
        
        Show the trend of CRF score changes during optimization.
        
        Args:
            experiment_data: Experiment data list
            
        Returns:
            matplotlib.figure.Figure: Convergence curve figure
        """
        if len(experiment_data) < 1:
            return None
        
        dataframe = pd.DataFrame(experiment_data)
        
        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
        
        ax.plot(dataframe.index + 1, dataframe['score'], 'o-', linewidth=2.5, 
               markersize=8, label='Current score', color='#1f77b4', alpha=0.7)
        
        best_scores = dataframe['score'].cummax()
        ax.plot(dataframe.index + 1, best_scores, 's-', linewidth=3, 
               markersize=8, label='Best score (cumulative)', color='#d62728')
        
        initial_mask = dataframe['is_initial']
        if initial_mask.sum() > 0:
            ax.scatter(dataframe[initial_mask].index + 1, dataframe[initial_mask]['score'], 
                      color='#2ca02c', s=150, marker='^', label='Initial points', 
                      zorder=5, edgecolors='black', linewidth=1.5)
        
        optimized_mask = ~initial_mask
        if optimized_mask.sum() > 0:
            ax.scatter(dataframe[optimized_mask].index + 1, dataframe[optimized_mask]['score'], 
                      color='#ff7f0e', s=150, marker='o', label='Optimized points', 
                      zorder=5, edgecolors='black', linewidth=1.5)
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('CRF Score', fontsize=12, fontweight='bold')
        ax.set_title('Optimization Convergence Curve', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
        
        plt.tight_layout()
        return fig
    
    def plot_analysis_time_distribution(self, experiment_data):
        """Plot analysis time distribution
        
        Show the trend and distribution of analysis time.
        
        Args:
            experiment_data: Experiment data list
            
        Returns:
            matplotlib.figure.Figure: Analysis time distribution figure
        """
        if len(experiment_data) < 1:
            return None
        
        dataframe = pd.DataFrame(experiment_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
        
        ax1.plot(dataframe.index + 1, dataframe['analysis_time'], 'o-', linewidth=2.5, 
                markersize=8, color='#1f77b4', label='Analysis time')
        ax1.axhline(y=max_analysis_time, color='#d62728', linestyle='--', 
                   linewidth=2.5, label=f'Max limit ({max_analysis_time}s)')
        ax1.axhline(y=min_analysis_time, color='#2ca02c', linestyle='--', 
                   linewidth=2.5, label=f'Min limit ({min_analysis_time}s)')
        ax1.fill_between(dataframe.index + 1, min_analysis_time, max_analysis_time, 
                        alpha=0.1, color='gray', label='Valid range')
        
        ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Analysis time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Analysis Time Trend', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10)
        
        initial_times = dataframe[dataframe['is_initial']]['analysis_time']
        optimized_times = dataframe[~dataframe['is_initial']]['analysis_time']
        
        bins = np.linspace(dataframe['analysis_time'].min()-10, dataframe['analysis_time'].max()+10, 15)
        
        ax2.hist(initial_times, bins=bins, alpha=0.6, label='Initial points', 
                color='#2ca02c', edgecolor='black', linewidth=1.2)
        ax2.hist(optimized_times, bins=bins, alpha=0.6, label='Optimized points', 
                color='#ff7f0e', edgecolor='black', linewidth=1.2)
        
        ax2.set_xlabel('Analysis time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Analysis Time Distribution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_parameter_sensitivity(self, experiment_data):
        """Plot parameter sensitivity
        
        Analyze the impact of each parameter on CRF score.
        
        Args:
            experiment_data: Experiment data list
            
        Returns:
            matplotlib.figure.Figure: Parameter sensitivity analysis figure
        """
        if len(experiment_data) < 3:
            return None
        
        dataframe = pd.DataFrame(experiment_data)
        params_array = np.array(dataframe['params'].tolist())
        scores = dataframe['score'].values
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 14), dpi=100)
        axes = axes.flatten()
        
        for idx in range(8):
            ax = axes[idx]
            
            param_name = parameter_names[idx]
            param_unit = parameter_units[param_name]
            param_values = params_array[:, idx]
            
            sorted_indices = np.argsort(param_values)
            sorted_params = param_values[sorted_indices]
            sorted_scores = scores[sorted_indices]
            
            scatter = ax.scatter(sorted_params, sorted_scores, 
                               c=dataframe.index, cmap='viridis', s=120, 
                               alpha=0.7, edgecolors='black', linewidth=1.5)
            
            if len(sorted_params) > 2:
                z = np.polyfit(sorted_params, sorted_scores, 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(sorted_params.min(), sorted_params.max(), 100)
                ax.plot(x_smooth, p(x_smooth), '--', color='#d62728', 
                       linewidth=2.5, alpha=0.8, label='Trend')
            
            ax.set_xlabel(f'{param_name} ({param_unit})', fontsize=11, fontweight='bold')
            ax.set_ylabel('CRF Score', fontsize=11, fontweight='bold')
            ax.set_title(f'Sensitivity Analysis: {param_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim([0, 1.05])
        
        fig.delaxes(axes[8])
        cbar = fig.colorbar(scatter, ax=axes[8], pad=0.1)
        cbar.set_label('Iteration', fontsize=11, fontweight='bold')
        
        plt.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig
    
    def plot_high_dimensional_space(self, experiment_data):
        """Plot high-dimensional parameter space
        
        Show the interactions between parameters and optimal parameter regions.
        
        Args:
            experiment_data: Experiment data list
            
        Returns:
            matplotlib.figure.Figure: High-dimensional parameter space figure
        """
        if len(experiment_data) < 2:
            return None
        
        dataframe = pd.DataFrame(experiment_data)
        params_array = np.array(dataframe['params'].tolist())
        scores = dataframe['score'].values
        
        key_pairs = [
            (0, 2, 'initial_temperature', 'ramp_rate_1'),
            (3, 6, 'target_temperature_1', 'target_temperature_2'),
            (2, 5, 'ramp_rate_1', 'ramp_rate_2')
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
        
        for ax_idx, (param1_idx, param2_idx, param1_name, param2_name) in enumerate(key_pairs):
            ax = axes[ax_idx]
            
            param1_values = params_array[:, param1_idx]
            param2_values = params_array[:, param2_idx]
            
            grid_x = np.linspace(param1_values.min(), param1_values.max(), 50)
            grid_y = np.linspace(param2_values.min(), param2_values.max(), 50)
            grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
            
            # Check data dimensions to avoid Qhull error
            if len(np.unique(param1_values)) < 2 or len(np.unique(param2_values)) < 2:
                # If data is low-dimensional, use linear interpolation
                grid_Z = griddata(
                    (param1_values, param2_values), scores,
                    (grid_X, grid_Y), method='linear'
                )
            else:
                # Use cubic interpolation for normal cases
                grid_Z = griddata(
                    (param1_values, param2_values), scores,
                    (grid_X, grid_Y), method='cubic'
                )
            
            contour = ax.contourf(grid_X, grid_Y, grid_Z, levels=15, cmap='RdYlGn', alpha=0.8)
            ax.contour(grid_X, grid_Y, grid_Z, levels=8, colors='black', alpha=0.3, linewidths=0.5)
            
            scatter = ax.scatter(param1_values, param2_values, c=scores, 
                               cmap='RdYlGn', s=150, edgecolors='black', 
                               linewidth=1.5, vmin=0, vmax=1, zorder=5)
            
            ax.set_xlabel(f'{param1_name} ({parameter_units[param1_name]})', 
                         fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{param2_name} ({parameter_units[param2_name]})', 
                         fontsize=12, fontweight='bold')
            ax.set_title(f'{param1_name} vs {param2_name}', fontsize=13, fontweight='bold')
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('CRF Score', fontsize=10, fontweight='bold')
        
        plt.suptitle('High-dimensional Parameter Space Exploration', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        return fig