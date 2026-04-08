"""Main window module (Enhanced version: history recovery, robust generate/submit/stop flow)"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from skopt import Optimizer

from config import *
from core.constraint import ConstraintHandler, get_initial_parameter_points
from core.optimizer import create_gp_regressor, StoppingCriteria, BayesianOptimizer
from data.manager import ExperimentDataManager
from data.asc_parser import process_asc_files, calculate_crf_from_asc_data, extract_signals_and_regions_from_asc_data
from visualization.plotter import VisualizationManager
from utils.file_handler import FileHandler
from utils.logger import logger


class BayesianOptimizationUI:
    """Enhanced main window: Implements consistent behavior with iterative exception handling"""


    def __init__(self, root):
        self.root = root
        self.root.title(ui_config['window_title'])
        self.root.geometry(ui_config['window_size'])

        # Paths and directories
        self.work_dir = work_directory
        self.data_dir = os.path.join(self.work_dir, "data")
        self.results_dir = os.path.join(self.work_dir, "results")
        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoints")

        for d in [self.data_dir, self.results_dir, self.checkpoint_dir]:
            os.makedirs(d, exist_ok=True)

        # Modules
        self.constraint_handler = ConstraintHandler()
        self.viz_manager = VisualizationManager(self.work_dir)
        self.experiment_data = ExperimentDataManager()

        # State
        self.current_round = 0
        self.next_params = None
        self.optimizer = None
        self.initial_points_used = 0
        self.has_pending_params = False
        self.optimization_stopped = False
        self.stop_reason = None
        self.generate_retry_count = 0
        self.scaler_params = None

        # Stopping criteria
        # StoppingCriteria reads configuration from STOPPING_CONFIG internally, no parameters needed
        self.stopping_criteria = StoppingCriteria()

        # UI
        self.create_ui()

        # Load and restore history (if exists)
        if self.load_history():
            if len(self.experiment_data) > 0:
                self.restore_optimization_state()

        self.update_status()
        self.update_stopping_criteria_display()
        logger.info("GC temperature program optimization system started (enhanced version)")

    # ----------------------- History Management -----------------------
    def load_history(self):
        """Load history checkpoint (using FileHandler)"""

        state, exp_data = FileHandler.load_checkpoint()
        if state is None and exp_data is None:
            logger.info("No historical data detected")
            return False

        if exp_data:
            self.experiment_data.experiment_data = exp_data

        self.current_round = state.get('current_round', 0)
        self.initial_points_used = state.get('initial_points_used', 0)
        self.has_pending_params = state.get('has_pending_params', False)
        self.optimization_stopped = state.get('optimization_stopped', False)
        self.stop_reason = state.get('stop_reason', None)

        # Restore next_params (if there are pending scores)
        if self.has_pending_params and self.current_round > 0:
            # Previous round is pending (current_round points to generated but unsubmitted round)
            for exp in self.experiment_data.get_all_experiments():
                if exp['round'] == self.current_round - 1:
                    self.next_params = exp['params']
                    break

        logger.info(f"History loaded: {self.current_round} rounds, initial points used: {self.initial_points_used}")
        return True

    def save_history(self):
        """Save history checkpoint (using FileHandler)"""

        state = {
            'current_round': self.current_round,
            'initial_points_used': self.initial_points_used,
            'has_pending_params': self.has_pending_params,
            'optimization_stopped': self.optimization_stopped,
            'stop_reason': self.stop_reason,
            'timestamp': datetime.now().isoformat()
        }
        FileHandler.save_checkpoint(state, self.experiment_data.get_all_experiments())

    # ----------------------- Restoration and Rebuilding -----------------------
    def restore_optimization_state(self):
        """Rebuild optimizer and stopping criteria from history data, and restore UI state"""

        try:
            logger.info("Restoring optimizer and stopping criteria...")
            self.rebuild_optimizer()
            self.rebuild_stopping_criteria()

            if self.optimizer is not None:
                self.viz_manager.set_gp_model(self.optimizer.base_estimator_)

            # Update UI
            self.show_history_summary()
            self.update_stats()
            self.update_history()
            self.update_stopping_criteria_display()
            self.show_initial_points()

            # If optimization was stopped last time, disable interaction buttons
            if self.optimization_stopped:
                self.next_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.DISABLED)
            else:
                self.next_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.NORMAL)

                if self.has_pending_params:
                    self.result_text.insert(tk.END, f"⏳ Round {self.current_round} still has pending parameters, please submit score\n")
                    if self.next_params is not None:
                        self.display_next_params(self.next_params)
                    self.analyze_btn.config(state=tk.NORMAL)
                    self.submit_btn.config(state=tk.NORMAL)
                else:
                    self.result_text.insert(tk.END, f"✓ Completed {self.current_round} rounds, can continue optimization\n")

            logger.info("History restoration completed")
        except Exception as e:
            logger.error(f"Failed to restore history: {e}")
            self.result_text.insert(tk.END, f"❌ Restoration failed: {e}\n")

    def rebuild_optimizer(self):
        """Rebuild Skopt Optimizer with historical samples and optionally fit GP"""

        try:
            # Use Bayesian optimizer and populate with historical data
            self.optimizer = BayesianOptimizer()

            for exp in self.experiment_data.get_all_experiments():
                # Use composite score if available, otherwise use score
                score = exp.get('composite_score', exp.get('score', 0.0))
                try:
                    self.optimizer.tell(exp['params'], score)
                except Exception:
                    pass

            # If enough samples, optionally fit the internal GP
            logger.info("Bayesian optimizer rebuilt based on historical samples")

        except Exception as e:
            logger.error(f"Failed to rebuild optimizer: {e}")
            raise

    def rebuild_stopping_criteria(self):
        """Rebuild stopping criteria state with historical scores"""

        try:
            # StoppingCriteria uses internal configuration from STOPPING_CONFIG, no explicit parameters needed
            self.stopping_criteria = StoppingCriteria()

            for exp in self.experiment_data.get_all_experiments():
                self.stopping_criteria.update(exp['score'], exp['params'])

            logger.info("Stopping criteria rebuilt")
        except Exception as e:
            logger.error(f"Failed to rebuild stopping criteria: {e}")
            raise

    def show_history_summary(self):
        """Show historical data summary in results panel"""

        try:
            self.result_text.delete(1.0, tk.END)

            all_experiments = self.experiment_data.get_all_experiments()
            if not all_experiments:
                self.result_text.insert(tk.END, "No historical data available\n")
                return

            df = pd.DataFrame(all_experiments)

            self.result_text.insert(tk.END, f"\n{'='*50}\n")
            self.result_text.insert(tk.END, f"📖 Historical Optimization Data Summary\n")
            self.result_text.insert(tk.END, f"{'='*50}\n\n")

            self.result_text.insert(tk.END, f"Total rounds: {len(df)}\n")

            initial_df = df[df['is_initial']]
            optimized_df = df[~df['is_initial']]

            self.result_text.insert(tk.END, f"  • Initial points: {len(initial_df)}/{max_initial_points}\n")
            self.result_text.insert(tk.END, f"  • Optimized points: {len(optimized_df)}\n\n")

            self.result_text.insert(tk.END, f"CRF value statistics:\n")
            self.result_text.insert(tk.END, f"  • Highest: {df['score'].max():.4f} (Round {df['score'].idxmax()+1})\n")
            self.result_text.insert(tk.END, f"  • Average: {df['score'].mean():.4f}\n")
            self.result_text.insert(tk.END, f"  • Lowest: {df['score'].min():.4f}\n")
            self.result_text.insert(tk.END, f"  • Standard deviation: {df['score'].std():.4f}\n\n")

            self.result_text.insert(tk.END, f"Analysis time:\n")
            self.result_text.insert(tk.END, f"  • Average: {df['analysis_time'].mean():.1f}s\n")
            self.result_text.insert(tk.END, f"  • Longest: {df['analysis_time'].max():.1f}s\n\n")

            self.result_text.insert(tk.END, f"{'='*50}\n")
            self.result_text.insert(tk.END, f"Recent 5 rounds:\n")
            self.result_text.insert(tk.END, f"{'='*50}\n")

            for idx, exp in enumerate(all_experiments[-5:], start=max(1, len(all_experiments)-4)):
                point_type = "Initial point" if exp['is_initial'] else "Optimized point"
                self.result_text.insert(tk.END, 
                    f"Round {idx} ({point_type}): CRF={exp['score']:.4f}, Time={exp['analysis_time']:.1f}s\n")

            self.result_text.insert(tk.END, f"\n{'='*50}\n\n")

        except Exception as e:
            logger.error(f"Failed to display history summary: {e}")
            self.result_text.insert(tk.END, f"❌ Failed to display history summary: {e}\n")

    def compute_composite_score(self, crf_value, analysis_time, iteration, total_iterations):
        """Compute composite score: Add analysis time penalty to CRF.

        New time penalty formula:
        f_time = 1 - (t - T_min) / (T_max - T_min)
        w_sep = 0.7 + 0.3 * (1 - n/N)
        w_time = 1 - w_sep
        composite = w_sep * f_sep + w_time * f_time
        where:
        - t: Actual analysis time
        - T_min: Minimum analysis time (80s)
        - T_max: Maximum analysis time (300s)
        - n: Current iteration
        - N: Total iterations
        - f_sep: Original CRF value
        Returns composite score (float)
        """

        try:
            # Calculate time function f_time
            t_min = 80  # Minimum analysis time
            t_max = 300 # Maximum analysis time
            f_time = 1 - (analysis_time - t_min) / (t_max - t_min)
            f_time = max(0.0, min(1.0, f_time))  # Limit to [0,1] range
            
            # Calculate weights
            w_sep = 0.7 + 0.3 * (1 - iteration / total_iterations)
            w_time = 1 - w_sep
            
            # Calculate composite score
            f_sep = float(crf_value)
            composite = w_sep * f_sep + w_time * f_time
            # Ensure numerical stability
            return composite
        except Exception as e:
            logger.warning(f"Failed to compute composite score: {e}")
            return float(crf_value)

    # ----------------------- UI Creation -----------------------
    def create_ui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_top_panel(main_container)

        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.create_left_panel(content_frame)
        self.create_right_panel(content_frame)

        self.create_status_bar(main_container)

    def create_top_panel(self, parent):
        top_frame = ttk.LabelFrame(parent, text="🎯 Control Panel", padding=10)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(top_frame)
        row1.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(row1, text="📂 Select ASC Files", command=self.select_asc_files).pack(side=tk.LEFT, padx=5)

        ttk.Button(row1, text="📊 Analyze ASC Files", command=self.analyze_asc_files, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        self.analyze_btn = row1.winfo_children()[-1]

        ttk.Button(row1, text="🚀 Initialize Optimization", command=self.initialize_optimization).pack(side=tk.LEFT, padx=5)

        ttk.Button(row1, text="⚡ Generate Next Params", command=self.generate_next_params, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        self.next_btn = row1.winfo_children()[-1]

        ttk.Button(row1, text="⛔ Stop Optimization", command=self.stop_optimization, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        self.stop_btn = row1.winfo_children()[-1]

        ttk.Button(row1, text="📈 Generate All Charts", command=self.show_all_visualizations, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        self.plot_btn = row1.winfo_children()[-1]

        ttk.Button(row1, text="🔄 Reset", command=self.reset_all).pack(side=tk.LEFT, padx=5)

        # Second row: Score input and submission
        row2 = ttk.Frame(top_frame)
        row2.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(row2, text="📝 CRF Value (0.0-1.0) for current params:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        self.crf_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.crf_var, width=10).pack(side=tk.LEFT, padx=5)

        ttk.Button(row2, text="✓ Submit Score → Generate Next", command=self.submit_crf_and_generate_next).pack(side=tk.LEFT, padx=5)
        self.submit_btn = row2.winfo_children()[-1]
        self.submit_btn.config(state=tk.DISABLED)

        ttk.Label(row2, text="  | Progress:").pack(side=tk.LEFT, padx=10)
        self.progress_var = tk.StringVar(value="Not started")
        ttk.Label(row2, textvariable=self.progress_var, foreground='blue', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=2)

        ttk.Button(row2, text="💾 Export Results", command=self.export_results).pack(side=tk.RIGHT, padx=5)

    def create_left_panel(self, parent):
        left_frame = ttk.LabelFrame(parent, text="📋 Current Params / Param List", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        text_frame = ttk.Frame(left_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.param_text = tk.Text(text_frame, height=30, width=45, yscrollcommand=scrollbar.set, font=('Courier', 10))
        self.param_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.param_text.yview)

    def create_right_panel(self, parent):
        right_frame = ttk.LabelFrame(parent, text="📊 Optimization Results", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Log
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="📝 Optimization Log")
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text = tk.Text(log_frame, height=30, width=45, yscrollcommand=scrollbar.set, font=('Courier', 9))
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.result_text.yview)

        # Statistics
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="📈 Statistics")
        self.stats_text = tk.Text(stats_frame, height=30, width=45, font=('Courier', 9))
        scrollbar2 = ttk.Scrollbar(stats_frame, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=scrollbar2.set)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)

        # History
        history_frame = ttk.Frame(notebook)
        notebook.add(history_frame, text="📊 Parameter History")
        scrollbar3 = ttk.Scrollbar(history_frame)
        scrollbar3.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_text = tk.Text(history_frame, height=30, width=45, yscrollcommand=scrollbar3.set, font=('Courier', 8))
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar3.config(command=self.history_text.yview)

        # Stopping Criteria
        stop_frame = ttk.Frame(notebook)
        notebook.add(stop_frame, text="⛔ Stopping Criteria")
        scrollbar4 = ttk.Scrollbar(stop_frame)
        scrollbar4.pack(side=tk.RIGHT, fill=tk.Y)
        self.stop_text = tk.Text(stop_frame, height=30, width=45, yscrollcommand=scrollbar4.set, font=('Courier', 9))
        self.stop_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar4.config(command=self.stop_text.yview)

    def create_status_bar(self, parent):
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.pack(fill=tk.X, side=tk.LEFT, expand=True)

        self.round_var = tk.StringVar(value="Round: 0")
        round_label = ttk.Label(status_frame, textvariable=self.round_var, relief=tk.SUNKEN, width=15)
        round_label.pack(side=tk.LEFT, padx=(5, 0))

        self.best_var = tk.StringVar(value="Best: --")
        best_label = ttk.Label(status_frame, textvariable=self.best_var, relief=tk.SUNKEN, width=15)
        best_label.pack(side=tk.LEFT, padx=(5, 0))

        self.opt_status_var = tk.StringVar(value="Status: Not running")
        opt_label = ttk.Label(status_frame, textvariable=self.opt_status_var, relief=tk.SUNKEN, width=20)
        opt_label.pack(side=tk.LEFT, padx=(5, 0))

    # ----------------------- Status / Display -----------------------
    def update_status(self):
        self.round_var.set(f"Round: {self.current_round}")

        if self.has_pending_params:
            self.progress_var.set(f"⏳ Waiting for score (Round {self.current_round})")
        elif self.optimizer is not None:
            self.progress_var.set(f"✓ Completed {self.current_round} rounds")
        else:
            self.progress_var.set("Not started")

        all_experiments = self.experiment_data.get_all_experiments()
        if all_experiments:
            best_score = max([e['score'] for e in all_experiments])
            self.best_var.set(f"Best: {best_score:.4f}")
        else:
            self.best_var.set("Best: --")

        if self.optimization_stopped:
            self.opt_status_var.set("Status: Stopped ✓")
        else:
            self.opt_status_var.set("Status: Running ▶")

    def show_initial_points(self):
        self.param_text.delete(1.0, tk.END)

        initial_points = get_initial_parameter_points()

        self.param_text.insert(tk.END, "="*50 + "\n")
        self.param_text.insert(tk.END, "   Initial Experimental Points (10 total)\n")
        self.param_text.insert(tk.END, "="*50 + "\n\n")

        for idx, point in enumerate(initial_points):
            analysis_time = self.constraint_handler.calculate_analysis_time(point)
            if idx < self.initial_points_used:
                used_marker = "✓ "
                status = "Completed"
            elif idx == self.initial_points_used and self.has_pending_params:
                used_marker = "⏳ "
                status = "Pending Score"
            else:
                used_marker = "  "
                status = "Pending"

            self.param_text.insert(tk.END, f"{used_marker}【Initial Point #{idx+1}】({status})\n")

            for i, (name, value) in enumerate(zip(parameter_names, point)):
                unit = parameter_units[name]
                # Ensure value is scalar, handle numpy array cases
                if hasattr(value, '__len__') and hasattr(value, '__getitem__'):
                    # If numpy scalar, use .item() method
                    if hasattr(value, 'item'):
                        scalar_value = value.item()
                    elif len(value) == 1:
                        scalar_value = float(value[0])
                    else:
                        scalar_value = float(value[0])  # Take first element
                else:
                    scalar_value = float(value)
                self.param_text.insert(tk.END, f"  {name:<12}: {scalar_value:>8.2f} {unit}\n")

            self.param_text.insert(tk.END, f"  Analysis time: {analysis_time:.1f}s\n")
            self.param_text.insert(tk.END, "-"*50 + "\n\n")

    # ----------------------- File Selection / Analysis -----------------------
    def select_asc_files(self):
        files = filedialog.askopenfilenames(
            title="Select ASC Data Files",
            filetypes=[("ASC files", "*.asc"), ("All files", "*.*")]
        )

        if files:
            self.current_asc_files = list(files)
            self.status_var.set(f"✓ Selected {len(files)} ASC files")
            self.analyze_btn.config(state=tk.NORMAL)

    def analyze_asc_files(self):
        if not self.current_asc_files:
            messagebox.showwarning("Warning", "Please select ASC files first")
            return

        self.status_var.set("Analyzing ASC files...")
        self.root.update()

        try:
            df = process_asc_files(self.current_asc_files, use_half_data=True)

            if df is not None:
                # Save ASC data for new ECRF calculation
                self._current_asc_df = df
                crf_value = calculate_crf_from_asc_data(df)

                self.result_text.insert(tk.END, f"\n{'='*45}\n")
                self.result_text.insert(tk.END, f"ASC file analysis completed \n")
                self.result_text.insert(tk.END, f"File count: {len(self.current_asc_files)}\n")
                self.result_text.insert(tk.END, f"Auto-calculated CRF: {crf_value:.4f}\n")
                self.result_text.insert(tk.END, f"{'='*45}\n\n")

                self.result_text.see(tk.END)

                self.crf_var.set(f"{crf_value:.4f}")

                self.status_var.set(f"✓ ASC analysis completed, CRF: {crf_value:.4f}")

                if self.has_pending_params:
                    self.submit_btn.config(state=tk.NORMAL)
            else:
                messagebox.showerror("Error", "Failed to process ASC files")
                self.status_var.set("✗ File processing failed")

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_var.set(f"✗ Analysis failed: {str(e)}")

    # ----------------------- Initialization / Parameter Generation -----------------------
    def initialize_optimization(self):
        all_experiments = self.experiment_data.get_all_experiments()

        if all_experiments:
            choice = messagebox.askyesno(
                "Confirmation",
                f"Historical optimization data detected ({self.current_round} rounds)\n\n"
                f"Choose:\n"
                f"【Yes】 Continue with historical optimization\n"
                f"【No】 Start new optimization"
            )

            if choice:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"✓ Continuing historical optimization...\n\n")
                self.show_history_summary()
                self.result_text.see(tk.END)
                self.next_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.NORMAL)

                self.update_stats()
                self.update_history()
                self.update_stopping_criteria_display()
                self.show_initial_points()

                self.status_var.set("✓ Continuing historical optimization")

                if self.current_round >= 2:
                    self.plot_btn.config(state=tk.NORMAL)

                if self.has_pending_params:
                    self.result_text.insert(tk.END, 
                        f"⏳ Round {self.current_round} still has pending parameters\n")
                    self.result_text.insert(tk.END, f"Please enter CRF value above and submit\n\n")

                    self.result_text.see(tk.END)

                    if self.next_params is not None:
                        self.display_next_params(self.next_params)

                    self.analyze_btn.config(state=tk.NORMAL)
                    self.submit_btn.config(state=tk.NORMAL)
                else:
                    self.result_text.insert(tk.END, 
                        f"✓ Completed {self.current_round} rounds\n")
                    self.result_text.insert(tk.END, f"Click 【⚡Generate Next Params】 to continue optimization\n\n")

                    self.result_text.see(tk.END)
                    self.show_initial_points()

                return
            else:
                self.reset_all()
                return

        if self.optimization_stopped:
            messagebox.showwarning("Warning", "Please reset before starting new optimization")
            return

        self.initial_points_used = 0
        self.has_pending_params = False

        initial_points = get_initial_parameter_points()
        initial_points = [self.constraint_handler.project_to_feasible_region(p) for p in initial_points]

        # Use Bayesian optimizer
        self.optimizer = BayesianOptimizer()

        for point in initial_points:
            # Use initial points as warm start
            self.optimizer.tell(point, 0.0)

        self.viz_manager.set_gp_model(self.optimizer.base_estimator_)

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"\n{'='*45}\n")
        self.result_text.insert(tk.END, f"🚀 New Bayesian optimization initialized\n")
        self.result_text.insert(tk.END, f"{'='*45}\n")
        self.result_text.insert(tk.END, f"Initial points: {max_initial_points}\n")
        self.result_text.insert(tk.END, f"Analysis time range: {min_analysis_time}-{max_analysis_time} seconds\n")
        self.result_text.insert(tk.END, f"{'='*45}\n\n")

        self.result_text.insert(tk.END, f"💡 Next, click 【⚡Generate Next Params】 to start optimization\n\n")
        self.result_text.see(tk.END)

        self.status_var.set("✓ Optimizer initialized")
        self.next_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.show_initial_points()
        self.update_status()

    def generate_next_params(self):
        if self.optimizer is None:
            messagebox.showwarning("Warning", "Please initialize optimizer first")
            return

        if self.optimization_stopped:
            messagebox.showinfo("Info", "Optimization has stopped, please reset and restart")
            return

        if self.has_pending_params:
            messagebox.showwarning("Warning", "There are pending parameters, please submit score first")
            return

        self.generate_retry_count = 0
        self._generate_next_params_impl()

    def _generate_next_params_impl(self):
        try:
            if self.generate_retry_count >= generate_retry_limit:
                messagebox.showerror(
                    "Error",
                    f"⚠️ Failed to generate feasible parameters {generate_retry_limit} consecutive times\n"
                    f"Suggest adjusting parameter constraints"
                )
                return

            if self.initial_points_used < max_initial_points:
                initial_points = get_initial_parameter_points()
                next_point = initial_points[self.initial_points_used]
                source = "Initial point library"
            else:
                try:
                    next_point = self.optimizer.ask()
                    source = "Optimizer"
                except Exception as e:
                    logger.warning(f"Optimizer generation failed: {e}, using heuristic method")
                    next_point = self.constraint_handler.generate_feasible_heuristic()
                    source = "Heuristic method"

            self.next_params = self.constraint_handler.project_to_feasible_region(next_point)

            is_feasible, msg = self.constraint_handler.is_feasible(self.next_params)

            if not is_feasible:
                self.generate_retry_count += 1
                self.result_text.insert(tk.END, f"\n⚠️ Attempt {self.generate_retry_count}: Generated parameters do not satisfy constraints: {msg}, regenerating...\n")
                self.result_text.see(tk.END)
                self.root.update()
                self._generate_next_params_impl()
                return

            self.has_pending_params = True
            self.current_round += 1
            self.generate_retry_count = 0

            is_initial = self.initial_points_used < max_initial_points

            if is_initial:
                self.show_initial_points()
            else:
                self.display_next_params(self.next_params)

            self.submit_btn.config(state=tk.NORMAL)

            point_type = "Initial point" if is_initial else "Optimized point"
            analysis_time = self.constraint_handler.calculate_analysis_time(self.next_params)

            self.result_text.insert(tk.END, f"\n✓ Round {self.current_round} parameters generated ({point_type}, source: {source})\n")
            self.result_text.insert(tk.END, f"Analysis time: {analysis_time:.1f}s (range: {min_analysis_time}-{max_analysis_time}s)\n")
            self.result_text.insert(tk.END, f"⏳ Please run experiment and enter CRF value\n\n")
            self.result_text.see(tk.END)

            self.update_status()
            self.status_var.set(f"✓ Round {self.current_round} parameters generated")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate parameters: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"✗ Failed to generate parameters: {str(e)}")

    def submit_crf_and_generate_next(self):
        if not self.has_pending_params:
            messagebox.showwarning("Warning", "No pending parameters to score")
            return

        if self.next_params is None:
            messagebox.showwarning("Warning", "Parameters lost, regenerating...")
            self.has_pending_params = False
            self.crf_var.set("")
            self.submit_btn.config(state=tk.DISABLED)
            self.generate_next_params()
            return

        try:
            crf_str = self.crf_var.get().strip()
            if not crf_str:
                messagebox.showwarning("Warning", "Please enter CRF value")
                return

            try:
                crf_value = float(crf_str)
            except ValueError:
                messagebox.showerror("Error", "CRF value must be a number")
                return

            if not (0 <= crf_value <= 1.0):
                messagebox.showerror("Error", "CRF value must be between 0.0-1.0")
                return

            analysis_time = self.constraint_handler.calculate_analysis_time(self.next_params)

            is_feasible, msg = self.constraint_handler.is_feasible(self.next_params)
            if not is_feasible:
                messagebox.showerror("Error", f"Parameters do not satisfy constraints: {msg}")
                return

            test_exp = {
                'round': self.current_round,
                'params': list(self.next_params),
                'score': crf_value,
                'analysis_time': analysis_time,
                'timestamp': datetime.now().isoformat(),
                'is_initial': self.initial_points_used < max_initial_points
            }

            if len(test_exp['params']) != len(parameter_names):
                raise ValueError(
                    f"Parameter length mismatch: {len(test_exp['params'])} != {len(parameter_names)}"
                )

            # Get current iteration
            status = self.stopping_criteria.get_status()
            current_iteration = status.get('iteration', 1)
            total_iterations = self.stopping_criteria.max_iterations
            
            # Calculate composite score and save to record (for GP fitting and optimization target)
            composite_score = self.compute_composite_score(crf_value, analysis_time, current_iteration, total_iterations)
            test_exp['composite_score'] = composite_score
            
            # Calculate time function f_time
            # Assume reasonable T_min and T_max values
            t_min = 80  # Can be adjusted based on actual conditions
            t_max = 300 # Can be adjusted based on actual conditions
            f_time = 1 - (analysis_time - t_min) / (t_max - t_min)
            f_time = np.clip(f_time, 0.0, 1.0)  # Limit to [0,1] range
            
            # Determine f_sep calculation method based on ASC data availability
            if hasattr(self, '_current_asc_df') and self._current_asc_df is not None:
                from core.signal_process import calculate_ecrf_comprehensive
                signal_data, regions = extract_signals_and_regions_from_asc_data(self._current_asc_df)
                if signal_data is not None and len(regions) > 0:
                    # With ASC data, use complete new ECRF calculation
                    try:
                        new_ecrf = calculate_ecrf_comprehensive(
                            signal=signal_data,
                            peak_regions=regions,
                            analysis_time=analysis_time,
                            current_iteration=current_iteration,
                            total_iterations=total_iterations
                        )
                        test_exp['ecrf'] = new_ecrf
                    except Exception as e:
                        logger.warning(f"Failed to calculate complete new ECRF: {e}, using simplified calculation")
                        # Use manually input CRF value as f_sep
                        f_sep = crf_value  # Use user-input CRF value as separation quality part
                        w_sep = 0.7 + 0.3 * (1 - current_iteration / total_iterations)
                        w_time = 1 - w_sep
                        simplified_ecrf = w_sep * f_sep + w_time * f_time
                        test_exp['ecrf'] = simplified_ecrf
                else:
                    # No valid signal data, use manually input CRF value
                    f_sep = crf_value
                    w_sep = 0.7 + 0.3 * (1 - current_iteration / total_iterations)
                    w_time = 1 - w_sep
                    simplified_ecrf = w_sep * f_sep + w_time * f_time
                    test_exp['ecrf'] = simplified_ecrf
            else:
                # No ASC data, use manually input CRF value as f_sep
                f_sep = crf_value  # Use user-input CRF value as separation quality part
                w_sep = 0.7 + 0.3 * (1 - current_iteration / total_iterations)
                w_time = 1 - w_sep
                simplified_ecrf = w_sep * f_sep + w_time * f_time
                test_exp['ecrf'] = simplified_ecrf
            self.experiment_data.add_experiment(test_exp)

            try:
                # Prefer new ECRF as optimization target, otherwise use traditional composite score
                if test_exp.get('ecrf', 0.0) != 0.0:
                    # Use new ECRF as optimization target
                    optimization_target = test_exp['ecrf']
                else:
                    # No new ECRF, use traditional composite score
                    optimization_target = composite_score
                
                # Pass optimization target to optimizer (negative sign because skopt minimizes by default)
                self.optimizer.tell(self.next_params, -optimization_target)
                # Stopping criteria based on selected optimization target
                self.stopping_criteria.update(optimization_target, self.next_params)
            except Exception as e:
                logger.warning(f"Optimizer update warning: {e}")

            if test_exp['is_initial']:
                self.initial_points_used += 1

            if len(self.experiment_data) >= 5:
                try:
                    all_experiments = self.experiment_data.get_all_experiments()
                    X_data = np.array([exp['params'] for exp in all_experiments])
                    y_data = np.array([exp.get('composite_score', exp['score']) for exp in all_experiments])

                    X_mean = X_data.mean(axis=0)
                    X_std = X_data.std(axis=0) + 1e-8
                    X_normalized = (X_data - X_mean) / X_std

                    self.scaler_params = {'mean': X_mean, 'std': X_std}

                    gp = self.optimizer.base_estimator_
                    gp.fit(X_normalized, y_data)
                    self.viz_manager.set_gp_model(gp)
                except Exception as e:
                    logger.warning(f"GP model update warning: {e}")

            self.has_pending_params = False
            self.next_params = None
            self.crf_var.set("")
            self.submit_btn.config(state=tk.DISABLED)

            self.update_stats()
            self.update_history()
            self.update_stopping_criteria_display()
            self.save_history()

            point_type = "【Initial Point】" if test_exp['is_initial'] else "【Optimized Point】"
            self.result_text.insert(tk.END, f"\n{point_type} Round {self.current_round} - Scoring completed\n")
            self.result_text.insert(tk.END, f"CRF value: {crf_value:.4f}\n")
            self.result_text.insert(tk.END, f"Analysis time: {analysis_time:.1f}s ✓\n\n")
            self.result_text.see(tk.END)

            self.update_status()

            if self.current_round >= 2:
                self.plot_btn.config(state=tk.NORMAL)

            should_stop, reasons = self.stopping_criteria.should_stop()
            if should_stop:
                self.stop_optimization(reasons)
                return

            self.result_text.insert(tk.END, f"→ Automatically generating Round {self.current_round + 1} parameters...\n\n")
            self.result_text.see(tk.END)
            self.root.update()

            self.generate_next_params()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to submit score: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"✗ Failed to submit score: {str(e)}")

    def stop_optimization(self, reasons=None):
        self.optimization_stopped = True
        self.stop_reason = " | ".join(reasons) if reasons else "Manual stop by user"

        self.stop_btn.config(state=tk.DISABLED)
        self.next_btn.config(state=tk.DISABLED)
        self.submit_btn.config(state=tk.DISABLED)

        if reasons is None:
            reasons = ["Manual stop by user"]

        self.result_text.insert(tk.END, f"\n{'='*45}\n")
        self.result_text.insert(tk.END, "⛔ Optimization stopped\n")
        self.result_text.insert(tk.END, f"{'='*45}\n")
        self.result_text.insert(tk.END, "Stop reasons:\n")
        for reason in reasons:
            self.result_text.insert(tk.END, f"  • {reason}\n")
        self.result_text.insert(tk.END, f"{'='*45}\n\n")
        self.result_text.see(tk.END)

        self.save_history()

        self.status_var.set("⛔ Optimization stopped")
        self.update_status()

        messagebox.showinfo("Optimization Stopped", f"Optimization has stopped\n\nStop reasons:\n" + "\n".join(reasons))

    def display_next_params(self, params):
        self.param_text.delete(1.0, tk.END)

        is_feasible, warnings_msg = self.constraint_handler.is_feasible(params)

        self.param_text.insert(tk.END, "="*50 + "\n")
        self.param_text.insert(tk.END, f"Round {self.current_round} Experiment Parameters (【Optimized Point】)\n")
        self.param_text.insert(tk.END, "【Pending】\n")
        self.param_text.insert(tk.END, "="*50 + "\n\n")

        for i, (name, value) in enumerate(zip(parameter_names, params)):
            unit = parameter_units[name]
            dim = parameter_dimensions[i]
            # Ensure value is scalar, handle numpy array cases
            if hasattr(value, '__len__') and hasattr(value, '__getitem__'):
                # If numpy scalar, use .item() method
                if hasattr(value, 'item'):
                    scalar_value = value.item()
                elif len(value) == 1:
                    scalar_value = float(value[0])
                else:
                    scalar_value = float(value[0])  # Take first element
            else:
                scalar_value = float(value)
            self.param_text.insert(tk.END, f"{name:<12}: {scalar_value:>8.2f} {unit:6} [{dim.low:6.1f}, {dim.high:6.1f}]\n")

        analysis_time = self.constraint_handler.calculate_analysis_time(params)
        self.param_text.insert(tk.END, f"\n{'Analysis time':<12}: {analysis_time:>8.1f} s")

        if analysis_time > max_analysis_time:
            self.param_text.insert(tk.END, f" ❌ (Exceeds limit {max_analysis_time}s)\n")
        elif analysis_time < min_analysis_time:
            self.param_text.insert(tk.END, f" ⚠️ (Below minimum {min_analysis_time}s)\n")
        else:
            margin = min(analysis_time - min_analysis_time, max_analysis_time - analysis_time)
            self.param_text.insert(tk.END, f" ✓ (Margin: ±{margin:.1f}s)\n")

        if is_feasible:
            self.param_text.insert(tk.END, "\n✅ Parameter validation passed\n")
        else:
            self.param_text.insert(tk.END, "\n⚠️ Parameter validation warning:\n")
            self.param_text.insert(tk.END, f"  {warnings_msg}\n")

        self.param_text.insert(tk.END, "\n" + "="*50 + "\n")
        self.param_text.insert(tk.END, "✅ Next steps:\n")
        self.param_text.insert(tk.END, "1. Run the above parameters on the GC instrument\n")
        self.param_text.insert(tk.END, "2. Enter the experiment's CRF value in the input box above\n")
        self.param_text.insert(tk.END, "3. Click the 【Submit Score → Generate Next】 button\n")
        self.param_text.insert(tk.END, "="*50 + "\n")

        self.param_text.see(1.0)

    def update_stats(self):
        self.stats_text.delete(1.0, tk.END)

        all_experiments = self.experiment_data.get_all_experiments()

        if not all_experiments:
            self.stats_text.insert(tk.END, "No data available\n")
            return

        df = pd.DataFrame(all_experiments)

        self.stats_text.insert(tk.END, "="*45 + "\n")
        self.stats_text.insert(tk.END, "Optimization Statistics\n")
        self.stats_text.insert(tk.END, "="*45 + "\n\n")

        initial_df = df[df['is_initial']]
        optimized_df = df[~df['is_initial']]

        self.stats_text.insert(tk.END, f"Total rounds: {len(df)}\n")
        self.stats_text.insert(tk.END, f"  Initial points: {len(initial_df)}/{max_initial_points}\n")
        self.stats_text.insert(tk.END, f"  Optimized points: {len(optimized_df)}\n\n")

        self.stats_text.insert(tk.END, f"Average CRF: {df['score'].mean():.4f}\n")
        self.stats_text.insert(tk.END, f"Highest CRF: {df['score'].max():.4f} (Round {df['score'].idxmax()+1})\n")
        self.stats_text.insert(tk.END, f"Lowest CRF: {df['score'].min():.4f}\n")
        self.stats_text.insert(tk.END, f"Standard deviation: {df['score'].std():.4f}\n\n")

        self.stats_text.insert(tk.END, "Analysis time statistics:\n")
        self.stats_text.insert(tk.END, f"  Average: {df['analysis_time'].mean():.1f}s\n")
        self.stats_text.insert(tk.END, f"  Shortest: {df['analysis_time'].min():.1f}s\n")
        self.stats_text.insert(tk.END, f"  Longest: {df['analysis_time'].max():.1f}s\n\n")

        best_idx = df['score'].idxmax()
        best_row = df.loc[best_idx]

        self.stats_text.insert(tk.END, "="*45 + "\n")
        self.stats_text.insert(tk.END, f"Best Configuration (Round {best_idx+1})\n")
        self.stats_text.insert(tk.END, "="*45 + "\n")
        self.stats_text.insert(tk.END, f"CRF value: {best_row['score']:.4f}\n")
        self.stats_text.insert(tk.END, f"Analysis time: {best_row['analysis_time']:.1f}s\n\n")

        for i, (name, value) in enumerate(zip(parameter_names, best_row['params'])):
            unit = parameter_units[name]
            # Ensure value is scalar, handle numpy array cases
            if hasattr(value, '__len__') and hasattr(value, '__getitem__'):
                # If numpy scalar, use .item() method
                if hasattr(value, 'item'):
                    scalar_value = value.item()
                elif len(value) == 1:
                    scalar_value = float(value[0])
                else:
                    scalar_value = float(value[0])  # Take first element
            else:
                scalar_value = float(value)
            self.stats_text.insert(tk.END, f"{name}: {scalar_value:.2f} {unit}\n")

    def update_history(self):
        self.history_text.delete(1.0, tk.END)

        all_experiments = self.experiment_data.get_all_experiments()

        if not all_experiments:
            self.history_text.insert(tk.END, "No data available\n")
            return

        self.history_text.insert(tk.END, f"{'Round':<4} {'Type':<6} {'CRF':<8} {'T_init':<8} {'R1':<8} {'T1':<8} {'T2':<8} {'Time':<8}\n")
        self.history_text.insert(tk.END, "-"*60 + "\n")

        for exp in all_experiments:
            params = exp['params']
            point_type = "Initial" if exp['is_initial'] else "Opt"
            
            # Ensure parameter values are scalars, handle numpy array cases
            def safe_float_convert(val):
                if hasattr(val, '__len__') and hasattr(val, '__getitem__'):
                    if hasattr(val, 'item'):
                        return val.item()
                    elif len(val) == 1:
                        return float(val[0])
                    else:
                        return float(val[0])
                else:
                    return float(val)
            
            p0 = safe_float_convert(params[0])
            p2 = safe_float_convert(params[2])
            p3 = safe_float_convert(params[3])
            p6 = safe_float_convert(params[6])
            
            self.history_text.insert(tk.END, 
                f"{exp['round']:<4} "
                f"{point_type:<6} "
                f"{exp['score']:<8.4f} "
                f"{p0:<8.2f} "
                f"{p2:<8.2f} "
                f"{p3:<8.2f} "
                f"{p6:<8.2f} "
                f"{exp['analysis_time']:<8.1f}\n"
            )

    def update_stopping_criteria_display(self):
        self.stop_text.delete(1.0, tk.END)

        status = self.stopping_criteria.get_status()

        self.stop_text.insert(tk.END, "="*45 + "\n")
        self.stop_text.insert(tk.END, "Optimization Stopping Criteria\n")
        self.stop_text.insert(tk.END, "="*45 + "\n\n")

        max_iter = self.stopping_criteria.max_iterations
        current_iter = status['iteration']
        progress1 = (current_iter / max_iter) * 100 if max_iter > 0 else 0
        self.stop_text.insert(tk.END, f"1️⃣ Maximum Iterations\n")
        self.stop_text.insert(tk.END, f"   {current_iter}/{max_iter} ({progress1:.1f}%)\n")
        self.stop_text.insert(tk.END, f"   {'█' * int(progress1/5)}{'░' * (20-int(progress1/5))}\n\n")

        max_no_improve = self.stopping_criteria.max_no_improvement
        no_improve = status['no_improvement_count']
        progress2 = (no_improve / max_no_improve) * 100 if max_no_improve > 0 else 0
        self.stop_text.insert(tk.END, f"2️⃣ No Improvement Count\n")
        self.stop_text.insert(tk.END, f"   {no_improve}/{max_no_improve} ({progress2:.1f}%)\n")
        self.stop_text.insert(tk.END, f"   {'█' * int(progress2/5)}{'░' * (20-int(progress2/5))}\n\n")

        best_score = status['best_score']
        target = self.stopping_criteria.target_score
        if target > 0 and best_score != -np.inf:
            progress3 = (best_score / target) * 100
            progress3 = max(0, min(100, progress3))  # Limit to 0-100
        else:
            progress3 = 0
        self.stop_text.insert(tk.END, f"3️⃣ Target Score\n")
        self.stop_text.insert(tk.END, f"   {best_score:.4f}/{target:.4f} ({progress3:.1f}%)\n")
        self.stop_text.insert(tk.END, f"   {'█' * int(progress3/5)}{'░' * (20-int(progress3/5))}\n\n")

        recent = status['recent_scores']
        if len(recent) >= 3:
            variance = np.var(recent)
            threshold = self.stopping_criteria.convergence_threshold ** 2
            progress4 = ((threshold - variance) / threshold) * 100 if threshold > 0 else 0
            progress4 = max(0, min(100, progress4))
        else:
            variance = 0
            threshold = self.stopping_criteria.convergence_threshold ** 2
            progress4 = 0

        self.stop_text.insert(tk.END, f"4️⃣ Convergence Check\n")
        self.stop_text.insert(tk.END, f"   Variance: {variance:.6f} (Threshold: {threshold:.6f})\n")
        self.stop_text.insert(tk.END, f"   Recent scores: {[f'{s:.4f}' for s in recent[-5:]]}\n")
        self.stop_text.insert(tk.END, f"   {'█' * int(progress4/5)}{'░' * (20-int(progress4/5))}\n\n")

        should_stop, reasons = self.stopping_criteria.should_stop()

        self.stop_text.insert(tk.END, f"{'='*45}\n")
        if should_stop:
            self.stop_text.insert(tk.END, "✓ Stopping criteria met\n\n")

            for reason in reasons:
                self.stop_text.insert(tk.END, f"  {reason}\n")
        else:
            self.stop_text.insert(tk.END, "▶ Continue optimization\n\n")

            self.stop_text.insert(tk.END, "  Stopping criteria not met\n")

        self.stop_text.insert(tk.END, f"{'='*45}\n")

    def show_all_visualizations(self):
        if len(self.experiment_data) < 2:
            messagebox.showwarning("Warning", "At least 2 data points are needed to generate charts")
            return

        try:
            self.status_var.set("Generating charts...")
            self.root.update()

            all_experiments = self.experiment_data.get_all_experiments()

            fig1 = self.viz_manager.plot_convergence_curve(all_experiments)
            if fig1:
                self.viz_manager.save_figure(fig1, "01_convergence_curve")
                plt.close(fig1)

            fig2 = self.viz_manager.plot_analysis_time_distribution(all_experiments)
            if fig2:
                self.viz_manager.save_figure(fig2, "02_analysis_time_distribution")
                plt.close(fig2)

            fig3 = self.viz_manager.plot_parameter_sensitivity(all_experiments)
            if fig3:
                self.viz_manager.save_figure(fig3, "03_parameter_sensitivity_analysis")
                plt.close(fig3)

            # 4. High-dimensional parameter space
            fig4 = self.viz_manager.plot_high_dimensional_space(all_experiments)
            if fig4:
                self.viz_manager.save_figure(fig4, "04_high_dimensional_space")
                plt.close(fig4)

            self.status_var.set("✓ Charts generated successfully!")
            messagebox.showinfo("Success", "All charts have been saved to the visualizations folder")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate charts: {str(e)}")
            self.status_var.set("✗ Failed to generate charts")

    def export_results(self):
        """Export results to CSV"""

        all_experiments = self.experiment_data.get_all_experiments()

        if not all_experiments:
            messagebox.showwarning("Warning", "No data available for export")
            return

        df = pd.DataFrame(all_experiments)

        params_df = pd.DataFrame(
            df['params'].tolist(),
            columns=PARAM_NAMES,
            index=df.index
        )

        # Include ECRF column if it exists
        columns_to_export = ['round', 'is_initial', 'score', 'analysis_time', 'timestamp']
        
        # Check if ecrf column exists
        if 'ecrf' in df.columns:
            columns_to_export.append('ecrf')
        
        export_df = pd.concat([
            df[columns_to_export],
            params_df
        ], axis=1)

        export_df['Type'] = export_df['is_initial'].apply(lambda x: 'Initial point' if x else 'Optimized point')

        filename = os.path.join(self.results_dir, f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        export_df.to_csv(filename, index=False, encoding='utf-8-sig')

        messagebox.showinfo("Success", f"Results exported:\n{filename}")
        self.status_var.set(f"✓ Results exported")

    def reset_all(self):
        """Reset and backup historical data"""
        all_experiments = self.experiment_data.get_all_experiments()

        if all_experiments and messagebox.askyesno("Confirmation",
            "Are you sure you want to reset all data and start a new optimization?\n\n"
            f"Current historical data: {self.current_round} rounds\n"
            f"This operation cannot be undone"):

            if all_experiments:
                backup_file = os.path.join(
                    self.results_dir,
                    f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                df = pd.DataFrame(all_experiments)
                df.to_csv(backup_file, index=False, encoding='utf-8-sig')
                logger.info(f"✓ Old data backed up: {backup_file}")

            self.experiment_data.clear_data()
            self.current_round = 0
            self.initial_points_used = 0
            self.next_params = None
            self.optimizer = None
            self.current_asc_files = []
            self.has_pending_params = False
            self.optimization_stopped = False
            self.stop_reason = None
            self.generate_retry_count = 0
            self.scaler_params = None

            self.stopping_criteria = StoppingCriteria()

            self.param_text.delete(1.0, tk.END)
            self.result_text.delete(1.0, tk.END)
            self.stats_text.delete(1.0, tk.END)
            self.history_text.delete(1.0, tk.END)
            self.stop_text.delete(1.0, tk.END)
            self.crf_var.set("")

            self.show_initial_points()
            self.update_status()
            self.save_history()

            self.stop_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.submit_btn.config(state=tk.DISABLED)
            self.analyze_btn.config(state=tk.DISABLED)
            self.plot_btn.config(state=tk.DISABLED)

            self.result_text.insert(tk.END, f"\n{'='*45}\n")
            self.result_text.insert(tk.END, "✓ All data reset\n")
            self.result_text.insert(tk.END, f"{'='*45}\n\n")

            self.result_text.insert(tk.END, "Click 【🚀 Initialize Optimization】 to start a new optimization\n\n")
            self.result_text.see(tk.END)

            self.status_var.set("✓ Reset completed")
            logger.info("✓ All data reset")


def main():
    root = tk.Tk()
    app = BayesianOptimizationUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()