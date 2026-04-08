"""主窗口模块（增强版：恢复历史、健壮的生成/提交/停止流程）"""

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
from core.constraint import ConstraintHandler, get_initial_points
from core.optimizer import create_gp_regressor, StoppingCriteria, MultiFidelityOptimizer
from data.manager import ExperimentDataManager
from data.asc_parser import process_asc_files, calculate_crf_from_asc_data, calculate_signals_and_regions_from_asc_data
from visualization.plotter import VisualizationManager
from utils.file_handler import FileHandler
from utils.logger import logger


class BayesianOptimizationUI:
    """增强版主窗口：实现与“修改迭代异常终止.py”一致的行为"""

    def __init__(self, root):
        self.root = root
        self.root.title(UI_CONFIG['window_title'])
        self.root.geometry(UI_CONFIG['window_size'])

        # 路径与目录
        self.work_dir = WORK_DIR
        self.data_dir = os.path.join(self.work_dir, "data")
        self.results_dir = os.path.join(self.work_dir, "results")
        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoints")

        for d in [self.data_dir, self.results_dir, self.checkpoint_dir]:
            os.makedirs(d, exist_ok=True)

        # 模块
        self.constraint_handler = ConstraintHandler()
        self.viz_manager = VisualizationManager(self.work_dir)
        self.experiment_data = ExperimentDataManager()

        # 状态
        self.current_round = 0
        self.next_params = None
        self.optimizer = None
        self.initial_points_used = 0
        self.has_pending_params = False
        self.optimization_stopped = False
        self.stop_reason = None
        self.generate_retry_count = 0
        self.scaler_params = None

        # 停止条件
        # StoppingCriteria 从 STOPPING_CONFIG 内部读取配置，因此无需传参
        self.stopping_criteria = StoppingCriteria()

        # UI
        self.create_ui()

        # 加载并恢复历史（如果存在）
        if self.load_history():
            if len(self.experiment_data) > 0:
                self.restore_optimization_state()

        self.update_status()
        logger.info("应用已启动（增强版）")

    # ----------------------- 历史存取 -----------------------
    def load_history(self):
        """加载历史检查点（利用 FileHandler）"""
        state, exp_data = FileHandler.load_checkpoint()
        if state is None and exp_data is None:
            logger.info("未检测到历史数据")
            return False

        if exp_data:
            self.experiment_data.data = exp_data

        self.current_round = state.get('current_round', 0)
        self.initial_points_used = state.get('initial_points_used', 0)
        self.has_pending_params = state.get('has_pending_params', False)
        self.optimization_stopped = state.get('optimization_stopped', False)
        self.stop_reason = state.get('stop_reason', None)

        # 恢复 next_params（如果有待评分）
        if self.has_pending_params and self.current_round > 0:
            # 上一轮是待评分的（current_round 指向已生成但未提交的轮次）
            for exp in self.experiment_data.get_all():
                if exp['round'] == self.current_round - 1:
                    self.next_params = exp['params']
                    break

        logger.info(f"历史已加载: {self.current_round} 轮, 初始点已用: {self.initial_points_used}")
        return True

    def save_history(self):
        """保存历史检查点（使用 FileHandler）"""
        state = {
            'current_round': self.current_round,
            'initial_points_used': self.initial_points_used,
            'has_pending_params': self.has_pending_params,
            'optimization_stopped': self.optimization_stopped,
            'stop_reason': self.stop_reason,
            'timestamp': datetime.now().isoformat()
        }
        FileHandler.save_checkpoint(state, self.experiment_data.get_all())

    # ----------------------- 恢复与重建 -----------------------
    def restore_optimization_state(self):
        """从历史数据重建优化器和停止条件，并恢复 UI 状态"""
        try:
            logger.info("正在恢复优化器与停止条件...")
            self.rebuild_optimizer()
            self.rebuild_stopping_criteria()

            if self.optimizer is not None:
                self.viz_manager.set_gp_model(self.optimizer.base_estimator_)

            # 更新 UI
            self.show_history_summary()
            self.update_stats()
            self.update_history()
            self.update_stopping_criteria_display()
            self.show_initial_points()

            # 如果上次已停止，禁用交互按钮
            if self.optimization_stopped:
                self.next_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.DISABLED)
            else:
                self.next_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.NORMAL)

                if self.has_pending_params:
                    self.result_text.insert(tk.END, f"⏳ 第 {self.current_round} 轮仍有待评分参数，请提交评分\n")
                    if self.next_params is not None:
                        self.display_next_params(self.next_params)
                    self.analyze_btn.config(state=tk.NORMAL)
                    self.submit_btn.config(state=tk.NORMAL)
                else:
                    self.result_text.insert(tk.END, f"✓ 已完成 {self.current_round} 轮，可继续优化\n")

            logger.info("历史恢复完成")
        except Exception as e:
            logger.error(f"恢复历史失败: {e}")
            self.result_text.insert(tk.END, f"❌ 恢复失败: {e}\n")

    def rebuild_optimizer(self):
        """用历史样本重建 Skopt Optimizer 并可选拟合 GP"""
        try:
            # 使用多保真度优化器并用历史数据填充
            self.optimizer = MultiFidelityOptimizer(dimensions=PARAM_DIMENSIONS)

            for exp in self.experiment_data.get_all():
                # 历史数据若包含 'fidelity' 字段则使用，否则默认 LF
                fidelity = exp.get('fidelity', 'lf')
                # 记录时使用复合评分（若存在），否则使用 score
                score = exp.get('composite_score', exp.get('score', 0.0))
                try:
                    self.optimizer.tell(exp['params'], score, fidelity=fidelity)
                except Exception:
                    # 若直接 tell 失败，尝试作为 LF
                    try:
                        self.optimizer.tell(exp['params'], score, fidelity='lf')
                    except Exception:
                        pass

            # 若样本足够，也可尝试对内部 GP 做一次拟合（由 MultiFidelityOptimizer 自动管理）
            logger.info("多保真度优化器已基于历史样本重建")

        except Exception as e:
            logger.error(f"重建优化器失败: {e}")
            raise

    def rebuild_stopping_criteria(self):
        """用历史评分重建停止条件状态"""
        try:
            # StoppingCriteria 使用 STOPPING_CONFIG 内部配置，无需显式传参
            self.stopping_criteria = StoppingCriteria()

            for exp in self.experiment_data.get_all():
                self.stopping_criteria.update(exp['score'], exp['params'])

            logger.info("停止条件已重建")
        except Exception as e:
            logger.error(f"重建停止条件失败: {e}")
            raise

    def show_history_summary(self):
        """显示历史数据摘要到结果面板"""
        try:
            self.result_text.delete(1.0, tk.END)

            all_experiments = self.experiment_data.get_all()
            if not all_experiments:
                self.result_text.insert(tk.END, "暂无历史数据\n")
                return

            df = pd.DataFrame(all_experiments)

            self.result_text.insert(tk.END, f"\n{'='*50}\n")
            self.result_text.insert(tk.END, f"📖 历史优化数据摘要\n")
            self.result_text.insert(tk.END, f"{'='*50}\n\n")

            self.result_text.insert(tk.END, f"总轮次: {len(df)}\n")

            initial_df = df[df['is_initial']]
            optimized_df = df[~df['is_initial']]

            self.result_text.insert(tk.END, f"  • 初始点: {len(initial_df)}/{MAX_INITIAL_POINTS}\n")
            self.result_text.insert(tk.END, f"  • 优化点: {len(optimized_df)}\n\n")

            self.result_text.insert(tk.END, f"CRF值统计:\n")
            self.result_text.insert(tk.END, f"  • 最高: {df['score'].max():.4f} (第{df['score'].idxmax()+1}轮)\n")
            self.result_text.insert(tk.END, f"  • 平均: {df['score'].mean():.4f}\n")
            self.result_text.insert(tk.END, f"  • 最低: {df['score'].min():.4f}\n")
            self.result_text.insert(tk.END, f"  • 标准差: {df['score'].std():.4f}\n\n")

            self.result_text.insert(tk.END, f"分析时间:\n")
            self.result_text.insert(tk.END, f"  • 平均: {df['analysis_time'].mean():.1f}s\n")
            self.result_text.insert(tk.END, f"  • 最长: {df['analysis_time'].max():.1f}s\n\n")

            self.result_text.insert(tk.END, f"{'='*50}\n")
            self.result_text.insert(tk.END, f"最近 5 轮:\n")
            self.result_text.insert(tk.END, f"{'='*50}\n")

            for idx, exp in enumerate(all_experiments[-5:], start=max(1, len(all_experiments)-4)):
                point_type = "初始点" if exp['is_initial'] else "优化点"
                self.result_text.insert(tk.END, 
                    f"第{idx}轮 ({point_type}): CRF={exp['score']:.4f}, 时间={exp['analysis_time']:.1f}s\n")

            self.result_text.insert(tk.END, f"\n{'='*50}\n\n")

        except Exception as e:
            logger.error(f"显示历史摘要失败: {e}")
            self.result_text.insert(tk.END, f"❌ 显示历史摘要失败: {e}\n")

    def compute_composite_score(self, crf_value, analysis_time):
        """计算复合评分：在 CRF 基础上加入分析时间惩罚。

        采用配置：ANALYSIS_TIME_TARGET, ANALYSIS_TIME_WEIGHT, ANALYSIS_TIME_PENALTY
        - penalty (squared): ((t - T)/T)**2
        - penalty (absolute): abs((t - T)/T)
        composite = crf - weight * penalty
        返回复合评分（float）
        """
        try:
            T = ANALYSIS_TIME_TARGET
            w = ANALYSIS_TIME_WEIGHT
            ptype = ANALYSIS_TIME_PENALTY

            rel = (analysis_time - T) / float(T)
            if ptype == 'squared':
                penalty = rel ** 2
            else:
                penalty = abs(rel)

            composite = float(crf_value) - float(w) * float(penalty)
            # 保证数值稳定性
            return composite
        except Exception as e:
            logger.warning(f"计算复合分数失败: {e}")
            return float(crf_value)

    # ----------------------- UI 创建 -----------------------
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
        top_frame = ttk.LabelFrame(parent, text="🎯 控制面板", padding=10)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(top_frame)
        row1.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(row1, text="📂 选择ASC文件", command=self.select_asc_files).pack(side=tk.LEFT, padx=5)

        ttk.Button(row1, text="📊 分析ASC文件", command=self.analyze_asc_files, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        self.analyze_btn = row1.winfo_children()[-1]

        ttk.Button(row1, text="🚀 初始化优化", command=self.initialize_optimization).pack(side=tk.LEFT, padx=5)

        ttk.Button(row1, text="⚡ 生成下一参数", command=self.generate_next_params, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        self.next_btn = row1.winfo_children()[-1]

        ttk.Button(row1, text="⛔ 停止优化", command=self.stop_optimization, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        self.stop_btn = row1.winfo_children()[-1]

        ttk.Button(row1, text="📈 生成所有图表", command=self.show_all_visualizations, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        self.plot_btn = row1.winfo_children()[-1]

        ttk.Button(row1, text="🔄 重置", command=self.reset_all).pack(side=tk.LEFT, padx=5)

        # 第二行：评分输入与提交
        row2 = ttk.Frame(top_frame)
        row2.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(row2, text="📝 CRF值 (0.0-1.0) 评分当前参数:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        self.crf_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.crf_var, width=10).pack(side=tk.LEFT, padx=5)

        ttk.Button(row2, text="✓ 提交评分→生成下一参数", command=self.submit_crf_and_generate_next).pack(side=tk.LEFT, padx=5)
        self.submit_btn = row2.winfo_children()[-1]
        self.submit_btn.config(state=tk.DISABLED)

        ttk.Label(row2, text="  | 进度:").pack(side=tk.LEFT, padx=10)
        self.progress_var = tk.StringVar(value="未开始")
        ttk.Label(row2, textvariable=self.progress_var, foreground='blue', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=2)

        ttk.Button(row2, text="💾 导出结果", command=self.export_results).pack(side=tk.RIGHT, padx=5)

    def create_left_panel(self, parent):
        left_frame = ttk.LabelFrame(parent, text="📋 当前参数 / 参数列表", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        text_frame = ttk.Frame(left_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.param_text = tk.Text(text_frame, height=30, width=45, yscrollcommand=scrollbar.set, font=('Courier', 10))
        self.param_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.param_text.yview)

    def create_right_panel(self, parent):
        right_frame = ttk.LabelFrame(parent, text="📊 优化结果", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # 日志
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="📝 优化记录")
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text = tk.Text(log_frame, height=30, width=45, yscrollcommand=scrollbar.set, font=('Courier', 9))
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.result_text.yview)

        # 统计
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="📈 统计信息")
        self.stats_text = tk.Text(stats_frame, height=30, width=45, font=('Courier', 9))
        scrollbar2 = ttk.Scrollbar(stats_frame, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=scrollbar2.set)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)

        # 历史
        history_frame = ttk.Frame(notebook)
        notebook.add(history_frame, text="📊 参数历史")
        scrollbar3 = ttk.Scrollbar(history_frame)
        scrollbar3.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_text = tk.Text(history_frame, height=30, width=45, yscrollcommand=scrollbar3.set, font=('Courier', 8))
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar3.config(command=self.history_text.yview)

        # 停止条件
        stop_frame = ttk.Frame(notebook)
        notebook.add(stop_frame, text="⛔ 停止条件")
        scrollbar4 = ttk.Scrollbar(stop_frame)
        scrollbar4.pack(side=tk.RIGHT, fill=tk.Y)
        self.stop_text = tk.Text(stop_frame, height=30, width=45, yscrollcommand=scrollbar4.set, font=('Courier', 9))
        self.stop_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar4.config(command=self.stop_text.yview)

    def create_status_bar(self, parent):
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.pack(fill=tk.X, side=tk.LEFT, expand=True)

        self.round_var = tk.StringVar(value="轮次: 0")
        round_label = ttk.Label(status_frame, textvariable=self.round_var, relief=tk.SUNKEN, width=15)
        round_label.pack(side=tk.LEFT, padx=(5, 0))

        self.best_var = tk.StringVar(value="最佳: --")
        best_label = ttk.Label(status_frame, textvariable=self.best_var, relief=tk.SUNKEN, width=15)
        best_label.pack(side=tk.LEFT, padx=(5, 0))

        self.opt_status_var = tk.StringVar(value="状态: 未运行")
        opt_label = ttk.Label(status_frame, textvariable=self.opt_status_var, relief=tk.SUNKEN, width=20)
        opt_label.pack(side=tk.LEFT, padx=(5, 0))

    # ----------------------- 状态 / 显示 -----------------------
    def update_status(self):
        self.round_var.set(f"轮次: {self.current_round}")

        if self.has_pending_params:
            self.progress_var.set(f"⏳ 等待评分 (第{self.current_round}轮)")
        elif self.optimizer is not None:
            self.progress_var.set(f"✓ 已完成 {self.current_round} 轮")
        else:
            self.progress_var.set("未开始")

        all_experiments = self.experiment_data.get_all()
        if all_experiments:
            best_score = max([e['score'] for e in all_experiments])
            self.best_var.set(f"最佳: {best_score:.4f}")
        else:
            self.best_var.set("最佳: --")

        if self.optimization_stopped:
            self.opt_status_var.set("状态: 已停止 ✓")
        else:
            self.opt_status_var.set("状态: 运行中 ▶")

    def show_initial_points(self):
        self.param_text.delete(1.0, tk.END)

        initial_points = get_initial_points()

        self.param_text.insert(tk.END, "="*50 + "\n")
        self.param_text.insert(tk.END, "   初始实验点设计 (共10个)\n")
        self.param_text.insert(tk.END, "="*50 + "\n\n")

        for idx, point in enumerate(initial_points):
            analysis_time = self.constraint_handler.calculate_analysis_time(point)
            if idx < self.initial_points_used:
                used_marker = "✓ "
                status = "已完成"
            elif idx == self.initial_points_used and self.has_pending_params:
                used_marker = "⏳ "
                status = "待评分"
            else:
                used_marker = "  "
                status = "待执行"

            self.param_text.insert(tk.END, f"{used_marker}【初始点 #{idx+1}】({status})\n")

            for i, (name, value) in enumerate(zip(PARAM_NAMES, point)):
                unit = PARAM_UNITS[name]
                # 确保value是标量值，处理numpy数组情况
                if hasattr(value, '__len__') and hasattr(value, '__getitem__'):
                    # 如果是numpy标量，使用.item()方法
                    if hasattr(value, 'item'):
                        scalar_value = value.item()
                    elif len(value) == 1:
                        scalar_value = float(value[0])
                    else:
                        scalar_value = float(value[0])  # 取第一个元素
                else:
                    scalar_value = float(value)
                self.param_text.insert(tk.END, f"  {name:<12}: {scalar_value:>8.2f} {unit}\n")

            self.param_text.insert(tk.END, f"  分析时间: {analysis_time:.1f}s\n")
            self.param_text.insert(tk.END, "-"*50 + "\n\n")

    # ----------------------- 文件选择 / 分析 -----------------------
    def select_asc_files(self):
        files = filedialog.askopenfilenames(
            title="选择ASC数据文件",
            filetypes=[("ASC files", "*.asc"), ("All files", "*.*")]
        )

        if files:
            self.current_asc_files = list(files)
            self.status_var.set(f"✓ 已选择 {len(files)} 个ASC文件")
            self.analyze_btn.config(state=tk.NORMAL)

    def analyze_asc_files(self):
        if not self.current_asc_files:
            messagebox.showwarning("警告", "请先选择ASC文件")
            return

        self.status_var.set("正在分析ASC文件...")
        self.root.update()

        try:
            df = process_asc_files(self.current_asc_files, use_half=True)

            if df is not None:
                # 保存ASC数据，用于新ECRF计算
                self._current_asc_df = df
                crf_value = calculate_crf_from_asc_data(df)

                self.result_text.insert(tk.END, f"\n{'='*45}\n")
                self.result_text.insert(tk.END, f"ASC文件分析完成 \n")
                self.result_text.insert(tk.END, f"文件数量: {len(self.current_asc_files)}\n")
                self.result_text.insert(tk.END, f"自动计算 CRF值: {crf_value:.4f}\n")
                self.result_text.insert(tk.END, f"{'='*45}\n\n")
                self.result_text.see(tk.END)

                self.crf_var.set(f"{crf_value:.4f}")

                self.status_var.set(f"✓ ASC分析完成，CRF: {crf_value:.4f}")

                if self.has_pending_params:
                    self.submit_btn.config(state=tk.NORMAL)
            else:
                messagebox.showerror("错误", "处理ASC文件失败")
                self.status_var.set("✗ 文件处理失败")

        except Exception as e:
            messagebox.showerror("错误", f"分析失败: {str(e)}")
            self.status_var.set(f"✗ 分析失败: {str(e)}")

    # ----------------------- 初始化 / 生成参数 -----------------------
    def initialize_optimization(self):
        all_experiments = self.experiment_data.get_all()

        if all_experiments:
            choice = messagebox.askyesno(
                "确认",
                f"检测到历史优化数据 ({self.current_round} 轮)\n\n"
                f"选择:\n"
                f"【是】 继续进行历史优化\n"
                f"【否】 重新开始新的优化"
            )

            if choice:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"✓ 继续历史优化...\n\n")
                self.show_history_summary()
                self.result_text.see(tk.END)
                self.next_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.NORMAL)

                self.update_stats()
                self.update_history()
                self.update_stopping_criteria_display()
                self.show_initial_points()

                self.status_var.set("✓ 继续历史优化")

                if self.current_round >= 2:
                    self.plot_btn.config(state=tk.NORMAL)

                if self.has_pending_params:
                    self.result_text.insert(tk.END, 
                        f"⏳ 第 {self.current_round} 轮仍有待评分的参数\n")
                    self.result_text.insert(tk.END, f"请在上方输入CRF值并提交\n\n")
                    self.result_text.see(tk.END)

                    if self.next_params is not None:
                        self.display_next_params(self.next_params)

                    self.analyze_btn.config(state=tk.NORMAL)
                    self.submit_btn.config(state=tk.NORMAL)
                else:
                    self.result_text.insert(tk.END, 
                        f"✓ 已完成 {self.current_round} 轮\n")
                    self.result_text.insert(tk.END, f"点击【⚡生成下一参数】继续优化\n\n")
                    self.result_text.see(tk.END)
                    self.show_initial_points()

                return
            else:
                self.reset_all()
                return

        if self.optimization_stopped:
            messagebox.showwarning("警告", "请先重置才能开始新的优化")
            return

        self.initial_points_used = 0
        self.has_pending_params = False

        initial_points = get_initial_points()
        initial_points = [self.constraint_handler.project_to_feasible(p) for p in initial_points]

        # 使用多保真度优化器（AR-CoK）替代原有 Skopt Optimizer
        self.optimizer = MultiFidelityOptimizer(dimensions=PARAM_DIMENSIONS)

        for point in initial_points:
            # 初始点作为低保真度供热启动使用
            self.optimizer.tell(point, 0.0, fidelity='lf')

        self.viz_manager.set_gp_model(self.optimizer.base_estimator_)

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"\n{'='*45}\n")
        self.result_text.insert(tk.END, f"🚀 新的贝叶斯优化已初始化\n")
        self.result_text.insert(tk.END, f"{'='*45}\n")
        self.result_text.insert(tk.END, f"初始点数: {MAX_INITIAL_POINTS}\n")
        self.result_text.insert(tk.END, f"分析时间范围: {MIN_ANALYSIS_TIME}-{MAX_ANALYSIS_TIME}秒\n")
        self.result_text.insert(tk.END, f"{'='*45}\n\n")
        self.result_text.insert(tk.END, f"💡 接下来请点击【⚡生成下一参数】来开始优化\n\n")
        self.result_text.see(tk.END)

        self.status_var.set("✓ 优化器已初始化")
        self.next_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.show_initial_points()
        self.update_status()

    def generate_next_params(self):
        if self.optimizer is None:
            messagebox.showwarning("警告", "请先初始化优化器")
            return

        if self.optimization_stopped:
            messagebox.showinfo("提示", "优化已停止，请重置后重新开始")
            return

        if self.has_pending_params:
            messagebox.showwarning("警告", "有待评分的参数，请先提交评分")
            return

        self.generate_retry_count = 0
        self._generate_next_params_impl()

    def _generate_next_params_impl(self):
        try:
            if self.generate_retry_count >= GENERATE_RETRY_LIMIT:
                messagebox.showerror(
                    "错误",
                    f"⚠️ 连续{GENERATE_RETRY_LIMIT}次无法生成满足约束的参数\n"
                    f"建议调整参数约束条件"
                )
                return

            if self.initial_points_used < MAX_INITIAL_POINTS:
                initial_points = get_initial_points()
                next_point = initial_points[self.initial_points_used]
                source = "初始点库"
                self.next_params_fidelity = 'lf'
            else:
                try:
                    next_point = self.optimizer.ask()
                    source = "优化器"
                    # 记录优化器建议的保真度（'lf' 或 'hf'）以便提示用户
                    self.next_params_fidelity = getattr(self.optimizer, 'last_suggested_fidelity', 'lf')
                except Exception as e:
                    logger.warning(f"优化器生成失败: {e}，使用启发式方法")
                    next_point = self.constraint_handler.generate_feasible_heuristic()
                    source = "启发式方法"
                    self.next_params_fidelity = 'lf'

            self.next_params = self.constraint_handler.project_to_feasible(next_point)

            is_feasible, msg = self.constraint_handler.is_feasible(self.next_params)

            if not is_feasible:
                self.generate_retry_count += 1
                self.result_text.insert(tk.END, f"\n⚠️ 第 {self.generate_retry_count} 次尝试：生成的参数不满足约束: {msg}，正在重新生成...\n")
                self.result_text.see(tk.END)
                self.root.update()
                self._generate_next_params_impl()
                return

            self.has_pending_params = True
            self.current_round += 1
            self.generate_retry_count = 0

            is_initial = self.initial_points_used < MAX_INITIAL_POINTS

            if is_initial:
                self.show_initial_points()
            else:
                self.display_next_params(self.next_params)

            self.submit_btn.config(state=tk.NORMAL)

            point_type = "初始点" if is_initial else "优化点"
            analysis_time = self.constraint_handler.calculate_analysis_time(self.next_params)

            self.result_text.insert(tk.END, f"\n✓ 第 {self.current_round} 轮参数已生成 ({point_type}, 来源: {source})\n")
            self.result_text.insert(tk.END, f"分析时间: {analysis_time:.1f}s (范围: {MIN_ANALYSIS_TIME}-{MAX_ANALYSIS_TIME}s)\n")
            self.result_text.insert(tk.END, f"⏳ 请运行实验并输入CRF值\n\n")
            if getattr(self, 'next_params_fidelity', 'lf') == 'hf':
                self.result_text.insert(tk.END, f"⏳ 建议高保真度 (HF)：请对该参数组合运行 {self.optimizer.hf_repeat} 次 GC 实验，并提交平均 CRF 值\n\n")
            else:
                self.result_text.insert(tk.END, f"⏳ 请运行实验并输入CRF值\n\n")
            self.result_text.see(tk.END)

            self.update_status()
            self.status_var.set(f"✓ 第 {self.current_round} 轮参数已生成")

        except Exception as e:
            messagebox.showerror("错误", f"生成参数失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"✗ 生成参数失败: {str(e)}")

    def submit_crf_and_generate_next(self):
        if not self.has_pending_params:
            messagebox.showwarning("警告", "没有待评分的参数")
            return

        if self.next_params is None:
            messagebox.showwarning("警告", "参数丢失，正在重新生成...")
            self.has_pending_params = False
            self.crf_var.set("")
            self.submit_btn.config(state=tk.DISABLED)
            self.generate_next_params()
            return

        try:
            crf_str = self.crf_var.get().strip()
            if not crf_str:
                messagebox.showwarning("警告", "请输入CRF值")
                return

            try:
                crf_value = float(crf_str)
            except ValueError:
                messagebox.showerror("错误", "CRF值必须是数字")
                return

            if not (0 <= crf_value <= 1.0):
                messagebox.showerror("错误", "CRF值必须在0.0-1.0之间")
                return

            analysis_time = self.constraint_handler.calculate_analysis_time(self.next_params)

            is_feasible, msg = self.constraint_handler.is_feasible(self.next_params)
            if not is_feasible:
                messagebox.showerror("错误", f"参数不满足约束: {msg}")
                return

            test_exp = {
                'round': self.current_round,
                'params': list(self.next_params),
                'score': crf_value,
                'analysis_time': analysis_time,
                'timestamp': datetime.now().isoformat(),
                'is_initial': self.initial_points_used < MAX_INITIAL_POINTS
            }

            if len(test_exp['params']) != len(PARAM_NAMES):
                raise ValueError(
                    f"参数长度不匹配: {len(test_exp['params'])} != {len(PARAM_NAMES)}"
                )

            # 计算复合分数并保存到记录（用于 GP 拟合与优化目标）
            composite_score = self.compute_composite_score(crf_value, analysis_time)
            test_exp['composite_score'] = composite_score
            
            # 计算新ECRF，无论是否有ASC数据
            # 获取当前迭代次数
            status = self.stopping_criteria.get_status()
            current_iteration = status.get('iteration', 1)
            total_iterations = self.stopping_criteria.max_iterations
            
            # 计算时间函数 f_time
            # 假设合理的T_min和T_max值
            t_min = 80  # 可以根据实际情况调整
            t_max = 300 # 可以根据实际情况调整
            f_time = 1 - (analysis_time - t_min) / (t_max - t_min)
            f_time = np.clip(f_time, 0.0, 1.0)  # 限制在[0,1]范围内
            
            # 根据是否有ASC数据决定f_sep的计算方式
            if hasattr(self, '_current_asc_df') and self._current_asc_df is not None:
                from core.signal_process import calculate_new_ecrf
                signal_data, regions = calculate_signals_and_regions_from_asc_data(self._current_asc_df)
                if signal_data is not None and len(regions) > 0:
                    # 有ASC数据时，使用完整的新ECRF计算
                    try:
                        new_ecrf = calculate_new_ecrf(
                            signal=signal_data,
                            regions=regions,
                            analysis_time=analysis_time,
                            iteration_num=current_iteration,
                            total_iterations=total_iterations
                        )
                        test_exp['ecrf'] = new_ecrf
                    except Exception as e:
                        logger.warning(f"计算完整新ECRF失败: {e}，使用简化计算")
                        # 使用手动输入的CRF值作为f_sep
                        f_sep = crf_value  # 使用用户输入的CRF值作为分离质量部分
                        w_sep = 0.7 + 0.3 * (1 - current_iteration / total_iterations)
                        w_time = 1 - w_sep
                        simplified_ecrf = w_sep * f_sep + w_time * f_time
                        test_exp['ecrf'] = simplified_ecrf
                else:
                    # 没有有效信号数据时，使用手动输入的CRF值
                    f_sep = crf_value
                    w_sep = 0.7 + 0.3 * (1 - current_iteration / total_iterations)
                    w_time = 1 - w_sep
                    simplified_ecrf = w_sep * f_sep + w_time * f_time
                    test_exp['ecrf'] = simplified_ecrf
            else:
                # 没有ASC数据时，使用手动输入的CRF值作为f_sep
                f_sep = crf_value  # 使用用户输入的CRF值作为分离质量部分
                w_sep = 0.7 + 0.3 * (1 - current_iteration / total_iterations)
                w_time = 1 - w_sep
                simplified_ecrf = w_sep * f_sep + w_time * f_time
                test_exp['ecrf'] = simplified_ecrf
            self.experiment_data.append(test_exp)

            try:
                # 优先使用新ECRF作为优化目标，否则使用传统复合评分
                if test_exp.get('ecrf', 0.0) != 0.0:
                    # 使用新ECRF作为优化目标
                    optimization_target = test_exp['ecrf']
                else:
                    # 没有新ECRF时使用传统复合评分
                    optimization_target = composite_score
                
                # 将优化目标作为优化器反馈（负号因为 skopt 默认最小化）
                self.optimizer.tell(self.next_params, -optimization_target)
                # 停止条件基于所选优化目标判断
                self.stopping_criteria.update(optimization_target, self.next_params)
            except Exception as e:
                logger.warning(f"优化器更新警告: {e}")

            if test_exp['is_initial']:
                self.initial_points_used += 1

            if len(self.experiment_data) >= 5:
                try:
                    all_experiments = self.experiment_data.get_all()
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
                    logger.warning(f"GP模型更新警告: {e}")

            self.has_pending_params = False
            self.next_params = None
            self.crf_var.set("")
            self.submit_btn.config(state=tk.DISABLED)

            self.update_stats()
            self.update_history()
            self.update_stopping_criteria_display()
            self.save_history()

            point_type = "【初始点】" if test_exp['is_initial'] else "【优化点】"
            self.result_text.insert(tk.END, f"\n{point_type} 第 {self.current_round} 轮 - 评分完成\n")
            self.result_text.insert(tk.END, f"CRF值: {crf_value:.4f}\n")
            self.result_text.insert(tk.END, f"分析时间: {analysis_time:.1f}s ✓\n\n")
            self.result_text.see(tk.END)

            self.update_status()

            if self.current_round >= 2:
                self.plot_btn.config(state=tk.NORMAL)

            should_stop, reasons = self.stopping_criteria.should_stop()
            if should_stop:
                self.stop_optimization(reasons)
                return

            self.result_text.insert(tk.END, f"→ 自动生成第 {self.current_round + 1} 轮参数...\n\n")
            self.result_text.see(tk.END)
            self.root.update()

            self.generate_next_params()

        except Exception as e:
            messagebox.showerror("错误", f"提交评分失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"✗ 提交评分失败: {str(e)}")

    def stop_optimization(self, reasons=None):
        self.optimization_stopped = True
        self.stop_reason = " | ".join(reasons) if reasons else "用户手动停止"

        self.stop_btn.config(state=tk.DISABLED)
        self.next_btn.config(state=tk.DISABLED)
        self.submit_btn.config(state=tk.DISABLED)

        if reasons is None:
            reasons = ["用户手动停止"]

        self.result_text.insert(tk.END, f"\n{'='*45}\n")
        self.result_text.insert(tk.END, "⛔ 优化已停止\n")
        self.result_text.insert(tk.END, f"{'='*45}\n")
        self.result_text.insert(tk.END, "停止原因:\n")
        for reason in reasons:
            self.result_text.insert(tk.END, f"  • {reason}\n")
        self.result_text.insert(tk.END, f"{'='*45}\n\n")
        self.result_text.see(tk.END)

        self.save_history()

        self.status_var.set("⛔ 优化已停止")
        self.update_status()

        messagebox.showinfo("优化停止", f"优化已停止\n\n停止原因:\n" + "\n".join(reasons))

    def display_next_params(self, params):
        self.param_text.delete(1.0, tk.END)

        is_feasible, warnings_msg = self.constraint_handler.is_feasible(params)

        self.param_text.insert(tk.END, "="*50 + "\n")
        self.param_text.insert(tk.END, f"第 {self.current_round} 轮实验参数 (【优化点】)\n")
        self.param_text.insert(tk.END, "【待执行】\n")
        self.param_text.insert(tk.END, "="*50 + "\n\n")

        for i, (name, value) in enumerate(zip(PARAM_NAMES, params)):
            unit = PARAM_UNITS[name]
            dim = PARAM_DIMENSIONS[i]
            # 确保value是标量值，处理numpy数组情况
            if hasattr(value, '__len__') and hasattr(value, '__getitem__'):
                # 如果是numpy标量，使用.item()方法
                if hasattr(value, 'item'):
                    scalar_value = value.item()
                elif len(value) == 1:
                    scalar_value = float(value[0])
                else:
                    scalar_value = float(value[0])  # 取第一个元素
            else:
                scalar_value = float(value)
            self.param_text.insert(tk.END, f"{name:<12}: {scalar_value:>8.2f} {unit:6} [{dim.low:6.1f}, {dim.high:6.1f}]\n")

        analysis_time = self.constraint_handler.calculate_analysis_time(params)
        self.param_text.insert(tk.END, f"\n{'分析时间':<12}: {analysis_time:>8.1f} s")

        if analysis_time > MAX_ANALYSIS_TIME:
            self.param_text.insert(tk.END, f" ❌ (超过限制 {MAX_ANALYSIS_TIME}s)\n")
        elif analysis_time < MIN_ANALYSIS_TIME:
            self.param_text.insert(tk.END, f" ⚠️ (低于最小 {MIN_ANALYSIS_TIME}s)\n")
        else:
            margin = min(analysis_time - MIN_ANALYSIS_TIME, MAX_ANALYSIS_TIME - analysis_time)
            self.param_text.insert(tk.END, f" ✓ (裕度: ±{margin:.1f}s)\n")

        if is_feasible:
            self.param_text.insert(tk.END, "\n✅ 参数校验通过\n")
        else:
            self.param_text.insert(tk.END, "\n⚠️ 参数校验警告:\n")
            self.param_text.insert(tk.END, f"  {warnings_msg}\n")

        self.param_text.insert(tk.END, "\n" + "="*50 + "\n")
        self.param_text.insert(tk.END, "✅ 下一步:\n")
        self.param_text.insert(tk.END, "1. 在GC仪器上运行上述参数\n")
        self.param_text.insert(tk.END, "2. 在上方输入框输入实验的CRF值\n")
        self.param_text.insert(tk.END, "3. 点击【提交评分→生成下一参数】按钮\n")
        self.param_text.insert(tk.END, "="*50 + "\n")

        self.param_text.see(1.0)

    def update_stats(self):
        self.stats_text.delete(1.0, tk.END)

        all_experiments = self.experiment_data.get_all()

        if not all_experiments:
            self.stats_text.insert(tk.END, "暂无数据\n")
            return

        df = pd.DataFrame(all_experiments)

        self.stats_text.insert(tk.END, "="*45 + "\n")
        self.stats_text.insert(tk.END, "优化统计信息\n")
        self.stats_text.insert(tk.END, "="*45 + "\n\n")

        initial_df = df[df['is_initial']]
        optimized_df = df[~df['is_initial']]

        self.stats_text.insert(tk.END, f"总轮次: {len(df)}\n")
        self.stats_text.insert(tk.END, f"  初始点: {len(initial_df)}/{MAX_INITIAL_POINTS}\n")
        self.stats_text.insert(tk.END, f"  优化点: {len(optimized_df)}\n\n")

        self.stats_text.insert(tk.END, f"平均CRF: {df['score'].mean():.4f}\n")
        self.stats_text.insert(tk.END, f"最高CRF: {df['score'].max():.4f} (第{df['score'].idxmax()+1}轮)\n")
        self.stats_text.insert(tk.END, f"最低CRF: {df['score'].min():.4f}\n")
        self.stats_text.insert(tk.END, f"标准差: {df['score'].std():.4f}\n\n")

        self.stats_text.insert(tk.END, "分析时间统计:\n")
        self.stats_text.insert(tk.END, f"  平均: {df['analysis_time'].mean():.1f}s\n")
        self.stats_text.insert(tk.END, f"  最短: {df['analysis_time'].min():.1f}s\n")
        self.stats_text.insert(tk.END, f"  最长: {df['analysis_time'].max():.1f}s\n\n")

        best_idx = df['score'].idxmax()
        best_row = df.loc[best_idx]

        self.stats_text.insert(tk.END, "="*45 + "\n")
        self.stats_text.insert(tk.END, f"最佳配置 (第 {best_idx+1} 轮)\n")
        self.stats_text.insert(tk.END, "="*45 + "\n")
        self.stats_text.insert(tk.END, f"CRF值: {best_row['score']:.4f}\n")
        self.stats_text.insert(tk.END, f"分析时间: {best_row['analysis_time']:.1f}s\n\n")

        for i, (name, value) in enumerate(zip(PARAM_NAMES, best_row['params'])):
            unit = PARAM_UNITS[name]
            # 确保value是标量值，处理numpy数组情况
            if hasattr(value, '__len__') and hasattr(value, '__getitem__'):
                # 如果是numpy标量，使用.item()方法
                if hasattr(value, 'item'):
                    scalar_value = value.item()
                elif len(value) == 1:
                    scalar_value = float(value[0])
                else:
                    scalar_value = float(value[0])  # 取第一个元素
            else:
                scalar_value = float(value)
            self.stats_text.insert(tk.END, f"{name}: {scalar_value:.2f} {unit}\n")

    def update_history(self):
        self.history_text.delete(1.0, tk.END)

        all_experiments = self.experiment_data.get_all()

        if not all_experiments:
            self.history_text.insert(tk.END, "暂无数据\n")
            return

        self.history_text.insert(tk.END, f"{'轮':<4} {'类型':<6} {'CRF':<8} {'T_init':<8} {'R1':<8} {'T1':<8} {'T2':<8} {'时间':<8}\n")
        self.history_text.insert(tk.END, "-"*60 + "\n")

        for exp in all_experiments:
            params = exp['params']
            point_type = "初始" if exp['is_initial'] else "优化"
            
            # 确保参数值是标量，处理numpy数组情况
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
        self.stop_text.insert(tk.END, "优化停止条件\n")
        self.stop_text.insert(tk.END, "="*45 + "\n\n")

        max_iter = self.stopping_criteria.max_iterations
        current_iter = status['iteration']
        progress1 = (current_iter / max_iter) * 100 if max_iter > 0 else 0
        self.stop_text.insert(tk.END, f"1️⃣ 最大迭代次数\n")
        self.stop_text.insert(tk.END, f"   {current_iter}/{max_iter} ({progress1:.1f}%)\n")
        self.stop_text.insert(tk.END, f"   {'█' * int(progress1/5)}{'░' * (20-int(progress1/5))}\n\n")

        max_no_improve = self.stopping_criteria.max_no_improvement
        no_improve = status['no_improvement_count']
        progress2 = (no_improve / max_no_improve) * 100 if max_no_improve > 0 else 0
        self.stop_text.insert(tk.END, f"2️⃣ 无改进次数\n")
        self.stop_text.insert(tk.END, f"   {no_improve}/{max_no_improve} ({progress2:.1f}%)\n")
        self.stop_text.insert(tk.END, f"   {'█' * int(progress2/5)}{'░' * (20-int(progress2/5))}\n\n")

        best_score = status['best_score']
        target = self.stopping_criteria.target_score
        progress3 = (best_score / target) * 100 if target > 0 else 0
        self.stop_text.insert(tk.END, f"3️⃣ 目标评分\n")
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

        self.stop_text.insert(tk.END, f"4️⃣ 收敛判定\n")
        self.stop_text.insert(tk.END, f"   方差: {variance:.6f} (阈值: {threshold:.6f})\n")
        self.stop_text.insert(tk.END, f"   近期得分: {[f'{s:.4f}' for s in recent[-5:]]}\n")
        self.stop_text.insert(tk.END, f"   {'█' * int(progress4/5)}{'░' * (20-int(progress4/5))}\n\n")

        should_stop, reasons = self.stopping_criteria.should_stop()

        self.stop_text.insert(tk.END, f"{'='*45}\n")
        if should_stop:
            self.stop_text.insert(tk.END, "✓ 满足停止条件\n\n")
            for reason in reasons:
                self.stop_text.insert(tk.END, f"  {reason}\n")
        else:
            self.stop_text.insert(tk.END, "▶ 继续优化\n\n")
            self.stop_text.insert(tk.END, "  暂未达到停止条件\n")

        self.stop_text.insert(tk.END, f"{'='*45}\n")

    def show_all_visualizations(self):
        if len(self.experiment_data) < 2:
            messagebox.showwarning("警告", "需要至少2个数据点来生成图表")
            return

        try:
            self.status_var.set("正在生成图表...")
            self.root.update()

            all_experiments = self.experiment_data.get_all()

            fig1 = self.viz_manager.plot_convergence_curve(all_experiments)
            if fig1:
                self.viz_manager.save_figure(fig1, "01_收敛曲线")
                plt.close(fig1)

            fig2 = self.viz_manager.plot_analysis_time_distribution(all_experiments)
            if fig2:
                self.viz_manager.save_figure(fig2, "02_分析时间分布")
                plt.close(fig2)

            fig3 = self.viz_manager.plot_parameter_sensitivity(all_experiments)
            if fig3:
                self.viz_manager.save_figure(fig3, "03_参数敏感性分析")
                plt.close(fig3)

            # 4. 高维参数空间
            fig4 = self.viz_manager.plot_high_dimensional_space(all_experiments)
            if fig4:
                self.viz_manager.save_figure(fig4, "04_高维参数空间")
                plt.close(fig4)

            self.status_var.set("✓ 图表生成完成！")
            messagebox.showinfo("成功", "所有图表已保存到 visualizations 文件夹")

        except Exception as e:
            messagebox.showerror("错误", f"生成图表失败: {str(e)}")
            self.status_var.set("✗ 图表生成失败")

    def export_results(self):
        """导出结果为 CSV"""
        all_experiments = self.experiment_data.get_all()

        if not all_experiments:
            messagebox.showwarning("警告", "暂无数据可导出")
            return

        df = pd.DataFrame(all_experiments)

        params_df = pd.DataFrame(
            df['params'].tolist(),
            columns=PARAM_NAMES,
            index=df.index
        )

        # 包含ECRF列（如果存在）
        columns_to_export = ['round', 'is_initial', 'score', 'analysis_time', 'timestamp']
        
        # 检查是否有ecrf列
        if 'ecrf' in df.columns:
            columns_to_export.append('ecrf')
        
        export_df = pd.concat([
            df[columns_to_export],
            params_df
        ], axis=1)

        export_df['类型'] = export_df['is_initial'].apply(lambda x: '初始点' if x else '优化点')

        filename = os.path.join(self.results_dir, f"优化结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        export_df.to_csv(filename, index=False, encoding='utf-8-sig')

        messagebox.showinfo("成功", f"结果已导出:\n{filename}")
        self.status_var.set(f"✓ 结果已导出")

    def reset_all(self):
        """重置并备份历史数据"""
        all_experiments = self.experiment_data.get_all()

        if all_experiments and messagebox.askyesno("确认",
            "确定要重置所有数据并开始新的优化吗?\n\n"
            f"当前历史数据: {self.current_round} 轮\n"
            f"此操作无法撤销"):

            if all_experiments:
                backup_file = os.path.join(
                    self.results_dir,
                    f"备份_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                df = pd.DataFrame(all_experiments)
                df.to_csv(backup_file, index=False, encoding='utf-8-sig')
                logger.info(f"✓ 旧数据已备份: {backup_file}")

            self.experiment_data.clear()
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
            self.result_text.insert(tk.END, "✓ 已重置所有数据\n")
            self.result_text.insert(tk.END, f"{'='*45}\n\n")
            self.result_text.insert(tk.END, "点击【🚀 初始化优化】开始新的优化\n\n")
            self.result_text.see(tk.END)

            self.status_var.set("✓ 已重置")
            logger.info("✓ 已重置所有数据")


def main():
    root = tk.Tk()
    app = BayesianOptimizationUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()