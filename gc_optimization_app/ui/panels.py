"""UI panels module"""

import tkinter as tk
from tkinter import ttk
from config import PARAM_NAMES, PARAM_UNITS

class ParameterPanel:
    """Parameter display panel"""
    
    def __init__(self, parent):
        self.frame = ttk.LabelFrame(parent, text="📋 Current Parameters", padding=10)
        
        scrollbar = ttk.Scrollbar(self.frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.text = tk.Text(
            self.frame, height=30, width=45, 
            yscrollcommand=scrollbar.set, 
            font=('Courier', 10)
        )
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.text.yview)
    
    def pack(self, **kwargs):
        self.frame.pack(**kwargs)
    
    def display_params(self, params, param_info=None):
        """Display parameters"""
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, "="*50 + "\n")
        self.text.insert(tk.END, "Current Parameters\n")
        self.text.insert(tk.END, "="*50 + "\n\n")
        
        for i, (name, value) in enumerate(zip(PARAM_NAMES, params)):
            unit = PARAM_UNITS[name]
            self.text.insert(tk.END, f"{name:<12}: {value:>8.2f} {unit}\n")
        
        if param_info:
            self.text.insert(tk.END, f"\n{param_info}")


class ResultPanel:
    """Result display panel"""
    
    def __init__(self, parent):
        self.notebook = ttk.Notebook(parent)
        
        # Tab 1: Optimization Log
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text="📝 Optimization Log")
        
        scrollbar = ttk.Scrollbar(self.log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(
            self.log_frame, height=30, width=45, 
            yscrollcommand=scrollbar.set, 
            font=('Courier', 9)
        )
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.result_text.yview)
        
        # Tab 2: Statistics
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="📈 Statistics")
        
        self.stats_text = tk.Text(
            self.stats_frame, height=30, width=45, 
            font=('Courier', 9)
        )
        scrollbar2 = ttk.Scrollbar(self.stats_frame, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=scrollbar2.set)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
    
    def pack(self, **kwargs):
        self.notebook.pack(**kwargs)
