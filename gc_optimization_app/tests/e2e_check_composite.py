from tkinter import Tk
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui.main_window import BayesianOptimizationUI

root = Tk()
root.withdraw()
app = BayesianOptimizationUI(root)
app.initialize_optimization()
app.generate_next_params()
params = app.next_params
at = app.constraint_handler.calculate_analysis_time(params)
print('analysis_time', at)
app.crf_var.set('0.88')
app.submit_crf_and_generate_next()
last = app.experiment_data.get_all()[-1]
print('last experiment:', last)
print('composite_score:', last.get('composite_score'))
root.destroy()