import tkinter as tk
import tkinter.messagebox as mb
import sys, os

# ensure package imports work when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# monkeypatch messagebox to avoid blocking dialogs during automated test
mb.askyesno = lambda *a, **k: False
mb.showinfo = lambda *a, **k: None
mb.showwarning = lambda *a, **k: None
mb.showerror = lambda *a, **k: None

from ui.main_window import BayesianOptimizationUI

root = tk.Tk()
root.withdraw()  # hide GUI
app = BayesianOptimizationUI(root)

# Initialize
app.initialize_optimization()

# Perform a few generate->submit cycles
for i in range(6):
    app.generate_next_params()
    if not app.has_pending_params:
        print(f"Round {i+1}: failed to generate params")
        break
    # simulate a measured CRF between 0.2 and 0.9
    crf = 0.5 + (i % 3) * 0.1
    app.crf_var.set(str(crf))
    app.submit_crf_and_generate_next()
    print(f"Round {i+1}: submitted CRF={crf}")

# Print summary
print("E2E simulate finished")
print(f"Total experiments: {len(app.experiment_data)}")
print(f"Optimization stopped: {app.optimization_stopped}, stop_reason: {app.stop_reason}")

# Save current checkpoint
app.save_history()
print("Checkpoint saved")

# 验证从检查点恢复
root2 = tk.Tk()
root2.withdraw()
app2 = BayesianOptimizationUI(root2)
loaded = app2.load_history()
print(f"Load history returned: {loaded}")
if loaded:
    app2.restore_optimization_state()
    print(f"Restored rounds: {app2.current_round}, experiments: {len(app2.experiment_data)}")
else:
    print("No history to restore")

root.destroy()
root2.destroy()