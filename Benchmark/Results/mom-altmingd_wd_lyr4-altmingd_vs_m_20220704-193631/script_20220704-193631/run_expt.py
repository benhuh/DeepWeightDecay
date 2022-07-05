import numpy as np
import math
import importlib
from collections import defaultdict
import meta_learn as meta
importlib.reload(meta)
import expt_utils as expt_utils
import sys
from pathlib import Path 
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

n_repeat = 1
steps = 1500

deep_config = dict(weight_decay=0.01, T=steps, epoch_div=3, epoch_div_factor = 1000,
        lrs = (4,4), momentum=(0.9,0.9), clip_val= 3e-1, optim_type='SGD',
        num_layer=4, wide=True, Optimize_C=True)

algos_dict = {
    'mom': [meta.apply_method_of_moments,{}, n_repeat, 'MoM', '--sg'],
    'altmingd_wd_lyr4': [meta.apply_alt_min_gd_wd, deep_config, n_repeat, 'AltMinGD-DWD-1L', '--sk'],
    'altmingd': [meta.apply_alt_min_gd, {'N_step': steps, 'U_init': None, 'stepsize': 1.0}, n_repeat, 'AltMinGD-BM', '--sm'],
}

# Experiment configs
d = 36 # input dimension
r = 4 # hidden dimension
m = [8, 12, 20] # number of observations per task
rho = 4 # ratio
t = None # when t is set to be none, it will be calculated from rho
sigma = 0 # noise
script = "run_expt.py"
dist_x = 'normal'
dist_y = 'normal'
metrics = ['avg_mse_loss', 'surrogate_loss']



metrics_list = expt_utils.algos_vs_var_metrics(
    d, r, t, m, sigma, algos_dict,
    dist_x=dist_x, dist_y=dist_y,
    loglog=True, metrics=metrics, plot_metrics=None,
    results_dir='Results', script_file = script, rho = rho
    )