#!/usr/bin/python
# visualization from log file
import pickle
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 4]
import meta_learn as meta
import numpy as np
import os
import shutil
from glob import glob
import sys
import argparse
import math

def search(run_logs, param_dict):
    for i in range(run_logs.shape[0]):
        if (run_logs[i, :-1] == np.array([param_dict['d'], param_dict['r'], param_dict['t'], param_dict['m'], param_dict['s'], param_dict['algo']])).all():
            return run_logs[i, -1]
    return None
def get_t(m, d, r, factor):
	# get the number of tasks at the statistical limit given all the other parameters
	return math.ceil(r * (d - r) / (m - r) * factor)

parser = argparse.ArgumentParser(description='Visualization')
parser.add_argument('-d', type = int, nargs = "*", help = 'dimension', default = 36)
parser.add_argument('-t', type = int, nargs = "*", help = 'num of tasks', default= None)
parser.add_argument('-r', type = int, nargs = "*", help = 'rank', default= 4)
parser.add_argument('-m', type = int, nargs = "*", help = 'num of observations per task', default = [12, 20])
parser.add_argument('-s', type = float, nargs = "*", help = 'noise', default=0)
parser.add_argument('--rho', type = float, nargs = "*", help = 'ratio', default=2)
parser.add_argument('--alg', type = str, nargs = "*", default=['mom', 'altmingd_wd_lyr1', 'altmingd_wd_lyr2', 'altmingd'], help = 'algorithm')
parser.add_argument('--metrics', type = str, nargs = "*", default=['avg_mse_loss', 'surrogate_loss'], help = 'metrics')
parser.add_argument('--loglog', type= str, default="True")
args = parser.parse_args()

if type(args.d) == list and len(args.d) == 1:
    args.d = args.d[0]
if type(args.t) == list and len(args.t) == 1:
    args.t = args.t[0]
if type(args.r) == list and len(args.r) == 1:
    args.r = args.r[0]
if type(args.m) == list and len(args.m) == 1:
    args.m = args.m[0]
if type(args.s) == list and len(args.s) == 1:
    args.s = args.s[0]
if type(args.rho) == list and len(args.rho) == 1:
    args.rho = args.rho[0]

print(f"Runing {args}")
d_instance = isinstance(args.d, list)
r_instance = isinstance(args.r, list)
t_instance = isinstance(args.t, list)
m_instance = isinstance(args.m, list)
s_instance = isinstance(args.s, list)
rho_instance = isinstance(args.rho, list)
if d_instance:
    list_var, var_list = 'd', args.d
elif r_instance:
    list_var, var_list = 'r', args.r
elif t_instance:
    list_var, var_list = 't', args.t
elif m_instance:
    list_var, var_list = 'm', args.m
elif s_instance:
    list_var, var_list = 'sigma', args.s
elif rho_instance:
    list_var, var_list = 'rho', args.rho

run_logs = np.loadtxt('./run_logs.csv', delimiter = ',', dtype = 'str')
metrics_list = {var:{algo:{metric:[] for metric in args.metrics} for algo in args.alg} for var in var_list}



for algo in args.alg:
    for var in var_list:
        d, t, r, m, s = (args.d, args.t, args.r, args.m, args.s)
        if list_var == 'd':
            d =  var
        elif list_var == 'r':
            r =  var
        elif list_var == 't':
            t =  var
        elif list_var == 'm':
            m =  var
        elif list_var == 'sigma':
            s =  var
        elif list_var == 'homogeneity':
            homogeneity = var
        if t is None:
            t = get_t(m, d, r, args.rho)
        path = search(run_logs, {'d': d, 'r': r, 't': t, 'm': m, 's': s, 'algo': algo})
        if path is None:
            raise ImportError
        f = open(path, 'rb')
        output = pickle.load(f)
        f.close()
        for metric in args.metrics:
            metrics_list[var][algo][metric]=output['metrics_list'][var][algo][metric]

# plot config
legend_dict = {"altmingd_wd_lyr2": "AltMinGD-Full-rank (N = 3)", "altmingd_wd_lyr1": "AltMinGD-Full-rank (N = 2)", 
              "mom": "MoM", "altmingd": "AltMinGD-BM"}
algo_style = {"altmingd_wd_lyr2": '--sk', "altmingd_wd_lyr1": "--sy", 
              "mom": '--sg', "altmingd": '--sm'}


if args.loglog == "None":
    loglog = None
elif args.loglog == 'False':
    loglog = False
else:
    loglog = True
if loglog is None:
    plot_func = plt.plot
elif not loglog:
    plot_func = plt.semilogy
elif loglog:
    plot_func = plt.loglog
else:
    raise ValueError
for metric in args.metrics:
    for algo in args.alg:
        tmp_list = [metrics_list[var][algo][metric] for var in var_list]
        plot_func(var_list, np.mean(tmp_list, axis=1), algo_style[algo], label=legend_dict[algo])
        y = np.mean(tmp_list, axis = 1)
        ci = np.std(tmp_list, axis = 1)
        if len(y.shape) > 1:
            y= y.flatten()
        if len(ci.shape) > 1:
            ci= ci.flatten()
        plt.fill_between(var_list, (y-ci), (y+ci), color = algo_style[algo][3], alpha=.1)
    plt.legend()
    plt.xscale('symlog', linthresh = 0.001)
    plot_file_name = '{}_{}'.format(metric, f"{list_var}{var_list}_d{d}_r{r}_m{m}_t{t}")
    if list_var == 'm':
        plt.xlabel('k')
    elif list_var == 'sigma':
        plt.xlabel(r'$\sigma$')
    else:
        plt.xlabel(list_var)
    if metric == "surrogate_loss":
        plt.ylabel(r"$U$ Error")
    else:
        plt.ylabel("Training Loss")
        
    plt.show()
    plt.savefig(os.path.join("Results", plot_file_name+'.pdf'))
    print("Saving to: %s"%(os.path.join("Results", plot_file_name+'.pdf')))
    plt.close()