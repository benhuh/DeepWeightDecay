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

DIR = str(sys.argv[1])
name = [y for x in os.walk(DIR) for y in glob(os.path.join(x[0], '*.pkl'))][0]
print(name)

f = open(name, 'rb')
output = pickle.load(f)
f.close()

locals().update(output)

loglog = str(sys.argv[2])

if loglog == "None":
    loglog = None
elif loglog == 'False':
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
if len(sys.argv) > 3:
    expt_name = str(sys.argv[3])
else:
    expt_name = '{}_vs_{}'.format('-'.join(algos_dict.keys()), list_var)
OUTPUT_DIR = os.path.dirname(name)
print(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok = True)

legend_dict = {"altmingd_wd_lyr2": "AltMinGD-Full-rank (N = 3)", "altmingd_wd_lyr1": "AltMinGD-Full-rank (N = 2)", 
              "mom": "MoM", "altmingd": "AltMinGD-BM"}

for metric in metrics:
    for algo in algos_dict:
        if algo == "altmingd_wd_lyr4":
            continue
        algo_label = legend_dict[algo]
        if algo == "altmingd_wd_lyr1":
            algo_style = "--sy"
        else:
            algo_style = algos_dict[algo][4]

        plot_func(var_list, np.mean(metrics_list[algo][metric], axis=1), algo_style, label=algo_label)
        y = np.mean(metrics_list[algo][metric], axis = 1)
        ci = np.std(metrics_list[algo][metric], axis = 1)
        if len(y.shape) > 1:
            y= y.flatten()
        if len(ci.shape) > 1:
            ci= ci.flatten()
        # print(algo_style)
        plt.fill_between(output[list_var], (y-ci), (y+ci), color = algo_style[3], alpha=.1)
    plt.legend()
    plt.xscale('symlog', linthresh = 0.001)
    plot_file_name = '{}_{}'.format(metric, expt_name)
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
    # plt.savefig(os.path.join(OUTPUT_DIR, plot_file_name+'_labeled.png'))
    plt.savefig(os.path.join(OUTPUT_DIR, plot_file_name+'_labeled.pdf'))
    plt.close()