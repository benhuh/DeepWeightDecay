import numpy as np
import matplotlib.pyplot as plt
import math
plt.rcParams['figure.figsize'] = [5, 2]

from scipy import linalg
from scipy.stats import linregress

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import importlib
import pickle
import timeit
import os
import csv

import meta_learn as meta
importlib.reload(meta)

def get_t(m, d, r, factor):
	# get the number of tasks at the statistical limit given all the other parameters
	return math.ceil(r * (d - r) / (m - r) * factor)

def set_directories(dir_name, args = None, func = None, env_var = None):
	dir_path = dir_name

	hparams_dir = func.__name__
	if True:
		hparams_dir = "%s|d%s_r%s_t%s_m%s_sigma%s"%(hparams_dir, env_var[0], env_var[1], env_var[2], env_var[3], env_var[4])
		for ks in args.keys():
			hparams_dir = "%s|%s_%s"%(hparams_dir, str(ks), str(args[ks]))	
		temp_dir = os.path.join(dir_path, hparams_dir or 'default')
		current_run = 0
		while True:
			tensorboard_dir = os.path.join(temp_dir, 'run' + str(current_run))
			current_run += 1
			if not os.path.exists(tensorboard_dir):
				break
	return tensorboard_dir

def dump_script(
	dirname, script_file, dest=None, timestamp=None, file_list=None):
	import glob, os, shutil, sys
	from datetime import datetime

	if dest is None:
		if timestamp is None:
			timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
		dest = os.path.join(
			dirname, 'script_{}'.format(timestamp))
	os.mkdir(dest)

	print('copying files to {}'.format(dest))
	if file_list is None:
		file_list = glob.glob("*.py")
	for file in file_list:
		print('copying {}'.format(file))
		shutil.copy2(file, dest)
	print('copying {}'.format(script_file))
	shutil.copy2(script_file, dest)

	with open(os.path.join(dest, "command.txt"), "w") as f:
			f.write(" ".join(sys.argv) + "\n")

def save_output(output, timestamp, dir_name=None):
	if dir_name == None:
		dir_name = '.'
	path = '{}/output_{}.pkl'.format(dir_name, timestamp)
	f = open(path, 'wb')
	pickle.dump(output,f)
	f.close()
	return path

def get_t(m, d, r, factor):
	return math.ceil(r * (d - r) / (m - r) * factor)
	

def algos_vs_var_metrics(
		d, r, t, m, sigma, algos_dict,
		homogeneity=None,
		dist_x=None, dist_y=None,
		loglog=False, xlims=None, ylims=None,
		metrics=None, plot_metrics=None,
		results_dir=None, script_file=None, rho = 1
		):
	"""
	algos_dict = {
		'alt': [meta.apply_alt_min,
			{'N_step': 10, 'U_init': None, 'init_mom': True}, 10],
	}
	"""
	N_trials = 1
	for algo in algos_dict:
		algo_N_trials = algos_dict[algo][2]
		if algo_N_trials < 1:
			raise ValueError
		N_trials = max(algo_N_trials, N_trials)
	limit_t = True if t == None else False
	instance_list = (list, np.ndarray)
	d_instance = isinstance(d, instance_list)
	r_instance = isinstance(r, instance_list)
	t_instance = isinstance(t, instance_list)
	m_instance = isinstance(m, instance_list)
	sigma_instance = isinstance(sigma, instance_list)
	homogeneity_instance = isinstance(homogeneity, instance_list)

	nof_list_vars = (
		d_instance + r_instance +
		t_instance + m_instance +
		sigma_instance + homogeneity_instance)
	if nof_list_vars > 1:
		ValueError('varying {} variables'.format(nof_list_vars))

	if d_instance:
		list_var, var_list = 'd', d
	elif r_instance:
		list_var, var_list = 'r', r
	elif t_instance:
		list_var, var_list = 't', t
	elif m_instance:
		list_var, var_list = 'm', m
	elif sigma_instance:
		list_var, var_list = 'sigma', sigma
	elif homogeneity_instance:
		list_var, var_list = 'homogeneity', homogeneity
	else:
		list_var, var_list = 'k', [None]


	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	results_dir = 'results/neurips_submit' if results_dir is None else results_dir
	expt_name = '{}_vs_{}_{}'.format(
			'-'.join(algos_dict.keys()), list_var, timestamp)
	dir_name = os.path.join(results_dir, expt_name)
	os.makedirs(dir_name, exist_ok=True)
	if script_file is not None:
		dump_script(
				dir_name, script_file, timestamp=timestamp,
				file_list=['meta_learn.py', 'expt_utils.py'])

	if metrics is None:
		metrics = ['dist_U', 'dist_U_spectral', 'avg_mse_loss']

	plt_titles = {
		'dist_U': 'Distance of U',
		'dist_U_spectral': 'Distance of U in spectral norm',
		'avg_mse_loss': 'Average MSE loss',
		'true_loss': 'True loss',
		'test_mse_loss': 'Test loss',
		'surrogate_loss': 'Surrogate loss'
	}

	plt_ylabels = {
		'dist_U': 'Subspace Distance',
		'dist_U_spectral': 'Subspace Distance',
		'avg_mse_loss': 'Average MSE loss',
		'true_loss': 'True loss',
		'test_mse_loss': 'Test loss',
		'surrogate_loss': 'Surrogate loss'
	}

	if plot_metrics is None:
		plot_metrics = metrics
	metrics_list = {var:{algo:{metric:[] for metric in metrics} for algo in algos_dict} for var in var_list}

	for var in var_list:
		if list_var == 'd':
			d =  var
		elif list_var == 'r':
			r =  var
		elif list_var == 't':
			t =  var
		elif list_var == 'm':
			m =  var
		elif list_var == 'sigma':
			sigma =  var
		elif list_var == 'homogeneity':
			homogeneity = var
		# _metrics_list = {
		# 	algo:{metric:[] for metric in metrics} for algo in algos_dict}
		if limit_t:
			t = get_t(m, d, r, rho)
		print(f"Runing m: {m}, d: {d}, r: {r}, t: {t}")
		for trial_idx in range(N_trials):
			
			prob = meta.MetaLearnProb(d, r, t, sigma=sigma, homogeneity=homogeneity)
			X, noise = prob.generate_data(m, noise=True, dist_x=dist_x, dist_y=dist_y)

			for algo, algo_setup in algos_dict.items():
				algo_func, algo_params, algo_N_trials, _, _ = algo_setup
				if not trial_idx < algo_N_trials:
					continue
				# generate writer
				env_var = d, r, t, m, sigma
				tensorboard_dir = set_directories(dir_name, algo_params, algo_func, env_var)
				os.makedirs(tensorboard_dir, exist_ok=True)
				summary_writer = SummaryWriter(log_dir=tensorboard_dir)
				output = algo_func(prob, **algo_params, writer = summary_writer)
				for metric in metrics:
					metrics_list[var][algo][metric].append(output['{}_list'.format(metric)][-1])
	outputs_dict = {
		'd' :d,
		'r': r,
		't': t,
		'm': m,
		'sigma': sigma,
		'homogeneity': homogeneity,
		list_var: var_list,

		'list_var': list_var,
		'var_list': var_list,

		'algos_dict': algos_dict,

		'plt_titles': plt_titles,
		'plt_ylabels': plt_ylabels,

		'dist_x': dist_x,

		'dist_y': dist_y,

		'loglog': loglog,
		'metrics': metrics,
		'plot_metrics': plot_metrics,

		'results_dir': results_dir,
		'script_file': script_file,

		'metrics_list': metrics_list,
	}
	# store experiment information
	path = save_output(outputs_dict, timestamp, dir_name=dir_name)
	with open("run_logs.csv", "a") as run_logs:
		writer = csv.writer(run_logs)
		for var in var_list:
			for algo, _ in algos_dict.items():
				writer.writerow([d, r, t, m, sigma, algo, path])
	return metrics_list


