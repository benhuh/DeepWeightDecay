###################################

import numpy as np
from tinydb import TinyDB, Query, where
from tinydb.storages import MemoryStorage
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import json


def get_db(base_path, db=None, load_flag=True):
    if db is None:
        db = TinyDB(storage=MemoryStorage)
    if load_flag:
        load(base_path, db)
    return db

def load(base_dir, db=None, print_flag=True): 
    if print_flag:
        print(f'Loading {base_dir}')
    
    log_dir_list = list(Path(base_dir).expanduser().glob('*'))
    log_dir_list.sort(key=os.path.getmtime)
    for log_dir in log_dir_list:
        if not log_dir.is_dir() or db.contains(Query().log_dir == str(log_dir)):
            continue
        try:
            if len(list(log_dir.glob('events.*')))>0:
                events_files = list(log_dir.glob('events.*'))
                config_files = list(log_dir.glob('config*'))
                events_files.sort()#key=os.path.getmtime)
                config_files.sort()#key=os.path.getmtime)
                # assert(len(events_files)==1)
                # assert(len(config_files)==1)
                for events_file, config_file in zip(events_files, config_files):
                    # print(f'Loading {events_file}')
                    evt_acc = EventAccumulator(str(events_file), purge_orphaned_data=False)
                    evt_acc.Reload()
                    
                    with open(str(config_file)) as file:
                        config = json.load(file)
                    # config['base_dir'] = str(base_dir)
                    # config['log_dir'] = str(log_dir)
                    
                    config = {**config, **config['problem']}
                    config.pop('problem')
                    config['events'] = evt_acc

                    db.insert(config)
            else: # Recursive load 
                load(log_dir, db, print_flag=True)
        except Exception as e: 
            print(e)

# def unload(base_dir):
#     for log_dir in Path(base_dir).expanduser().glob('*'):
#         if not log_dir.is_dir() or not db.contains(Run.log_dir == str(log_dir)):
#             continue
#         db.remove(where('log_dir') == str(log_dir))
#         print('Removing', log_dir)
        
def get_values(run, tag):
    timestamps, steps, values = zip(*run['events'].Scalars(tag))
    return np.array(values)

def get_steps(run, tag):
    timestamps, steps, values = zip(*run['events'].Scalars(tag))
    return np.array(steps)


###########################################
import os
from torch.utils.tensorboard import SummaryWriter
from logging import Logger #getLogger, FileHandler

def set_logging(config, kwarg_dict):

    tensorboard_path, partial_path = get_tensorboard_path(config, kwarg_dict)
    print(partial_path)

    _writer = SummaryWriter(tensorboard_path, flush_secs=30)
    _log = Logger(name=tensorboard_path)
    return _writer, _log, tensorboard_path

def save_config(config, path, filename, pop_list:list=None):
    os.makedirs(path, exist_ok=True)
    config_path = os.path.join(path, filename)
    
    config_dict = vars(config).copy()
    config_dict['problem'] = vars(config_dict['problem'])
    if pop_list is not None: #len(pop_list)>0:
        for pop_key in pop_list:
            config_dict.pop(pop_key)
                
    with open(config_path, 'a') as f:
        f.write(json.dumps(config_dict, sort_keys=True) + "\n")
            

def get_base_path(config): 
    root_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(root_path, 
                             'results', 
                             config.problem.name,
                             config.experiment,
                             # dict2str(vars(config.problem)), 
                             )
    return base_path

import re
def dict2str(cfg_dict):
    cfg_dict_ = cfg_dict.copy()
    for key, val in cfg_dict.items():
        if key == 'name' or val == None: # remove 'name' and any None entries
            cfg_dict_.pop(key)
            
    config_str = json.dumps(cfg_dict_) #, sort_keys=True)
    config_str = re.sub(r"('|{|}| )", "", config_str)
    config_str = re.sub(r'"', '', config_str)
    config_str = re.sub(r",", "|", config_str)
    return config_str


def get_tensorboard_path(config, kwarg_dict): # **kwargs): 
    # config.set_params(**kwarg_dict)    
    base_path = get_base_path(config)
    
    add_path = dict2str(kwarg_dict)
    
#     _wide = '_wide' if config.wide else '_narrow'
#     add_path = f'depth{config.depth}_wd{config.weight_decay}_init_scale{config.init_scale}' + _wide 

    path = os.path.join(base_path, add_path)
    tensorboard_path, run_path = get_run_path(path, run_str = 'run')
    return tensorboard_path, add_path+'/'+run_path

def get_run_path(path, run_str = 'run', suffix=''):
    run_num = 0
    while True:
        run_path = run_str + str(run_num)+suffix
        full_path = os.path.join(path, run_path)
        if not os.path.exists(full_path):
            break
        run_num += 1
    return full_path, run_path

###########################################

import matplotlib.pyplot as plt

def make_2_plot(db, loop_params, params, loss_or_svd=None):
    key1, vals1 = loop_params.popitem()  # last=False
    key2, vals2 = loop_params.popitem()

    fig, axes = plt.subplots(len(vals1), len(vals2), figsize=(20, 12) , sharex=True, sharey=True)
    fig.tight_layout(w_pad=3, h_pad=3)
    
    for i, val1 in enumerate(vals1):
        params[key1] = val1 

        for j, val2 in enumerate(vals2):
            params[key2] = val2
            
            ax = axes[i,j]
            runs = db.search(Query().fragment(params))
            # print(params)
            # runs = runs[-1:]
            # print(run)
            # test_loss = get_values(run, 'loss/test')[-1]
            # train_loss = get_values(run, 'loss/train')[-1]

            make_1_plot(runs, loss_or_svd, ax=ax)
                
            # ax.set_title(f'{_wide}, depth{depth}, wd{wd}, loss: [{test_loss:.4f}, {train_loss:.4f}]')
            ax.set_title(params)
            # ax.set_xlim(left = 0, right = T) 


def make_1_plot(db, loop_params, params, loss_or_svd=None):
    key1, vals1 = loop_params.popitem()  # last=False

    fig, axes = plt.subplots(len(vals1),  figsize=(20, 12) , sharex=True, sharey=True)
    fig.tight_layout(w_pad=3, h_pad=3)
    
    for i, val1 in enumerate(vals1):
        params[key1] = val1 

        ax = axes[i]
        runs = db.search(Query().fragment(params))
        # print(params)
        # runs = runs[-1:]
        # print(run)
        # test_loss = get_values(run, 'loss/test')[-1]
        # train_loss = get_values(run, 'loss/train')[-1]

        make_0_plot(runs, loss_or_svd, ax=ax)

        # ax.set_title(f'{_wide}, depth{depth}, wd{wd}, loss: [{test_loss:.4f}, {train_loss:.4f}]')
        ax.set_title(params)
        # ax.set_xlim(left = 0, right = T) 

            
            
def make_0_plot(db, loss_or_svd=None, params=None, ax=None):
    if params is None:
        runs = db
    else:
        runs = db.search(Query().fragment(params))

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes()

    alpha=1
    
    for run in runs:
    
        # print(depth, dataset, lr, np.min(get_values(run, 'loss/test')))
    #     xs = get_steps(run, 'singular_values/0')
        xs = get_steps(run, 'loss/test')

        if loss_or_svd in [None, 'loss']:
    #                     ax.semilogy(xs, get_values(run, 'loss/surrogate'), '-', color='blue', alpha=alpha)
            ax.semilogy(xs, get_values(run, 'loss/test'), '--', color=plt.cm.hot(0 / 10), alpha=alpha)
            ax.semilogy(xs, get_values(run, 'loss/train'), ':', color=plt.cm.hot(3 / 10), alpha=alpha)
            # ax.set_ylim(bottom = 0.0001, top = 3)

        if loss_or_svd in [None, 'svd']:
            for k in range(20):
                try:
                    ax.semilogy(xs, get_values(run, f'singular_values/{k}'), color=plt.cm.summer(k / 10), alpha=alpha)
                except:
                    pass
        ax.set_ylim(bottom = 1e-4, top = 100)    