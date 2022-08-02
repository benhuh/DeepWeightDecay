import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pdb import set_trace

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from MTLR_init import base_task_batch

##########
def plot_all(history, *names, skip=10):  # def plot_all(loss_list, *args):
#     names = ['losses'] + list(names)  # , 'err'
    semilogy_list = ['losses', 'grad_norm', 'sing_val', 'test_loss', 'test_mse', 'train_mse'] #, 'proj_norm']
    print(names)
    i_max = len(names)
    fig, axes = plt.subplots(1, i_max, figsize=(6*i_max, 4), sharex=True, sharey=False)
    
    for name, ax in zip(names, axes):
        arg = history[name]
        
        arg = np.stack(arg)

#         skip_ = skip if arg[0].shape[0]<50 else 10*skip


#         if arg[0].shape[0]>50:
        
        skip2 = 1 if arg.shape[1]<50 else arg.shape[1]//50
            
        t_ = np.arange(len(arg), step=skip)
        arg = arg[::skip, ::skip2]
        
        plot = ax.semilogy if name in semilogy_list else ax.plot         #or len(arg[0].shape)==0 or arg[0].shape[0]<5 :
            
        if name == 'losses':
            plot(t_, arg); 
#             ax.legend(['trueloss', 'surrogate', 'test', 'train', 'trueloss_LB'])
            # ax.legend(['trueloss', 'surrogate', 'train', 'trueloss_LB', 'W_norm', 'eps'])
            ax.legend(['U error', 'train', 'eps'])
        else:
            plot(t_, arg);  
            # ax.legend([name]); 
        ax.set_title(name, fontsize=16)
        ax.set_xlabel('iter', fontsize=16)
    plt.show()
    return fig



    
def plot_loss_profiles(array, loss_type=None, t_f = None, ymin=None, ymax=None, plot_type='plot', log=False, skip=1, save_name=None,
                      task_batch_factors = None, x_batch_all = None  ):
#                       task_batch_factors = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2], x_batch_all = [5,7,9,12,24,36]  ):
    idx_dict = dict(true=0, surrogate=1, train=2, test=3) # , generalization=4)
#     idx_dict = dict(true1=0, true2=1, surrogate=2, train=3, test=4) # , generalization=4)
    idx = idx_dict[loss_type] if loss_type is not None else None #[0,1,2,3]
    
    i_max, j_max, n_repeat, loss_types, t_f_ = array.shape  # misc: true_loss, surrogate_loss, train_loss, test_loss
    if t_f is None:
        t_f = t_f_
#     else:
    array = array[...,:t_f:skip]   # clip Time
#     array = array[...,::skip]
    
    if idx is None:
        array = array[:,:,:,:3,:]
    else:
        array = array[:,:,:,idx,:]
    
    
    t_ = np.arange(t_f, step=skip)

    fig, axes2d = plt.subplots(max(i_max,2), max(j_max,2), figsize=(8,8), sharex=True, sharey=True)
    
    
    assert len(task_batch_factors) == j_max
    assert len(x_batch_all) == i_max

    for i, row in enumerate(axes2d):
        for j, cell in enumerate(row):  
            if i>i_max-1 or j>j_max-1:
                pass
            else:
                array_ij = array[i,j]
                array_ij = np.moveaxis(array_ij,-1,0) #move time axis to 0
                if plot_type=='plot': 
                    cell.plot(t_, array_ij)
                elif plot_type=='semilogy':
                    cell.semilogy(t_, array_ij)
                elif plot_type=='mean_semilogy':
                    cell.semilogy(t_, np.mean(array_ij,axis=1))
                    
                elif plot_type=='errorbar':
                    print(i,j)
                    # if j==0:
                    d_ = array_ij #[:,:,k]
                    set_trace()
                    df = pd.DataFrame(d_.T)
                    df = pd.melt(frame = df,  var_name = 'time',  value_name = 'log-loss')
                    sns.lineplot(ax = cell, data = df, x = 'time', y = 'log-loss') #, sort = False)
                    cell.set_yscale("log", nonpositive='clip')
            
                elif plot_type=='hist':
                    array_ij=array_ij[...,-1].reshape(-1)
                    cell.hist(array_ij, bins=50, log=log)

                if ymin is not None:
    #                 plt.ylim(bottom=ymin)
                    cell.set_ylim(bottom=ymin)
                if ymax is not None:
                    cell.set_ylim(top=ymax)

                if i == len(axes2d) - 1:
                    cell.set_xlabel(r"$\rho$" + f" = {task_batch_factors[j]}") #+r" $T^*$")  #+r"$\times T^*$")
                if j == 0:
                    cell.set_ylabel("k = {0:d}".format(x_batch_all[i]))                
                    
    plt.tight_layout()
    
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel("Iterations", labelpad=20)
    plt.ylabel("Loss", labelpad=23)    
    plt.show()
    
    
    if save_name is not None:
        fig.savefig(f'{save_name}.pdf')
    
    
def plot_loss(filename, loss_type='surrogate', **kwargs):
    file = torch.load(filename)
    plot_loss_profiles(file['losses_stack'], loss_type=loss_type, **kwargs)


def plot_singular(filename, i=1): #, eps=0.0001, num_proj = None):
    file = torch.load(filename)
    sig = file['singular_values'][:,:,i]
    proj_norm = file['singular_vectors'][:,:,i]
    plot_loss_profiles(sig, None, plot_type='semilogy')
    plot_loss_profiles(proj_norm, None, ymin=-0.1, ymax=1.1, plot_type='plot')
#     proj_norm = proj_norm[...,-1]
#     proj_norm = proj_norm.reshape(proj_norm.shape[0],proj_norm.shape[1],-1)

    plot_loss_profiles(np.log(1e-2+sig), None, plot_type='hist', ymax=40, log=False)
#     top_n = 4
#     proj_top_mean=np.sqrt((proj_norm[...,:top_n,-1]**2).sum(axis=-1)).mean(axis=-1) #/top_n
#     plt.imshow(proj_top_mean)
#     plt.colorbar()
#     plt.show()
    
def plot_singular_hist(filename): #, eps=0.0001, num_proj = None):
    file = torch.load(filename)
    sig = file['singular_values']
    proj_norm = file['singular_vectors']
#     plot_loss_profiles(sig, None), plot_type='semilogy'
#     plot_loss_profiles(proj_norm, None, ymin=-0.1, ymax=1.1, plot_type='plot')
#     proj_norm = proj_norm[...,-1]
#     proj_norm = proj_norm.reshape(proj_norm.shape[0],proj_norm.shape[1],-1)

    plot_loss_profiles(np.log(1e-2+sig), None, plot_type='hist', ymax=40, log=False)
#     plot_loss_profiles(proj_norm, None, plot_type='hist', ymax=80)
    set_trace()
    top_n = 4
    proj_top_mean=np.sqrt((proj_norm[...,:top_n,-1]**2).sum(axis=-1)).mean(axis=-1) #/top_n
    plt.imshow(proj_top_mean)
    plt.colorbar()
    plt.show()
#     num_proj = num_proj or proj_norm.shape[3]  #     num_sig = sig.shape[3]  
#     for eps in [ 0, 0.001]: #, 0.1]:
#     print('eps:', eps)
#     sig_score = eps/(sig[...,:num_proj,:]**2+eps)    #     sig_score = sig<eps
#     trueloss =   1* (4 - (proj_norm[...,:num_proj,:]**2).sum(axis=3)) + (proj_norm[...,:num_proj,:]**2 * sig_score).sum(axis=3)
#     trueloss = np.expand_dims(trueloss, axis=3)
#     plot_loss_profiles(trueloss, 'true', ymin=1e-3)    
        
#     trueloss_final = trueloss.squeeze(axis=3)[...,-1]
#     return trueloss_final
        
        
    

####################################

task_factor_str = r"$\rho$"

def plot_pandas_sns(temp, skip=100, loss_type_num=3, save_name=None):
    
    data_long = get_datalong(temp, skip=skip, loss_type_num=loss_type_num)

    fig = plt.figure(figsize=(8,8))
    g = sns.FacetGrid(data=data_long, col=task_factor_str, row='batch', hue='loss_type', sharey= True, sharex=True)    
    g.map_dataframe(sns.lineplot, x='time', y='value')    # g.map(sns.lineplot, 'time', 'value2') 
    g.map_dataframe(annotate_plot, col=task_factor_str, row='batch')
    g.set(yscale='log') # g.set(ylim=(1e-1, None))
    g.add_legend()     # plt.legend(labels=loss_types)

    # fig.add_subplot(111, frame_on=False)
    # plt.tick_params(labelcolor="none", bottom=False, left=False)
    # plt.xlabel("Iterations", labelpad=20)
    # plt.ylabel("Loss", labelpad=23)    
    
    # ax.set_xlabel('')
    # ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.02))
    # plt.tight_layout()  # fit legend and labels into the figure
    
    plt.show()
    
    if save_name is not None:
        g.savefig(f'{save_name}.pdf')    
    
def get_datalong(temp, skip=100, loss_type_num=3):
    array = temp['losses_stack']
    batches=temp['x_batch_all']
    task_factors=temp['task_batch_factors']
    loss_types = ['eval','surrogate','train','test','W_norm']
    
    t_f= array.shape[-1]
    array = array[...,::skip]
    array = array[:,:,:,:loss_type_num,:]
    loss_types = loss_types[:loss_type_num]
    
    runs = range(array.shape[2])
    times = np.arange(t_f, step=skip)

    index = pd.MultiIndex.from_product([batches,task_factors,runs,loss_types,times], names=['batch',task_factor_str,'run#','loss_type','time']) 
    df0 = pd.DataFrame({'_': array.flatten()}, index=index)
    data_long = df0.melt(ignore_index=False, value_name='value').reset_index()
    return data_long


def annotate_plot(data, **kws):
    # n = len(data)
    # print(kws)
    if kws['label'] =='eval':

        k = np.mean(data['batch'])
        factor = np.mean(data[task_factor_str]) 
        task_num = base_task_batch(Nx=36, Nctx=4, x_batch=k, factor=factor, Ny=1)

        ax = plt.gca()
        ax.text(.7, .8, f"T = {task_num}", transform=ax.transAxes)
