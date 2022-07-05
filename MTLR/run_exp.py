import torch
import numpy as np
from collections import defaultdict

from MTLR_init import initialize_task_simple, init_WC, get_task_batch_arch #initialize_task # get_ortho, base_task_batch #initialize_task_best, initialize_task_better, initialize_task_eye, initialize_task_worst
from MTLR_train_simple import run_grad_descent
# from MTLR_train_NGD import run_grad_descent
from MTLR_utils import get_arch_filename, set_seed_everywhere, const_padding, zip_list_padded
from MTLR_plot import plot_loss_profiles, plot_all, plot_loss, plot_singular, plot_pandas_sns

# grad_fn = 'grad_auto' #'grad' # #nat_grad_sqrt
# init_task_fnc = initialize_task


zeroW = False
zeroC = True
augment=False 
# orth=dict(C=False, X=False)

# clip_val_C = 0.1



Nx, Nctx, Ny = 36, 4, 1


def run_exp(seed=0, n_repeat=1,
            num_layer=1,
            wide=True,
            init_scale = defaultdict(float, ortho=1/4, randn=0, W_0=0),
            W0_overlap=4,
            orth=dict(C=False, X=False),
            Optimize_C = True,
            single_task = False,
            weight_decay = [0.01], 
            nuclear_decay = [0.0],
            noise = 0,
            T=[1000],
            lrs = (4,0), #(0.0002, 0), 
            momentum = (0.9,0),
            optim_type = 'SGD',
            clip_val= 3e-1,
            x_batch_all = [5, 6, 8, 12, 20, 36],
            rho_all = [0.75, 1, 1.25, 1.5, 1.75], 
            device = "cpu", #None, #
            epoch_div = 1, epoch_div_factor=100, weight_decay_min=1e-8,
            additional_name=None, 
            names_to_plot=['losses', 'sing_val', 'sing_vec', 'W'], # 'grad_norm'
            norm_grad = True):
# def run_exp(x_batch_all, rho_all, additional_name=None, *names_to_plot):


    device = device or torch.device("cuda") if torch.cuda.is_available()  else torch.device("cpu")        # device = "cuda" if torch.cuda.is_available()  else "cpu"
         
    def initialize(x_batch, rho, seeds):
        task_batch, N_arch = get_task_batch_arch(Nx, Nctx, x_batch, rho, Ny, num_layer, wide)

        x_batch_train, x_batch_test = x_batch, 0
        # W0, C0, Xs, Ys = init_task_fnc(Nx, Nctx, task_batch, x_batch_test, x_batch_train, augment=augment, orth=orth, noise=noise, Ny=Ny, device=device)
        W0, C0, Xs, Ys = initialize_task_simple(Nx, Nctx, task_batch, x_batch_test, x_batch_train, Ny=Ny, augment=augment, orth=orth, noise=noise, device=device, seed_task=seeds['task'], seed_data=seeds['data'], single_task=single_task)
        W, C = init_WC(Nx, Nctx, task_batch, N_arch, init_scale, zeroW = zeroW, zeroC= zeroC, Ny=Ny, seed=seeds['model']) #, W0=W0, W0_overlap=W0_overlap)         
            
        print_setting1(x_batch, rho) 
        return W0, C0, Xs, Ys, W, C
    
    def run(x_batch, rho, T_, lrs_, weight_decay_, nuclear_decay_, clip_val_,seeds):
        history = defaultdict(list)
        W0, C0, Xs, Ys, W, C = initialize(x_batch, rho, seeds)
        _ = run_grad_descent(W, C, W0, Xs, Ys, T_, history, lrs=lrs_, momentum=momentum, weight_decay=weight_decay_, nuclear_decay=nuclear_decay_, clip_val=clip_val_, Optimize_C=Optimize_C, optim_type=optim_type, device=device, epoch_div=epoch_div, epoch_div_factor=epoch_div_factor, weight_decay_min=weight_decay_min)

        fig = plot_all(history, *names_to_plot) 
        loss_ = np.stack(history['losses'],axis=1);        sing_val_ = np.stack(history['sing_val'],axis=1);        sing_vec_ = np.stack(history['sing_vec'],axis=1) 
        return loss_, sing_val_, sing_vec_, fig
    
    def print_setting1(x_batch, rho):
        task_batch, N_arch = get_task_batch_arch(Nx, Nctx, x_batch, rho, Ny, num_layer, wide)
        print(f'x_batch:{x_batch}, task_batch:{task_batch}, wide:{wide}, orth:{orth}')
        # print(f'x_batch:{x_batch}, task_batch:{task_batch}, init_scale:{dict(init_scale)}, wide:{wide}, orth:{orth}, N_arch:{N_arch}')  # Nx:{Nx}, Nctx:{Nctx}, 

    def print_setting2(weight_decay, lrs, clip_val, Optimize_C):
        print(f'weight_decay:{weight_decay}, lrs:{lrs}, clip_val:{clip_val}, Optimize_C:{Optimize_C}') #, max_W:{max_W}, momentum:{momentum}')
    

    N_arch, file_name =  get_arch_filename(Nx, Nctx, num_layer, wide, dict(init_scale), lrs, Optimize_C, orth['C'], weight_decay, n_repeat, noise, x_batch_all, rho_all)
    if additional_name is not None:
        file_name+='_'+additional_name
    
    print('file_name:',file_name)
    losses_all0, sig_all0, vec_all0 = [], [], []
    
    for i, x_batch in enumerate(x_batch_all): # batch per task
        losses_all1, sig_all1, vec_all1 = [], [], []
        for j, rho in enumerate(rho_all):     # num of tasks
            x_batch_train, x_batch_test = x_batch, 0
            losses_all2, sig_all2, vec_all2 = [], [], []
            print_setting2(weight_decay, lrs, clip_val, Optimize_C)
            for idx, (T_, lrs_, weight_decay_, nuclear_decay_, clip_val_) in enumerate(zip_list_padded(T, lrs, weight_decay, nuclear_decay, clip_val)):
                if idx>0:
                    raise ValueError('only one step.. of T_')
                    
                losses_all3, sig_all3, vec_all3 = [], [], []
                for k in range(n_repeat):
                    print('x_batch, rho, repeat =', x_batch, rho, k)
                    if seed == 0:
                        seeds = dict(task=0, data=0, model=0)
                    else:
                        seed0 = 100*seed
                        seeds = dict(task=seed0+k, data=seed0+k, model=seed0+k)
                    loss_, sing_val_, sing_vec_, fig = run(x_batch, rho, T_, lrs_, weight_decay_, nuclear_decay_, clip_val_,seeds)
                    fig.savefig(f'depth{num_layer}_wide{wide}_k{x_batch}_rho{rho}.pdf')
                    losses_all3.append(const_padding(loss_, sum(T))); sig_all3.append(const_padding(sing_val_, sum(T))); vec_all3.append(const_padding(sing_vec_, sum(T)))
                # losses_all2.append(np.stack(losses_all3));  sig_all2.append(np.stack(sig_all3));  vec_all2.append(np.stack(vec_all3))
            losses_all2 = losses_all3;  sig_all2 = sig_all3;  vec_all2 = vec_all3            
            losses_all1.append(np.stack(losses_all2));  sig_all1.append(np.stack(sig_all2));  vec_all1.append(np.stack(vec_all2))
        losses_all0.append(np.stack(losses_all1));  sig_all0.append(np.stack(sig_all1));  vec_all0.append(np.stack(vec_all1))
    losses_all0 = np.stack(losses_all0);  sig_all0 = np.stack(sig_all0);  vec_all0 = np.stack(vec_all0)
    
    params = dict(lrs=lrs, momentum = momentum, T=T,  N_arch=N_arch, init_scale = init_scale, clip_val=clip_val, seed=seed, weight_decay=weight_decay, Optimize_C=Optimize_C, noise=noise)
    
    torch.save(dict(losses_stack=losses_all0, singular_values=sig_all0, singular_vectors=vec_all0, 
                    rho_all=rho_all,  x_batch_all=x_batch_all,
                    params = params), 
               file_name)
    
    return file_name, losses_all0