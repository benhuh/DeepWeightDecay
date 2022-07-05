import torch
import numpy as np
from itertools import zip_longest


def set_seed_everywhere(seed):
    if seed>0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
#         os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
#         random.seed(seed)


def const_padding(hist, T):
    t_f = hist.shape[1]
    if t_f<T:
        padding = hist[:,-2:-1] * np.ones((1, T-t_f))
        return np.concatenate([hist, padding], axis=1)
    else:
        return hist[:,:T]


def get_arch_filename(Nx, Nctx, num_layer, wide, init_scale, lrs, Optimize_C, orth_C, eps, n_repeat, noise, x_batch_all, task_batch_factors):
    layer = f'_{num_layer}layer'
    width = '_wide' if wide else '_narrow'
    init = f'_init{init_scale}'
    lr = f'_lr{lrs[0]}'
    optim_C = '_optimC' if Optimize_C else ''
#     C0 = '_orthC0' if orth_C else '_randnC0'
    eps = f'_eps{eps}'
    # max_W = f'_maxW{max_W}' if max_W is not None else ''
    noise = f'_noise{noise}' if noise else ''
      
    N_arch = [Nctx]*(not wide)+[Nx]*(wide)+[Nx]*num_layer
    
    file_name = f'NewExp{layer}{width}{init}{lr}{optim_C}{eps}{noise}_{len(x_batch_all)}batches_{len(task_batch_factors)}_factors_{n_repeat}runs.save'
    return N_arch, file_name


def zip_list_padded(*lists):
    lists = [ l if isinstance(l, list) else [l]  for l in lists ]
    zipped_list=[]
    for [*ls] in zip_longest(*lists):
        zipped_list.append( [l or li[-1] for (l, li)  in zip(ls, lists)])
    return zipped_list
            
