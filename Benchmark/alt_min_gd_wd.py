from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import defaultdict
import torch

from utils import update_V, distance_U, distance_U_spectral

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from MTLR.MTLR_init import init_WC
from MTLR.model_simple import Multi_Linear
from MTLR.MTLR_plot import plot_all
from MTLR.MTLR_train_simple import GroupRMSprop, run_grad_descent
from MTLR.measurements2 import get_true_loss2, svd_analysis, get_surrogate_loss2
# from MTLR.measurements2 import measurements, svd_analysis, get_true_loss2 #, loss_True


def apply_alt_min_gd_wd(prob, steps,
                        lrs, momentum = (0,0), seed = 0,
                        weight_decay=0,  Optimize_C=True,
                        clip_val=0.1, early_stop=False, 
                        device="cpu", num_layer = 4, wide = True,
                        optim_type = 'RMS', 
                        init_scale = defaultdict(float, ortho=0.25, randn=0, W_0=0),
                        epoch_div=1, epoch_div_factor=100, writer = None, rho = 1, if_output_hist = False):
    x_batch = prob.m
    history = defaultdict(list)
    seeds = dict(task=0, data=0, model=0)
    if not wide:
        arch = [prob.r] + [prob.d]*num_layer # decide the architacture of the network
    else:
        arch = [prob.d] + [prob.d]*num_layer # decide the architacture of the network
    W, C = init_WC(prob.d, prob.r, prob.t, arch, init_scale = init_scale) # Ref
    W0 = torch.Tensor(prob.U).t()
    Xs, noise = (prob.X, prob.noise)
    Ys = np.einsum(
            'ir,dr,ijd->ij', prob.V, prob.U, prob.X)
    Ys = Ys + noise
    Xs, Ys = prob.get_torch_data(Xs, Ys)
    Xs = (Xs, None)
    Ys = (Ys, None)
    _ = run_grad_descent(W, C, W0, Xs, Ys, steps, history, lrs=lrs, momentum=momentum, weight_decay=weight_decay, clip_val=clip_val, Optimize_C=Optimize_C, optim_type=optim_type, device=device, epoch_div=epoch_div, epoch_div_factor=epoch_div_factor)
    out_history = np.array(history["losses"])
    out_history = out_history.T
    output = {
        'surrogate_loss_list': out_history[0, :],
        'avg_mse_loss_list': out_history[1, :],
        }
    if if_output_hist:
        return output, history
    else:
        return output
        
