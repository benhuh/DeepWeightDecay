import torch
import numpy as np
# from torch.linalg import pinv
from measurements2 import measurements, svd_analysis, get_true_loss2 #, loss_True
# from MTLR_gradient_manual import grad, nat_grad, nat_grad_sqrt

from model_NGD import Multi_Linear

import time
from pdb import set_trace

Dt = 100

#############################

def run_grad_descent(Ws, C, W0, C0, Xs, Ys, T, history, lrs, momentum = (0,0), 
                     q=0, eps=0, eps_W=0, eps_W_each=0, max_W=1,  eps_scaled = True, 
                     indep_W_decay=True,
                     Optimize_C = True, grad_C_amp = True,
                     clip_val=0.1, clip_val_C=0.1, noise=0, early_stop=True, 
                     cheat=False, device="cpu", norm_grad = True):

    
    assert eps_W==0 
    lr_W, lr_C = lrs;    W_mom, C_mom = momentum;   lr_W*=(1-W_mom); lr_C*=(1-C_mom);
    clip_val = torch.tensor(clip_val)
    time0=time.time()
    
    C = C if not Optimize_C else None
    model = Multi_Linear(Ws, q, C, Ny=1)
    model.to(device)
    # print(model.module_list[0].W.device, model.C.device)    
    # set_trace()

    if Optimize_C:
        optim = torch.optim.SGD(model.parameters(), lr=lr_W, momentum=W_mom)     
    else:
        optim = torch.optim.SGD( [{'params': model.module_list.parameters()}, {'params': model.C, 'lr': lr_C, 'momentum': C_mom}], lr=lr_W, momentum=W_mom) 
    
    for t in range(T):
        optim.zero_grad()
        losses, grad_norm = train_step(model, history, Xs, Ys, eps, eps_W_each, indep_W_decay, clip_val, cheat, W0, norm_grad)
        
        if early_stop and t>400: # and t>2*Dt:
            termination_cond = get_termination_cond(history['losses'][t-Dt:], losses, Dt, threshold=1e-6)
            if all(termination_cond):
                print('term_cond: true_loss, surrogate_loss:', termination_cond[0], termination_cond[1])
                break
        optim.step() 

    print(f'Time:{time.time()-time0}, Losses: train:{losses[2]}, eval:{losses[0]}, surrogate:{losses[1]}')  # , ', eval_LB:', losses[3]
    return None


###############################################

def train_step(model, history, Xs, Ys, eps, eps_W_each, indep_W_decay, clip_val, cheat, W0, norm_grad = True):
    W_norm=model.norm()
    normalizer = (W_norm**2/eps)**(1/model.num_layers)  
    eps_ = eps*W_norm**2;       
    eps_W_each_ = eps_W_each * eps_/((1+eps_)**2)  /model.num_layers
    # print(eps_W_each_)
    l_train, C_opt = evaluate(model, Xs, Ys, eps_, eps_W_each_ if not indep_W_decay else torch.tensor(0), cheat, W0)
    l_W = eps_W_each_ * model.weight_loss() 

    # l_train=l_train.detach()   # for debugging
    # if indep_W_decay:
    #     l_W = l_W.detach()  # skip l_W gradient 
    (l_train + l_W).backward()
    # l_train.backward()
    if norm_grad:
        model.normalize_gradient(normalizer, clip_val, eps_W_each_ if indep_W_decay else torch.tensor(0))
    
    losses, grad_norm = record_history(history, model, C_opt, W0, l_train, l_W, eps=eps, W_norm=W_norm, clip_val_=clip_val*normalizer)
    return losses, grad_norm
    
    
def evaluate(model, Xs, Ys, eps_, eps_W_each_, cheat, W0):
    X_train, X_test = Xs
    Y_train, Y_test = Ys
    
    # if Optimize_C:   
    l_train, C_opt = model.evaluate(X_train, Y_train, eps_, eps_W_each_)
    # else:
    #     assert X_test is None or X_test.numel()==0
    #     l_train, C_opt = model.evaluate(X_train, Y_train, eps_)

    if cheat>0: 
        with torch.no_grad():
            W = model.net_Weight()
        l_train = (1-cheat)*l_train + cheat*get_true_loss2(W, W0, eps_) 
                
    return l_train, C_opt

###############
def get_termination_cond(loss_history, losses, Dt, threshold=1e-6):
    loss_history = np.array(loss_history)
    true_loss_traj, surrogate_loss_traj = loss_history[:,0], loss_history[:,1]
    termination_cond_true      = np.linalg.norm(np.diff(np.log(true_loss_traj)))/Dt < threshold
    termination_cond_surrogate = np.linalg.norm(np.diff(np.log(surrogate_loss_traj)))/Dt < threshold 
    return (termination_cond_true, termination_cond_surrogate)  # or losses[1]<1e-5 or losses[0]<1e-5 or losses[2]<1e-6 

################

def record_history(history, model, C, W0, l_train, l_W, eps, W_norm, clip_val_):
    with torch.no_grad():
        W = model.net_Weight()
    grad_W = model.gradient()
    
    losses, grad_norm  = measurements(W, grad_W, W0, l_train.item(), 0, l_W.item(), eps=eps, W_norm=W_norm)
    
    sig, vec, proj, proj_norm = svd_analysis(W.cpu(), W0.cpu(), None, None) 
    
    history['losses'].append(losses)      
    history['C'].append(C.cpu().detach().clone().reshape(-1).numpy())
    history['W'].append(W.cpu().detach().clone().view(-1).numpy())
    history['grad_norm'].append(torch.stack(grad_norm+[clip_val_]).cpu().detach().numpy())  
    history['sing_val'].append(sig)
    history['sing_vec'].append(proj_norm.detach().numpy())
    
    return losses, grad_norm #, trueloss_log_grad


###############################################


# grad_fnc_dict = dict(grad_auto=grad_autodiff,
#                      grad=grad,
#                      nat_grad=nat_grad,
#                      nat_grad_sqrt=nat_grad_sqrt )


