import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_  #, clip_grad_value_
from torch.nn.functional import linear, mse_loss

# from torch.linalg import pinv
from measurements2 import measurements, svd_analysis, get_true_loss2 #, loss_True
# from MTLR_gradient_manual import grad, nat_grad, nat_grad_sqrt

# from model_NGD import Multi_Linear
from model_simple import Multi_Linear
# from MTLR_utils import set_seed_everywhere

import time
from pdb import set_trace

Dt = 100

#############################

def run_grad_descent(Ws, C, W0, Xs, Ys, T, history, lrs, momentum = (0,0), 
                     weight_decay=0,  nuclear_decay = 0, Optimize_C=True,
                     clip_val=0.1, early_stop=True, 
                     device="cpu", 
                     optim_type = 'SGD',
                     epoch_div=1, epoch_div_factor=100, weight_decay_min=1e-6,
                     Xe = None, Ye = None
                     ):
    """run gradiant descent

    Args:
        Ws (torch.Tensor): model parameters
        C (torch.Tensor): true task specific parameters
        W0 (torch.Tensor): true representation parameters
        Xs (torch.Tensor): training X
        Ys (torch.Tensor): training Y
        T (int): total steps
        history (list): history recorder
        lrs (tuple): learning rate e.g. (0.1, 0.1)
        momentum (tuple, optional): _description_. Defaults to (0,0).
        weight_decay (int, optional): _description_. Defaults to 0.
        nuclear_decay (int, optional): _description_. Defaults to 0.
        Optimize_C (bool, optional): _description_. Defaults to True.
        clip_val (float, optional): _description_. Defaults to 0.1.
        early_stop (bool, optional): _description_. Defaults to True.
        device (str, optional): _description_. Defaults to "cpu".
        optim_type (str, optional): _description_. Defaults to 'SGD'.
        epoch_div (int, optional): _description_. Defaults to 1.
        epoch_div_factor (int, optional): _description_. Defaults to 100.
        weight_decay_min (_type_, optional): _description_. Defaults to 1e-6.
        Xe (torch.Tensor, optional): validation dataset. Defaults to None.
        Ye (torch.Tensor, optional): validation dataset. Defaults to None.
    """

    lr_W, lr_C = lrs;    W_mom, C_mom = momentum;   lr_W*=(1-W_mom); lr_C*=(1-C_mom)
    clip_val = torch.tensor(clip_val)
    time0=time.time()
    
    model = Multi_Linear(Ws, None if Optimize_C else C, Ny=1)
    model.to(device)
     
    optimizer = torch.optim.SGD if optim_type == 'SGD' else GroupRMSprop
#     print(optim_type, optimizer)

    if Optimize_C:
        optim = optimizer(model.parameters(), lr=lr_W/model.num_layers, momentum=W_mom)     
    else:
        # lr_C = 4*lr_W;    C_mom = W_mom
        optim = optimizer([{'params': model.module_list.parameters()}, {'params': model.C, 'lr': lr_C, 'momentum': C_mom}], lr=lr_W/model.num_layers, momentum=W_mom)
    
    for t in range(T):
        p = t*epoch_div//T
        wd = max(weight_decay/epoch_div_factor**p, weight_decay_min)
        optim.zero_grad()
        losses, grad_norm = train_step(model, history, Xs, Ys, wd, nuclear_decay, clip_val, W0)
        if W0 is None:
            l_test = test_evaluate(model, (Xe, None), (Ye, None), wd, 2)
            l_train = test_evaluate(model, Xs, Ys, wd, 2)
            # l_test, _ = evaluate(model, (Xe, None), (Ye, None), wd)
            # l_train, _ = evaluate(model, Xs, Ys, wd)
            
            history['test_mse'].append(l_test.cpu().detach().clone().view(-1).numpy())
            history['train_mse'].append(l_train.cpu().detach().clone().view(-1).numpy())
        
        if early_stop and t>400 and W0 is not None: # and t>2*Dt:
            termination_cond = get_termination_cond(history['losses'][t-Dt:], losses, Dt, threshold=1e-6)
            if termination_cond:
                print('term_cond: surrogate_loss:', termination_cond)
            # if any(termination_cond):
            #     print('term_cond: true_loss, surrogate_loss:', termination_cond[0], termination_cond[1])
                break
        optim.step() 

    # print(f'Time:{time.time()-time0}, Losses: train:{losses[2]}, eval:{losses[0]}, surrogate:{losses[1]}')  # , ', eval_LB:', losses[3]
    print(f'Time:{time.time()-time0}, Losses: train:{losses[1]}, surrogate:{losses[0]}')  # , ', eval_LB:', losses[3]
    return None


###############################################
    

def train_step(model, history, Xs, Ys, weight_decay, nuclear_decay, clip_val, W0):
    W_norm=model.norm()
    l_train, C_opt = evaluate(model, Xs, Ys, weight_decay) 
    l_W = weight_decay * model.weight_loss() if nuclear_decay==0 else nuclear_decay*model.nuclear_norm()

    (l_train + l_W).backward()

    if clip_val>0:
        clip_grad_norm_(model.parameters(), clip_val)
    
    losses, grad_norm = record_history(history, model, C_opt, W0, l_train, l_W, weight_decay=weight_decay, W_norm=W_norm, clip_val_=clip_val)
    return losses, grad_norm
    
def test_evaluate(model, Xs, Ys, weight_decay, r):
    X_train, _ = Xs
    Y_train, _ = Ys
    # l_train, C_opt = model.evaluate(X_train[:, :, :(2*r)], Y_train[:, :, :(2*r)], weight_decay)
    # out = C_opt @ X_train[:, :, (2*r):]  #linear.apply(X, C) 
    # loss = mse_loss(out, Y_train[:, :, (2*r):]) #, reduction='sum')
    l_train, C_opt = model.evaluate(X_train[:, :, :(2*r)], Y_train[:, :, :(2*r)], weight_decay)
    out = C_opt @ model.forward(X_train[:, :, (2*r):])  #linear.apply(X, C) 
    loss = mse_loss(out, Y_train[:, :, (2*r):]) #, reduction='sum')
    return loss
    
def evaluate(model, Xs, Ys, weight_decay):
    X_train, X_test = Xs
    Y_train, Y_test = Ys
    
    l_train, C_opt = model.evaluate(X_train, Y_train, weight_decay)
    return l_train, C_opt

###############
def get_termination_cond(loss_history, losses, Dt, threshold=1e-6, threshold2=3e-7):
    loss_history = np.array(loss_history)
    # true_loss_traj, surrogate_loss_traj = loss_history[:,0], loss_history[:,1]
    surrogate_loss_traj, train_loss_traj,  = loss_history[:,0], loss_history[:,1]
    # termination_cond_true      = np.linalg.norm(np.diff(np.log(true_loss_traj)))/Dt < threshold
    termination_cond_surrogate = surrogate_loss_traj[-1] < threshold2
    # termination_cond_surrogate = np.linalg.norm(np.diff(np.log(surrogate_loss_traj)))/Dt < threshold 
    # return (termination_cond_true, termination_cond_surrogate)  # or losses[1]<1e-5 or losses[0]<1e-5 or losses[2]<1e-6 
    return (termination_cond_surrogate)  # or losses[1]<1e-5 or losses[0]<1e-5 or losses[2]<1e-6 

################

def record_history(history, model, C, W0, l_train, l_W, weight_decay, W_norm, clip_val_):
    with torch.no_grad():
        W = model.net_Weight()
    grad_W = model.gradient()
    
    # for testing loss calculation
    get_formard_X = lambda X: model.forward(X)
    get_opt_V = lambda X, Y: np.mean(model.adapt(get_formard_X(X), Y, torch.tensor(weight_decay)).detach().numpy(), 1)
    
    losses, grad_norm, test_loss  = measurements(W, grad_W, W0, l_train.item(), 0, l_W.item(), weight_decay=weight_decay, W_norm=W_norm, get_opt_V = get_opt_V)
    
    sig, vec, proj, proj_norm = svd_analysis(W.cpu(), W0.cpu() if W0 is not None else W0, None, None) 
    
    history['losses'].append(losses)
    history['test_loss'].append(test_loss)
    history['C'].append(C.cpu().detach().clone().reshape(-1).numpy())
    history['W'].append(W.cpu().detach().clone().view(-1).numpy())
    # history['grad_norm'].append(torch.stack(grad_norm+[clip_val_]).cpu().detach().numpy())  
    history['grad_norm'].append(torch.stack(grad_norm).cpu().detach().numpy()) #[clip_val_]))      
    history['sing_val'].append(sig)
    history['sing_vec'].append(proj_norm.detach().numpy())
    
    return losses, grad_norm #, trueloss_log_grad


###############################################


# grad_fnc_dict = dict(grad_auto=grad_autodiff,
#                      grad=grad,
#                      nat_grad=nat_grad,
#                      nat_grad_sqrt=nat_grad_sqrt )


from torch.optim.optimizer import Optimizer

class GroupRMSprop(Optimizer):
    """A different version of RMSprop optimizer with a global learning rate adjusting.
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-6, weight_decay=0, momentum=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, alpha=alpha, eps=eps, adjusted_lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            state = self.state
            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['square_avg'] = torch.tensor(0., device=group['params'][0].device)
        
            alpha = group['alpha']
            lr = group['lr'] #/ avg
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            square_avg = state['square_avg']
            square_avg.mul_(alpha)

            state['step'] += 1

            for p in group['params']:
                state_p = self.state[p]
                
                if len(state_p) == 0:   # state_p initialization
                    if momentum > 0:
                        state_p['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)                
                
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('GroupRMSprop does not support sparse gradients')
                square_avg.to(grad.device)
                square_avg.add_((1 - alpha) * grad.pow(2).sum())#.cpu().float())
                # square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

            avg = square_avg.div(1 - alpha**state['step']).sqrt_().add_(eps).to(p.device)
            # lr = group['lr'] #/ avg
            group['adjusted_lr'] = lr / avg

            for p in group['params']:
                if p.grad is None:
                    continue
                
                
                grad = p.grad.data
                
                wd = lr / avg * weight_decay
#                 adjusted_lr = lr / avg
#                 p_ = p - wd*p - adjusted_lr*grad

                if weight_decay != 0:
#                     grad = grad.add(p, alpha = -wd)
                    p.data.mul_(1 - wd)
#                     p.data.mul_( torch.exp(- wd))

                # avg = square_avg.sqrt().add_(eps)
                if momentum > 0:
                    buf = self.state[p]['momentum_buffer']
                    buf.mul_(momentum).addcdiv_(grad, avg)
                    p.data.add_(buf, alpha=-lr)
                else:
                    # p.data.add_(-lr.to(grad.device) * grad)
                    p.data.addcdiv_(grad, avg, value=-lr)                
                    
#                 print('p-p_', p-p_)
#                 import pdb; pdb.set_trace()

        return loss