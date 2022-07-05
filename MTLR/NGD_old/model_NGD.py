##################################
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_  #, clip_grad_value_

from pdb import set_trace


## Deep Linear Net for Natural Gradient descent 

class Multi_Linear(nn.Module):
    def __init__(self, Ws, q, C=None, Ny=1):
        super().__init__()
        # Ws = Ws + [C]
        self.C = nn.Parameter(C) if C is not None else None
        self.num_layers = len(Ws)
        self.module_list = nn.ModuleList([Linear(W, q) for W in Ws[::-1]])
        self.Ny = Ny
        self.q = q
        
#     def forward_W(self, X):
#         return self.net_Weight() @ X
    
    def forward(self, X, weight_decay):
        
        dummy = torch.zeros(X.shape[0], X.shape[1], self.Ny, device=X.device)
        for m in self.module_list:
            X, dummy = m(X, dummy, weight_decay)
        return  X, dummy

    def evaluate(self, X, Y, weight_decay): # eps_, eps_W_each_): 
        task_batch, Ny, x_batch = Y.shape

        X, dummy = self.forward(X, weight_decay)
        if self.C is None:
            with torch.no_grad():
                C = self.adapt(X, Y, weight_decay)  # C_opt
        else:
            print('huh?')
            C = self.C
        out, dummy = linear_fnc.apply(X, dummy, C, self.q, weight_decay) #  C_opt.detach())
        loss = mse_loss0.apply(out, dummy, Y)
        # loss = mse_loss_C.apply(X, dummy, Y, C_opt) 
        loss_reg = loss + weight_decay*(C.detach()**2).sum()/task_batch 
        return loss_reg, C
    
    def adapt(self, X, Y, eps): # returns C_opt
        eps__ = eps * Y.shape[-1]   # effective_eps = eps*x_batch 
        return torch.stack([ Yt @ pinv_eps(Xt, eps__)   for Xt, Yt in zip(X, Y)], dim=0) #/ (task_batch * x_batch)
    
    def weight_loss(self):
        weight_losses = [m.weight_loss() for m in self.module_list]
        return sum(weight_losses)
    
    @torch.no_grad()
    def norm(self):
        return self.net_Weight().norm('fro')
        
    def net_Weight(self):
        for i, m in enumerate(self.module_list): 
            netW = m.W  if i==0 else m.W @ netW
        return netW
    
    def gradient(self):
        return [m.W.grad for m in self.module_list]

    def normalize_gradient(self, normalizer, clip_val_, eps_W_each_):
        for m in self.module_list:
            # if indep_W_decay:
            m.W.grad += 2 * eps_W_each_ * m.W.data    # replaces l_W.backward()
            m.W.grad *= normalizer
            
        if clip_val_>0:
            clip_grad_norm_(self.parameters(), normalizer*clip_val_)
        
    

class Linear(nn.Module):
    def __init__(self, W, q):
        super().__init__()
        self.W = nn.Parameter(W)
        self.q = q
    
    def forward(self, X, dummy, weight_decay):
        return linear_fnc.apply(X, dummy, self.W, self.q, weight_decay)  # self.W @ X
    
    def weight_loss(self):
        return self.W.norm()**2
        # if len(self.W.shape)==3:             # multi-task C
        #     return self.W.norm()**2 / self.W.shape[0]  # divide by task_batch     
    
################
from torch.autograd import Function

# Ny=1

class linear_fnc(Function):
    @staticmethod
    def forward(ctx, input, dummy_in, weight, q=0, alpha0=1e-1): #1e-4):
        ctx.save_for_backward(input, weight, torch.tensor(q), torch.tensor(alpha0))     # input shape: [T, Nin, B]  # weight shape: [Nout, Nin]
        output = weight @ input          # + bias                 # output shape: [T, Nout, B]
        dummy_out  = torch.zeros(output.shape[0], output.shape[1], dummy_in.shape[-1], device=dummy_in.device)  #  [T, Nout, Ny=1]
        
        # assumed: L_W = eps_W_each*weight.norm()**2
        return output, dummy_out

    @staticmethod
    def backward(ctx, delta_out, Delta_out):  # Delta_out = C_opt shape: [T, Ny, Nx]
        input, weight, q, eps_W_each, alpha0 = ctx.saved_tensors
        
        weight_T = weight.T if len(weight.shape)==2 else weight.transpose(1,2)  
        delta_in = weight_T @ delta_out  # delta_out shape: [T, Nout, B]   ->  [T, Nin, B]  # Ny=1 
        Delta_in = weight_T @ Delta_out  # Haploid version: [T, Nout, Ny]  ->  [T, Nin, Ny]
        
        grad_weight = outer_prod(delta_out, input) if len(weight.shape)==2 else outer_prod2(delta_out, input)
        # if not indep_W_decay:
        grad_weight += 2 * eps_W_each * weight
        
        XX = outer_prod(input);        x_norm = XX.norm()/np.sqrt(XX.shape[0])
        DD = outer_prod(Delta_out);    D_norm = DD.norm()/np.sqrt(DD.shape[0])
        if q==0:        # SGD
            pass
            # grad_weight = grad_weight/x_norm/D_norm
        else:
            alpha=0 if q<=0.5 else  alpha0 # NGD regularizer            
            
            grad_weight = inv_alpha(DD, alpha, q) @ grad_weight @ inv_alpha(XX, alpha, q)
            # grad_weight = inv_alpha(DD/D_norm, alpha, q) @ grad_weight @ inv_alpha(XX/x_norm, alpha, q)  /x_norm/D_norm
            
        return delta_in, Delta_in, grad_weight, None, None

    
# class linear_fnc_C(Function):
#     @staticmethod
#     def forward(ctx, input, dummy_in, weight):
#         ctx.save_for_backward(input, weight)     # input shape: [T, Nin, B]  # weight shape: [Nout, Nin]
#         output = weight @ input          # + bias                 # output shape: [T, Nout, B]
#         dummy_out  = torch.zeros(output.shape[0], output.shape[1], dummy_in.shape[-1], device=dummy_in.device)  #  [T, Nout, Ny=1]
        
#         return output, dummy_out

#     @staticmethod
#     def backward(ctx, delta_out, Delta_out):  # Delta_out = C_opt shape: [T, Ny, Nx]
#         input, weight = ctx.saved_tensors
        
#         weight_T =  weight.transpose(1,2)  
#         delta_in = weight_T @ delta_out  # delta_out shape: [T, Nout, B]   ->  [T, Nin, B]  # Ny=1 
#         Delta_in = weight_T @ Delta_out  # Haploid version: [T, Nout, Ny]  ->  [T, Nin, Ny]
        
#         grad_weight = None
            
#         return delta_in, Delta_in, grad_weight, None, None


class mse_loss0(Function):
    @staticmethod
    def forward(ctx, input, dummy_in, target):        
        error = input - target            #  [T, Ny, B] 
        cost = (error**2).mean()          # .sum() / task_batch / x_batch / Ny 
        ctx.save_for_backward(error, )
        return cost

    @staticmethod
    def backward(ctx, delta_cost): 
        error,  = ctx.saved_tensors
        delta_in = (2 * error)              # [T, Nin, B] 
        Delta_in = np.sqrt(2) * torch.ones(error.shape[0], error.shape[1], 1, device=error.device)     # [T, Nin, Ny=1]

        return delta_in, Delta_in, None , None

# class mse_loss_C(Function):
#     @staticmethod
#     def forward(ctx, input, dummy_in, target, C_opt):        
#         error = C_opt @ input - target    # [T, Ny, Nin] @ [T, Nin, B] -> [T, Ny, B] 
#         cost = (error**2).mean()          # .sum() / task_batch / x_batch / Ny 
#         ctx.save_for_backward(error, C_opt)
#         return cost

#     @staticmethod
#     def backward(ctx, delta_cost): 
#         error, C_opt  = ctx.saved_tensors
#         C_opt_T = C_opt.transpose(1,2)                # [T, Ny, Nin] ->  [T, Nin, Ny]  # Ny=1 
#         delta_in = C_opt_T @ (error * 2)              # [T, Nin, Ny] @ [T, Ny, B] -> [T, Nin, B] 
#         Delta_in = C_opt_T * np.sqrt(2)  

#         return delta_in, Delta_in, None , None


##################
def outer_prod(X, Y=None):
    if Y is None:
        Y = X
    task_batch, _, x_batch = X.shape
    return torch.einsum('tib,tjb->ij', X, Y)  / task_batch / x_batch  

##################
def outer_prod2(X, Y):
    task_batch, _, x_batch = X.shape
    return torch.einsum('tib,tjb->tij', X, Y)  / x_batch  

# def some_prod(X, Y):
#     return torch.einsum('tob,tox->txb', X, Y)



###############################################
def pinv_eps(A, eps):  # epsilon regularized pinv
    u, s, v = A.svd()
    return v * (s/(s**2+eps)) @ u.t()

def inv_alpha(A, alpha, q=1):  # epsilon regularized pinv
    u, s, v = A.svd()
    return v * (s+alpha)**(-q) @ u.t()

