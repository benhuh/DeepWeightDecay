##################################
import torch
import torch.nn as nn
import numpy as np
# from torch.nn.utils import clip_grad_norm_  #, clip_grad_value_

from pdb import set_trace
from torch.nn.functional import linear, mse_loss

## Deep Linear Net for Natural Gradient descent 

class Multi_Linear(nn.Module):
    def __init__(self, Ws, C=None, Ny=1):
        super().__init__()
        self.C = nn.Parameter(C) if C is not None else None
        self.num_layers = len(Ws)
        self.module_list = nn.ModuleList([Linear(W) for W in Ws[::-1]])
        self.Ny = Ny
        
#     def forward_W(self, X):
#         return self.net_Weight() @ X
    
    def forward(self, X):
        
        for m in self.module_list:
            X = m(X)
        return  X

    def evaluate(self, X, Y, weight_decay): 
        task_batch, Ny, x_batch = Y.shape

        X = self.forward(X)
        if self.C is None:
            with torch.no_grad():
                C = self.adapt(X, Y, weight_decay)  # C_opt
        else:
            if self.C.norm() == 0:  # assuming zero-init..
                with torch.no_grad():
                    self.C.data = self.adapt(X, Y, weight_decay)  # C_opt at first only..
            C = self.C
        out = C @ X  #linear.apply(X, C) 
        loss = mse_loss(out, Y) #, reduction='sum')
        
        loss += weight_decay*(C**2).sum()/task_batch 
        return loss, C
    
    def adapt(self, X, Y, weight_decay): # returns C_opt
        weight_decay_ = weight_decay * Y.shape[-1]   # effective_eps = eps*x_batch 
        return torch.stack([ Yt @ pinv_eps(Xt, weight_decay_)   for Xt, Yt in zip(X, Y)], dim=0) #/ (task_batch * x_batch)
    
    def weight_loss(self):
        weight_losses = [m.weight_loss() for m in self.module_list]
        return sum(weight_losses)
    
    @torch.no_grad()
    def norm(self):
        return self.net_Weight().norm('fro')
    
    def nuclear_norm(self):
        return self.net_Weight().norm('nuc')
    
    def net_Weight(self):
        for i, m in enumerate(self.module_list): 
            netW = m.W  if i==0 else m.W @ netW
        return netW
    
    def gradient(self):
        return [m.W.grad for m in self.module_list]

    # def normalize_gradient(self, clip_val_):
    #     if clip_val_>0:
    #         clip_grad_norm_(self.parameters(), clip_val_)
        
    

class Linear(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.W = nn.Parameter(W)
    
    def forward(self, X):
        return self.W @ X  # linear(X, self.W)  # 
    
    def weight_loss(self):
        return self.W.norm()**2
        
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

