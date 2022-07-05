from pdb import set_trace
import numpy as np
import torch
# from MTLR_util import get_C_optim_all

from collections import defaultdict

from MTLR_utils import set_seed_everywhere

##############


def get_task_batch_arch(Nx, Nctx, x_batch, factor, Ny, num_layer, wide):
    task_batch = base_task_batch(Nx, Nctx, x_batch, factor=factor, Ny=Ny)
    N_arch = [Nctx]*(not wide)+[Nx]*(wide)+[Nx]*num_layer
    return task_batch, N_arch

def clone_W0_svd(W0, N_arch, W0_overlap, sig_noise=0.001):
    idx_start= W0.shape[0]-W0_overlap
    _,sig,v=W0.svd(some=False); 
    v_ = v.T[idx_start:idx_start+N_arch[0]]                   # partial W0_overlap with W0 
    sig = torch.cat((sig, torch.zeros(W0.shape[1] - sig.shape[0]))) 
    sig = sig[idx_start:idx_start+N_arch[0]] 
    # return get_ortho(N_arch[0],N_arch[0])@torch.diag( torch.ones(N_arch[0]) +sig_noise*torch.randn(N_arch[0]))@v_
    return get_ortho(N_arch[0],N_arch[0])@torch.diag( sig + sig_noise*torch.randn(N_arch[0]))@v_

def init_WC(Nx, Nctx, task_batch, N_arch=None, init_scale= defaultdict(float, ortho=0, randn=1, W_0=0), zeroW=False, zeroC=True, Ny=1, seed=None): #, W0=None, W0_overlap=None): #, identity_W=False):
    if seed is not None:
        set_seed_everywhere(seed)

    if N_arch==None:
        N_arch = [Nctx, Nx]
    else:
        assert isinstance(N_arch, list)
    
    rand_scale = init_scale['randn']**(1/(len(N_arch)-1))
    orth_scale = init_scale['ortho']**(1/(len(N_arch)-1))
    
    Os = [get_ortho(N_arch[i],N_arch[i]) for i in range(len(N_arch))]
    Ws = [rand_scale*torch.randn(N_arch[i],N_arch[i+1])/np.sqrt(Nx) +  orth_scale*Os[i][:,:min(N_arch[i],N_arch[i+1])]@Os[i+1][:,:min(N_arch[i],N_arch[i+1])].t()  for i in range(len(N_arch)-1)]
    # Ws = [ orth_scale*Os[i][:,:min(N_arch[i],N_arch[i+1])]@Os[i+1][:,:min(N_arch[i],N_arch[i+1])].t()  for i in range(len(N_arch)-1)]
    
#     if init_scale['W_0']>0:
#         # assert len(N_arch)==2
#         if N_arch[0]==Nx:
#             assert W0_overlap==W0.shape[0]
#         # Ws[0] += init_scale['W_0']*clone_W0_svd(W0, N_arch, W0_overlap)
#         Ws = [init_scale['W_0']*clone_W0_svd(W0, N_arch, W0_overlap)] + [init_scale['W_0']*torch.eye(N_arch[i+1])  for i in range(1, len(N_arch)-1)]
        
    # Ws = [rand_scale*torch.randn(N_arch[i],N_arch[i+1])/np.sqrt(Nx) +  Ws[i]  for i in range(len(N_arch)-1)]

    # C = init_scale*aggregate_orth(N_arch[0], task_batch*Ny).t().view(task_batch, Ny, N_arch[0])  
    C = torch.zeros(task_batch, Ny, N_arch[0])

    return Ws, C

###################################################################

def initialize_task_simple(Nx, Nctx, task_batch, x_batch_test, x_batch_train, Ny=1, augment=False, orth=dict(C=False, X=False), noise=0, device="cpu", seed_task=None, seed_data=None, single_task=False):
    if seed_task is not None:
        set_seed_everywhere(seed_task)

    # assert orth==dict(C=False, X=False)
    W0 = get_ortho(Nctx,Nx)
    
    if orth['C']:
        C0 = aggregate_orth(Nctx, task_batch*Ny).t().view(task_batch, Ny, Nctx)  
    else:
        C0 = torch.randn(task_batch, Ny, Nctx)
    C0 = normalize(C0, dim=-1)

    if seed_data is not None:
        set_seed_everywhere(seed_data)
    if single_task:
        X = torch.randn(1, Nx, x_batch_test+x_batch_train).repeat([task_batch,1,1])
    else:
        X = torch.randn(task_batch, Nx, x_batch_test+x_batch_train)
    X = normalize(X, dim=1)
    Xs, Ys = get_Xs_Yx(W0.to(device), C0.to(device), X.to(device), x_batch_train, augment, noise)
    return W0.to(device), C0.to(device), Xs, Ys

###################################################################
def get_W0(Nctx,Nx):
#     W0 = get_ortho(Nctx,Nx)
    W0_temp = get_ortho(Nx,Nx)
    W0, W0_ = W0_temp[:Nctx], W0_temp[Nctx:]  # shadow matrix
    return W0, W0_

def initialize_task(Nx, Nctx, task_batch, x_batch_test, x_batch_train, Ny=1, augment=False, orth=dict(C=False, X=False), noise=0, device="cpu"):
    
    W0, W0_ = get_W0(Nctx,Nx)
    
    if orth['C']:
        C0 = aggregate_orth(Nctx, task_batch*Ny).t().view(task_batch, Ny, Nctx)  
    else:
        C0 = torch.randn(task_batch, Ny, Nctx)
    C0 = normalize(C0, dim=-1)
        
    if orth['X']:
        X = aggregate_orth(x_batch_test+x_batch_train, task_batch*Nx).t().view(task_batch, Nx, x_batch_test+x_batch_train)  
#         X = torch.stack([aggregate_orth(x_batch_test+x_batch_train, Nx).t() for _ in range(task_batch)], dim=0) 
#         X = torch.stack([aggregate_orth(Nx, x_batch_test+x_batch_train) for _ in range(task_batch)], dim=0) 
        X*=torch.sqrt(Nx) # ??
    else:
        X = torch.randn(task_batch, Nx, x_batch_test+x_batch_train)
        X = normalize(X, dim=1)

    Xs, Ys = get_Xs_Yx(W0.to(device), C0.to(device), X.to(device), x_batch_train, augment, noise)
    
    return W0.to(device), C0.to(device), Xs, Ys

##############

def initialize_task_best(Nx, Nctx, task_batch, x_batch_test, x_batch_train, Ny=1, augment=False, orth=False, noise=0):
    dk = x_batch_train-Nctx
    T1 = int((Nx-Nctx)/dk) 
    T2 = int(Nctx/Ny)
    assert x_batch_test==0
    assert (Nx-Nctx)%dk == 0
    assert task_batch == int(T1*T2)  # factor==1
    
    W0, W0_ = get_W0(Nctx,Nx)
    
    X0 = W0  
    X_rests = W0_.split(dk)  # v[Nctx:].split(dk)  # should be of same lengths
    
    assert len(X_rests) == T1
    
    X = torch.stack([torch.cat([X0, X_]).t() for X_ in X_rests], dim=0)
    X = torch.tile(X, (T2,1,1))
    
    C0 = torch.tile(get_ortho(T2*Ny,Nctx).unsqueeze(1), (1,T1,1)).view(T1*T2,Ny,-1)  
    
    # Huh? correct normalization?
    X*=np.sqrt(Nx)
    C0*=np.sqrt(Nctx)
    
    Xs, Ys = get_Xs_Yx(W0, C0, X, x_batch_train, augment, noise)
    return W0, C0, Xs, Ys


def initialize_task_better(Nx, Nctx, task_batch, x_batch_test, x_batch_train, augment=False, orth=False, noise=0):
    W0, _, Xs = initialize_task_best(Nx, Nctx, task_batch, x_batch_test, x_batch_train, augment, orth)
    C0 = aggregate_orth(Nctx, task_batch).t().unsqueeze(1)   
    return W0, C0, Xs, Ys

def initialize_task_eye(Nx, Nctx, task_batch, x_batch_test, x_batch_train, augment=False, orth=False, noise=0):
    dk = x_batch_train-Nctx
    T1 = int((Nx-Nctx)/dk) 
    T2 = Nctx
    assert x_batch_test==0
    assert (Nx-Nctx)%dk == 0
    assert task_batch == int(T1*T2)  # factor==1
    

    eye = torch.eye(Nx,Nx)
    W0 = eye[:Nctx]
    
#     u,s,v=W0.svd(some=False)
    X0 = eye[:Nctx]
    X_rests = eye[Nctx:].split(dk)  # should be of same lengths
    
    assert len(X_rests) == T1
    
    X = torch.stack([torch.cat([X0, X_]).t() for X_ in X_rests], dim=0)
    X = torch.tile(X, (T2,1,1))
    
    C0 = torch.tile(torch.eye(T2,Nctx).unsqueeze(1), (1,T1,1)).view(T1*T2,1,-1)  
    
    Xs, Ys = get_Xs_Yx(W0, C0, X, x_batch_train, augment, noise)
    return W0, C0, Xs, Ys


def initialize_task_worst(Nx, Nctx, task_batch, x_batch_test, x_batch_train, augment=False, orth=False, noise=0):
    assert x_batch_test==0
    dk = x_batch_train-Nctx
    assert (Nx-Nctx)%dk == 0
    
    W0 = get_ortho(Nctx,Nx)
    
    C0 = aggregate_orth(Nctx, task_batch).t().unsqueeze(1)  
    
    u,s,v=W0.svd(some=False)
    X0 = v[:Nctx]
    X_rests = v[Nctx:].split(dk)  # should be of same lengths
    
#     X = torch.stack([torch.cat([X0, X_]).t() for X_ in X_rests], dim=0)
    X = torch.stack([torch.cat([X0+X_rests[0], X_rests[1]]).t() , torch.cat([X0+X_rests[1], X_rests[0]]).t()   ], dim=0)
    Xs, Ys = get_Xs_Yx(W0, C0, X, x_batch_train, augment, noise)
    return W0, C0, Xs, Ys



##############################


def task_batch_min(Nx, Nctx, x_batch, Ny=1):
    if isinstance(x_batch,list):
        x_batch=np.array(x_batch)
    task_batch = (Nctx/Ny) *(Nx-Nctx) / (x_batch - Nctx)  
    return task_batch

def base_task_batch(Nx, Nctx, x_batch, factor=1, Ny=1):
    task_batch = task_batch_min(Nx, Nctx, x_batch, Ny)
    # return int(np.floor(factor*task_batch))
    return int(np.ceil(factor*task_batch))
#     return int(factor*task_batch)

def get_ortho(n0, n1):
    return torch.nn.init.orthogonal_(torch.empty(n0,n1))

def initialize(x_batch, rho, seeds):
    task_batch, N_arch = get_task_batch_arch(Nx, Nctx, x_batch, rho, Ny, num_layer, wide)
    x_batch_train, x_batch_test = x_batch, 0
    W0, C0, Xs, Ys = initialize_task_simple(Nx, Nctx, task_batch, x_batch_test, x_batch_train, Ny=Ny, augment=augment, orth=orth, noise=noise, device=device, seed_task=seeds['task'], seed_data=seeds['data'])
    W, C = init_WC(Nx, Nctx, task_batch, N_arch, init_scale, zeroW = zeroW, zeroC= zeroC, Ny=Ny, seed=seeds['model']) #, W0=W0, W0_overlap=W0_overlap)         
    print_setting1(x_batch, rho) 
    return W0, C0, Xs, Ys, W, C

def aggregate_orth(n0, n1):  #     assert  n1 > n0 (typically)
    M = [get_ortho(n0, n0) for _ in range(n1//n0)]
    if n1%n0>0:
        M += [get_ortho(n0, n1%n0)]
    return torch.cat(M, dim=1)

def normalize(mat, dim=0):
    return mat * (mat.shape[dim]/ (mat**2).sum(dim=dim, keepdim=True)).sqrt()
#########################################

def get_Xs_Yx(W0, C0, X, x_batch_train, augment, noise):
    Y = C0@W0@ X
    z = torch.randn_like(Y)
    Y = Y+noise*z if noise>0 else Y*(1+noise*z)  # additive or multiplicative
    Xs = separate_Xtrain_Xtest(X, x_batch_train, augment)
    Ys = separate_Xtrain_Xtest(Y, x_batch_train, augment)
    
#     print('X_train norm:', Xs[0].norm()**2/Xs[0].numel(), ', Y_train norm', Ys[0].norm()**2/Ys[0].numel())
    return Xs, Ys

def separate_Xtrain_Xtest(X, x_batch_train, augment):
    X_train, X_test = X[:,:,:x_batch_train], X[:,:,x_batch_train:]
    if augment:
        X_all = torch.cat([X_test, X_train], dim=-1);  
        X_train = X_test = X_all
    return X_train, X_test

