import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def get_weight_decay(sig, N):
    return ((N-2)/(N-1)*sig/2)**(2-2/N) / (1-2/N)

def sig_factor(sig,target,decay,N):
    return decay+sig**(1-2/N)*(sig-target)


class Deep_Linear_svd(nn.Module):
    def __init__(self, n_in = 5, depth=1, init_scale=1, device='cuda', lr=1, weight_decay=0, **kwargs): #n_data = None):
        super().__init__()
#         print('weight_decay', weight_decay, lr*depth)
        self.depth = depth  
        self.weight_decay=weight_decay #*depth*lr
        
        init_scale = init_scale**(1/depth)
        self.param_list = [nn.Parameter(init_scale*torch.ones(n_in).to(device = device) ) for _ in range(depth)]
        self.w_gt = torch.exp(torch.linspace(1,-5, n_in)).to(device = device)        
      
    @property
    def shared_weight(self):
        for i, p in enumerate(self.param_list):
            W = p if i==0   else p * W
        return W    

    def params(self, task_lr=None): #, dict_out=False):
        return self.param_list #self.parameters()    
    
    @property
    def e2e(self):
        return self.shared_weight
    
    @property
    def weight_loss(self):
        l=0
        for i, p in enumerate(self.param_list):
            l += p.pow(2).sum() #mean()
        return l
    
    def train_loss(self):
        return (self.w_gt - self.e2e).pow(2).sum() + self.weight_decay*self.weight_loss
    
    def test_loss(self):
        return self.train_loss()
      
class Deep_Linear_simple(nn.Module):
    def __init__(self, n_in = 5, depth=1, init_scale=1, device='cuda', lr=1, weight_decay=0, **kwargs): #n_data = None):
        super().__init__()
        n_out = n_in
        layers = [n_in] * (depth+1)
        module_list = [nn.Linear(f_in, f_out, bias=False) for (f_in, f_out) in zip(layers, layers[1:])]  

        self.depth = depth  
        self.weight_decay=weight_decay #*lr*depth
        self.shared_layer = nn.Sequential(*module_list)
        self.init_weights(init_scale**(1/depth)) #, _log)

        u, s, v = torch.rand(n_in, n_out).svd()
        s_ = torch.exp(torch.linspace(1,-5, min(n_in, n_out)))
        W = v * s_ @ u.t()
        self.w_gt = torch.tensor(W, device = device)        
        
    def params(self, task_lr=None): #, dict_out=False):
        return self.shared_layer.parameters()
        
    @property
    def shared_weight(self):
        for i, layer in enumerate(self.shared_layer.children()):
            W = layer.weight if i==0   else layer.weight @ W
        return W
    
    @property
    def weight_loss(self):
        l=0
        for i, layer in enumerate(self.shared_layer.children()):
            l += layer.weight.norm()**2
        return l
        
    
    @property
    def e2e(self):
        return self.shared_weight
    
    def train_loss(self):
        return (self.w_gt.T - self.e2e).pow(2).sum() + self.weight_decay*self.weight_loss
    
    def test_loss(self):
        return self.train_loss()
    
    ############################
    
    def init_weights(self, scale):
        for param in self.shared_layer.parameters():  #self.parameters():
            nn.init.eye_(param)
            param.data.mul_(scale)
        
        
class Deep_Linear(nn.Module):
    def __init__(self, config, n_in = None, n_out = None, n_feature = None, **kwargs): #n_data = None):
        super().__init__()
        
        n_out = n_out or n_in
        
        assert config.depth>0 
        n_hidden = n_in if config.wide else n_feature
        layers = [n_in] + [n_hidden]*(config.depth-1) + [n_out]  # n_out == n_task
        module_list = [nn.Linear(f_in, f_out, bias=False) for (f_in, f_out) in zip(layers, layers[1:])]  

        self.config = config
        self.depth = config.depth #- 1   if config.problem.name=='multi-task' else config.depth   #  and config.optimize_task        # self.depth = min(config.depth-1,1)
        assert self.depth > 0
        self.problem = get_problem(config)
        print(self.problem)

        if config.problem.name=='multi-task':
            self.shared_layer = nn.Sequential(*module_list[:-1])
            self.task_layer   = module_list[-1]
        else:
            self.shared_layer = nn.Sequential(*module_list)
            
        self.init_weights(layers) #, _log)

        
    def forward(self, X):
        return self.shared_layer(X)  # X @ self.shared_weight.T  # F.linear(X, self.shared_weight)
    
    def task(self, X):
        if len(X.shape)>2:
            return X @ self.task_layer.weight.unsqueeze(2) 
        else:
            return self.task_layer(X)

    def evaluate(self, X, Y, coeff):              # [n_task, n_data, n_y] = Y.shape
        X = self.forward(X)
        if self.config.optimize_task:
            self.adapt_task(X, Y, coeff)  
        out = self.task(X) 
        
        loss = F.mse_loss(out, Y) 
        
        if self.config.optimize_task or self.config.skip_task_layer:
            pass
        else:
            task_weight_loss = self.task_layer.weight.pow(2).sum()
            # task_weight_loss=task_weight_loss.detach()
            loss = loss + weight_decay * task_weight_loss / Y.shape[0]
        return loss
    
    def adapt_task(self, X, Y, coeff):
        with torch.no_grad():
            weight_decay_ = coeff*self.wd_task * Y.shape[1]   # effective_eps = eps*n_data
            task_weights =  torch.stack([ Yt.T @ pinv_eps(Xt.T, weight_decay_)   for Xt, Yt in zip(X, Y)], dim=0) 
        self.task_layer.weight.data = task_weights.data.squeeze()
        return None
        
        
    def params(self, task_lr=None): #, dict_out=False):
        if task_lr is None: #dict_out:
            if self.config.optimize_task or self.config.skip_task_layer:
                return self.shared_layer.parameters()
            else:
                return self.parameters()
        else:
            if self.config.optimize_task or self.config.skip_task_layer:
                return [{'params':  self.shared_layer.parameters()},]
            else:
                return [{'params':  self.shared_layer.parameters()},
                        {'params':  self.task_layer.parameters(), 'lr': task_lr},  ]

    def train_loss(self, coeff=1):
        if self.config.problem.name=='multi-task':
            return self.evaluate(self.problem.xs, self.problem.ys_, coeff)  + self.config.weight_decay * self.W_loss
        else:
            return self.problem.get_train_loss(self.e2e)    + self.config.weight_decay * self.W_loss
            
    
    def test_loss(self):
        return self.problem.get_test_loss(self.e2e)
    
    def d_e2e(self):
        return self.problem.get_d_e2e(self.e2e)
            
    
    @property
    def shared_weight(self):
        for i, layer in enumerate(self.shared_layer.children()):
            # assert isinstance(layer, nn.Linear) and layer.bias is None
            W = layer.weight if i==0   else layer.weight @ W
        return W
    
    @property
    def W_loss(self):
        l = 0
        for i, layer in enumerate(self.shared_layer.children()):
            l += layer.weight.norm('fro')**2
        return l
    
    @property
    def W_norm(self):
        return self.shared_weight.norm('fro').item()
    
    @property
    def e2e(self):
        if hasattr(self,'task_layer'):
            e2e = self.task_layer.weight @ self.shared_weight   # e2e = self.task_layer(self.shared_weight.T).T  
            # e2e.unsqueeze(2) == self.shared_weight.T @ self.task_layer.weight.unsqueeze(2)
            return e2e
        else: 
            return self.shared_weight
    
    @property
    def lr_factor(self):
#         if self.config.normalize and self.config.weight_decay>0:  # and self.config.optimize_task
#             raise ValueError
#             # return ((self.W_norm**2)**(1/self.depth))
#             return min(1e4/2, (self.W_norm**2/self.config.weight_decay)**(1/self.depth))
#             # return ((self.W_norm**2/self.shared_weight.numel()/self.config.weight_decay)**(1/self.depth))
#             # return ((1/self.config.weight_decay)**(1/self.depth))
#         else:
            return 1
            
    
    @property
    def wd_factor(self):
#         if self.config.normalize: # and self.config.optimize_task:
#             raise ValueError
#             # eps_ = self.config.weight_decay*self.W_norm**2
#             # wd = eps_ / (1+eps_)**2 /self.depth
#             # return wd/self.config.weight_decay
#             return (self.W_norm**2/self.depth)
#             # return (self.W_norm**2)
#         else:
            return 1 #*self.lr_factor

#     @property
#     def wd_task(self):
#         if self.config.normalize: # and self.config.optimize_task:
#             eps_ = self.config.weight_decay*self.W_norm**2
#             return eps_
#         else:
#             return self.config.weight_decay

    
#     @property
#     def wd_shared(self):
#         eps_ = self.wd_task
#         return eps_ / (1+eps_)**2 /self.depth
#         # return eps_ / self.depth
        
    ############################
    
    def init_weights(self, layers): #, _log):
        config = self.config
        initialization = config.initialization
        init_scale = config.init_scale
        depth = self.depth
        
        def check_e2e_norm():
            e2e = self.e2e #get_e2e(self).detach().cpu().numpy()
            e2e_fro = np.linalg.norm(e2e, 'fro')
            desired_fro = config.init_scale * np.sqrt(n)
            # _log.info(f"[check] e2e fro norm: {e2e_fro:.6e}, desired = {desired_fro:.6e}")
            assert 0.8 <= e2e_fro / desired_fro <= 1.2

        if initialization == 'orthogonal':
            matrices = []
            for param, n in zip(self.parameters(), layers):
                scale = (init_scale * np.sqrt(n))**(1/depth)
                nn.init.orthogonal_(param)
                param.data.mul_(scale)
                matrices.append(param.data.cpu().numpy())
            for a, b in zip(matrices, matrices[1:]):
                assert np.allclose(a.dot(a.T), b.T.dot(b), atol=1e-6)
        elif initialization == 'identity':
            assert config.wide
            scale = init_scale #init_scale**(1/depth)
            for param in self.shared_layer.parameters():  #self.parameters():
                nn.init.eye_(param)
                param.data.mul_(scale)
        elif initialization == 'gaussian':
            # assert layers[0] == layers[-1]
            for param, n in zip(self.parameters(), layers):
                scale = init_scale**(1/depth) * n**(-0.5)
                nn.init.normal_(param, std=scale)
            # check_e2e_norm()
        elif initialization == 'uniform':
            # assert layers[0] == layers[-1]
            for param, n in zip(self.parameters(), layers):
                scale = np.sqrt(3)*init_scale**(1/depth) * n**(-0.5)
                nn.init.uniform_(param, a=-scale, b=scale)
            # check_e2e_norm()
        else:
            raise ValueError(f'Invalid initialization: {initialization}')
        

###############################################
def pinv_eps(A, eps):  # epsilon regularized pinv
    u, s, v = A.svd()
    return v * (s/(s**2+eps)) @ u.t()

            
#################################################

def get_problem(config):
    PROBLEM_dict = { 'matrix-completion': MatrixCompletion,
                     'matrix-sensing': MatrixSensing,
                     'multi-task': MultiTask,
                     'SVD': SVD,
                     # 'ml-100k': MovieLens100k,
                    }
    return PROBLEM_dict[config.problem.name](device = config.device,  **vars(config.problem))

     
##################################


class PROBLEM(): 
    ys: torch.Tensor
    
    def __init__(self, device = None, n_in = None, n_out = None, n_feature = None, n_data = None, noise = 0, seed=0, **kwargs): 
        super().__init__()
        n_out = n_out or n_in
        
        # seed_everything(seed)
            
        self.device = device
        self.gen_GroundTruth(n_in, n_out, n_feature)
        
        with torch.no_grad():
            self.gen_obs(n_in, n_data, n_out) 
        if noise>0:
            self.apply_noise(noise)
                
    # _log.warning('[%s] Saved %d samples to %s', problem, n_data, obs_path)

    def gen_GroundTruth(self, n_in, n_out, n_feature): 
        U = np.random.randn(n_feature, n_in).astype(np.float32)
        A = np.random.randn(n_out, n_feature).astype(np.float32) 
        w_gt = A@U / np.sqrt(n_feature)
        w_gt = w_gt / np.linalg.norm(w_gt, 'fro') * np.sqrt(n_in * n_out)
        self.w_gt = torch.tensor(w_gt, device = self.device)
        
#     def gen_GroundTruth(self, n_in, n_out, n_feature): 
#         U = torch.randn(n_feature, n_in, device = self.device)
#         A = torch.randn(n_out, n_feature, device = self.device) 
#         w_gt = A@U / np.sqrt(n_feature)
#         w_gt = w_gt / w_gt.norm('fro') * np.sqrt(n_in * n_out)
#         self.w_gt = w_gt
        
        # oracle_sv = np.linalg.svd(w_gt, compute_uv=False)
        # lz.log.info("singular values = %s, Fro(w) = %.3f", oracle_sv[:r], np.linalg.norm(w_gt, ord='fro'))

    def apply_noise(self, noise):
        self.ys_ += noise*torch.randn_like(self.ys_)
        
    def gen_obs(self, n_in, n_data, n_out):
        pass
    def get_train_loss(self, e2e):
        pass
    def get_test_loss(self, e2e):
        pass
    def get_d_e2e(self, e2e):
        pass


##################################

  
class MatrixCompletion(PROBLEM):

    def gen_obs(self, n_in, n_data, n_out):
            
        indices = torch.multinomial(torch.ones(n_in * n_in), n_data, replacement=False)
        us, vs = indices // n_out, indices % n_out
        ys_ = self.w_gt.cpu().T[us, vs]
        assert 0.8 <= ys_.pow(2).mean().sqrt() <= 1.2
        self.us, self.vs, self.ys_ =  us.to(self.device), vs.to(self.device), ys_.to(self.device)

        
    def get_train_loss(self, e2e):
        self.ys = e2e[self.us, self.vs]
        return (self.ys - self.ys_).pow(2).mean()

    def get_test_loss(self, e2e):
        return (self.w_gt.T - e2e).pow(2).mean()

    # @CONFIG.inject
    def get_d_e2e(self, e2e):
        d_e2e = torch.zeros(self.w_gt.T.shape, device=self.w_gt.device)
        d_e2e[self.us, self.vs] = self.ys - self.ys_
        return d_e2e / len(self.ys_)

#     def get_cvx_opt_constraints(self, x, shape):
#         A = np.zeros(shape)
#         mask = np.zeros(shape)
#         A[self.us, self.vs] = self.ys_
#         mask[self.us, self.vs] = 1
#         eps = 1.e-3
#         constraints = [cvx.abs(cvx.multiply(x - A, mask)) <= eps]
#         return constraints



############################    

class MatrixSensing(PROBLEM):

    def gen_obs(self, n_in, n_data, n_out):
        xs = torch.randn(n_data, n_in, n_out, device = self.device) / np.sqrt(n_in * n_out)
        ys_ = (xs * self.w_gt.T).sum(dim=-1).sum(dim=-1)
        assert 0.8 <= ys_.pow(2).mean().sqrt() <= 1.2
        self.xs, self.ys_ = xs, ys_
        
    def get_train_loss(self, e2e):
        self.ys = (self.xs * e2e.T).sum(dim=-1).sum(dim=-1)
        return (self.ys - self.ys_).pow(2).mean()

    def get_test_loss(self, e2e):
#         print(self.w_gt.shape, e2e.shape)
#         import pdb; pdb.set_trace()
        return (self.w_gt - e2e).pow(2).mean()

    # @CONFIG.inject
    def get_d_e2e(self, e2e): 
        shape = self.w_gt.T.shape 
        d_e2e = self.xs.view(-1, *shape) * (self.ys - self.ys_).view(len(self.xs), 1, 1)
        d_e2e = d_e2e.sum(0)
        return d_e2e

    # def get_cvx_opt_constraints(self, X):
    #     eps = 1e-3
    #     constraints = []
    #     for x, y_ in zip(self.xs, self.ys_):
    #         constraints.append(cvx.abs(cvx.sum(cvx.multiply(X, x)) - y_) <= eps)
    #     return constraints

############################    


class SVD(PROBLEM): 
    ys: torch.Tensor
    
    def gen_GroundTruth(self, n_in, n_out, n_feature):
        W=torch.rand(n_in, n_out)
        u, s, v = W.svd()
        s_ = torch.exp(torch.linspace(1,-5, min(n_in, n_out)))
#         if symmetric:
#             W = v * s_ @ v.t()
#         else:
        W = v * s_ @ u.t()
        self.w_gt = torch.tensor(W, device = 'cuda')
#         self.w_gt.device = self.device
        
    def get_train_loss(self, e2e):
        return (self.w_gt.T - e2e).pow(2).sum()

    def get_test_loss(self, e2e):
        return (self.w_gt.T - e2e).pow(2).sum()

    def get_d_e2e(self, e2e):
        pass
    

############################  

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from Benchmark.utils import distance_U, distance_U_spectral
from MTLR.measurements2 import get_surrogate_loss2

class MultiTask(PROBLEM):
    

    # def gen_GroundTruth(self, n_in, n_out, n_feature): 
    #     self.U = np.random.randn(n_feature, n_in).astype(np.float32)
    #     nn.init.orthogonal_(self.U)
    #     A = np.random.randn(n_out, n_feature).astype(np.float32) 
    #     w_gt = A@U / np.sqrt(n_feature)
    #     w_gt = w_gt / np.linalg.norm(w_gt, 'fro') * np.sqrt(n_in * n_out)
    #     self.w_gt = torch.Tensor(w_gt) 
    
    def gen_GroundTruth(self, n_in, n_out, n_feature): 
        self.U = torch.randn(n_feature, n_in, device = self.device)
        nn.init.orthogonal_(self.U)
        self.A = torch.randn(n_out, n_feature, device = self.device)
        w_gt = self.A@self.U / np.sqrt(n_feature)
        w_gt = w_gt / w_gt.norm('fro') * np.sqrt(n_in * n_out)
        self.w_gt = w_gt
    
    def gen_obs(self, n_in, n_data, n_task):
        xs = torch.randn(n_task, n_data, n_in, device = self.device) / np.sqrt(n_in)   
        ys_ =  xs @ self.w_gt.unsqueeze(2)  # C0@W0@xs        
        self.xs, self.ys_ = xs, ys_
        
    # def get_train_loss(self, e2e):
    #     self.ys = self.xs @ e2e.unsqueeze(2)
    #     return (self.ys - self.ys_).pow(2).mean()

    def get_test_loss(self, e2e):
        return (self.w_gt - e2e).pow(2).mean()
    
    def get_d_e2e(self, e2e): 
        err = self.ys - self.ys_
        d_e2e = torch.einsum('txb,tb->tx', self.xs, err.squeeze()).T / self.xs.shape[-1]
        return d_e2e

    def distance_U(self, shared_weight):
        return distance_U(shared_weight.cpu().detach().numpy(),  self.U.cpu().detach().numpy())

    def get_surrogate_loss2(self, shared_weight, eps):
        return get_surrogate_loss2(shared_weight,  self.U, eps)
    

import random
def seed_everything(seed):
    if seed>0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        