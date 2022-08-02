import torch
from pdb import set_trace
import numpy as np
from MTLR.MTLR_init import normalize
###########################


def measurements(W, grad_W, W0, l_train, l_test, l_W, weight_decay, W_norm, get_opt_V = None):
    
    lW_coeff=0
    # lW_coeff=1
    
    eps_ = weight_decay #eps*W_norm**2
    factor = 1 #2*eps_                 # for W0_overlap = 4  -> min -loss = 4*eps_
    # factor = torch.tensor(eps)    # for W0_overlap = 0  -> constant loss (4)
    
    sig_threshold = 1e-6 #weight_decay #1e-6
    if W0 is not None:
        trueloss =  get_true_loss2(W, W0, eps=eps_) 
        testloss = test_mse_loss(W, W0, get_opt_V = get_opt_V).reshape(-1)
        surrogate_loss = get_surrogate_loss2(W, W0, sig_threshold).item()/factor
        lower_bound = get_true_loss_lower_bound(W, W0, eps=eps_)
    else:
        trueloss = None
        testloss = None
        surrogate_loss = None
        lower_bound = None
        
    # set_trace()
    losses = np.array([
                        # (trueloss.item() + lW_coeff*l_W)/factor, 
                          surrogate_loss,
                          (l_train + lW_coeff*l_W )/factor,  #*W_norm**2, #
                          # (lower_bound.item() + lW_coeff*l_W)/factor, 
                          # l_W/factor,
                          # W.norm().detach(),
                          weight_decay,  #torch.tensor(eps)
                         ])
    
    assert isinstance(grad_W,list)
    grad_norm = [sum([g.reshape(-1).norm()**2 for g in grad_W]).sqrt()]
    
    
    return losses, grad_norm, testloss # , trueloss_log_grad #, sig, proj_norm


###########################
# new code for testing loss
def init_V(t, r):
    V = torch.randn(t, 1, r)
    V = normalize(V, dim=-1).detach().numpy()
    V = np.mean(V, 1)
    return V
def get_torch_data(Xs, Ys):
    Xs = Xs.swapaxes(1, 2)
    Ys = np.expand_dims(Ys, 1)
    Ys = torch.from_numpy(Ys)
    Xs = torch.from_numpy(Xs)
    Xs,Ys=Xs.float(),Ys.float()
    return Xs, Ys
def test_mse_loss(W, W0, m = None, t = 30, get_opt_V = None):
    return np.array([0])
    W0 = W0.t().detach().numpy()
    W = W.t().detach().numpy()
    d, r = W0.shape # check
    V_news = init_V(t, r)
    X = np.random.normal(size=[t, r, d]) # r samples for each task
    y = np.einsum(
        'ir,dr,ijd->ij', V_news, W0, X)
    X_t, y_t = get_torch_data(X, y)
    V_est = get_opt_V(X_t, y_t)
    regressors = np.einsum(
        'ir,dr->id', V_news, W0)
    regressors_est = np.einsum(
        'ir,dr->id', V_est, W)
    return ((regressors - regressors_est)**2).mean()
    

def get_true_loss2(W, W0, eps = 0.0):  # agress with get_true_loss0 but noisier
    Q = W@W0.t()
    if Q.norm()**2>eps/100:
        u, s, v = Q.svd()  #     Q = u * s @ v.t()
        s_pinv = s/(s**2 + eps)
        Q_pinv = v * s_pinv @ u.t()
        return  (W0 - Q_pinv @ W).norm()**2 + eps* (s_pinv**2).sum()  
    else:
        Q_pinv = Q.T@(Q@Q.T + eps*torch.eye(W.shape[0])).inverse()
        return  (W0 - Q_pinv @ W).norm()**2 + eps* Q_pinv.norm()**2  # W0.shape[0] - 2*(Q_pinv@Q).trace() + (Q_pinv@W).norm()**2 + eps* Q_pinv.norm()**2eps*Q_pinv.norm()**2
    # QQ = Q@Q.T;     WW = W@W.T;    RR = WW-QQ;     
    # grad_W =  -eps*M@( M@QQ + (eps*(M@RR + RR@M) -2*RR)@M )  # = gradient @ W.T

def get_true_loss_lower_bound(W, W0, eps = 0.0):  # full observation adaptation
    _, sig, v = W.svd()
    sig_I_PW = eps / (sig**2 + eps)   # (I - PW).svd()
    
    ps = W0 @ v * sig_I_PW.sqrt()
    return ps.norm()**2

# def get_true_loss_lower_bound(W, W0, eps = 0.0):  # full observation adaptation
#     _, sig, v = W.svd()
#     sig_I_PW =  sig**2 / (sig**2 + eps)   # soft orthogonalized W: OW
    
#     # ps = W0 @ v * sig_I_PW.sqrt()
#     err = W0 - W0 @ OW.T
#     return err.norm()**2


def get_surrogate_loss2(W, W0, eps = 0.0):
    _, sig, v = W.svd()
    sig_PW = sig**2 / (sig**2 + eps)  
    WW_normalized = v * sig_PW @ v.T
    err = WW_normalized - W0.T@W0
    surrogate_loss = (err.norm()**2/  err.numel()).sqrt()
    return surrogate_loss

def get_surrogate_loss3(W, W0, eps = 0.0):
    _, sig, v = W.svd()
    sig_PW = sig**2 / (sig**2 + eps)  
    WW_normalized = v * sig_PW.sqrt() @ v.T
    err = WW_normalized - W0.T@W0
    surrogate_loss = err.norm()**2/  err.numel()
    return surrogate_loss

#######################################

def get_true_loss3(W, W0, W0_, eps = 0.0):  # agress with get_true_loss0 but noisier
    Q = W@W0.t()
    Q_ = W@W0_.t()
    u, s, _ = Q.svd()
    s0_inv = 1/(s**2 + eps)
    
    I_PQ = (eps*s0_inv).sum();    PQ2 = ((s**2*s0_inv)**2).sum();    Q_pinv_W = (s*s0_inv) @ u.t() @ W
    return  I_PQ - PQ2 + Q_pinv_W.norm()**2
#     QQ_ = Q_ @ Q_.t() ;    WW = W@ W.t() ;    QQ = Q@Q.t();    # QQ_ = WW-QQ
#     Q_pinv_Q_ =  s*s0_inv @ u.t() @ Q_;     Q_pinv_W =  s*s0_inv @ u.t() @ W;     PQ2 = ((s**2*s0_inv)**2).sum();  # Q_pinv_Q_.norm()**2 = Q_pinv_W.norm()**2 - PQ2
#     return  (eps*s0_inv).sum() + ((s*s0_inv) @ u.t() @ Q_).norm()**2 


# def get_surrogate_loss2(sig, proj_norm):
#     p = sig**2/ (sig**2).sum() 
#     surrogate_loss = (p @ (1 - proj_norm**2) ).sum()
#     return surrogate_loss
#     err = u@ sig @(torch.eye(W0.shape[1]) - W0.t()@W0)@ sig @ u.t()
#     err = sig**2 @ v.t()@(torch.eye(W0.shape[1]) - W0.t()@W0) @ v
#     err = sig**2 @ (torch.eye(W0.shape[1]) - proj.t()@proj) 
#     err.trace() = (sig**2 @ (1 - proj_norm**2) ).sum()
#     surrogate_loss = err.trace()/ sig**2.sum() 
#     surrogate_loss = (sig**2 @ (1 - proj_norm**2) ).sum()/ sig**2.sum() 
#     return surrogate_loss



# def get_surrogate_loss(W, W0):
#     err = W@(torch.eye(W0.shape[1]) - W0.t()@W0)@ W.t()
#     surrogate_loss = err.trace()/ W.norm('fro')**2  
#     return surrogate_loss




# def get_true_loss0(W, W0, eps=0): #, use_only=None):
#     Q = W@W0.t()
#     u, s, v = get_svd_small(Q)
#     s0_inv = 1/(s**2 + eps)
#     s1 = s**2 *s0_inv #= (1 - eps*s0_inv)
    
# #     s2 = (s*s0_inv)**2    
# #     M = u * s2 @ u.t()
# #     term2 = W.t()@M@W
#     s2_sqrt = s*s0_inv
#     Wu =  W.t() @ u * s2_sqrt
#     term2 = Wu.t() @ Wu  # = Wu @ Wu.t()
    
#     term3 = s1**2 + s1  # = 2*s1 - eps*s2

#     trueloss0 =   W0.shape[0]  + term2.trace() - term3.sum() 
#     return trueloss0 


#################


def loss_True(W, U, eps):  # agress with get_true_loss0 but noisier
    # W.grad=None;  
    
    I2 = torch.eye(U.shape[0])
    I3 = torch.eye(W.shape[0])

    Q = W@U.t()
    QQ = Q@Q.T  
    WW = W@W.T
    RR = WW - Q@Q.T  
    
    if Q.norm()**2>eps/100:
        u, s, v = Q.svd() 
        Q_pinv = v * (s/(s**2 + eps)) @ u.T      

        # u, s, v = Q.svd(some=False); s = torch.cat((s, torch.zeros(Q.shape[0] - s.shape[0]))) 
        # M = u * (1/(s**2 + eps)) @ u.T   
    else:
        Q_pinv = Q.T@(Q@Q.T + eps*I3).inverse()
        # M = (QQ + eps*I3).inverse()
        
    P = Q_pinv@Q
    
    QRRQ = Q_pinv@RR@Q_pinv.T 
    
    L0 =(I2 - P).trace()/2 
    L_QRRQ = QRRQ.trace()/2  
    Lw = eps/(1+eps)**2*W.norm()**2/2
    
    print('L0', L0.data/eps,'L_QRRQ', L_QRRQ.data/eps, 'Lw', Lw.data/eps, 'eps', eps)
    L=L0+L_QRRQ+Lw

    return L


##################################################

def svd_analysis(W, W0, sig_prev = None, vec_prev = None):
    try:
        u, sig, vec = W.svd()
        sig = sig.detach().numpy()
        vec = vec.detach().numpy()
        # if sig_prev is not None: 
        #     sig, vec = track_SVD_order(sig, vec, sig_prev, vec_prev)
        if W0 is not None:
            proj = W0 @ vec   
            proj_norm = (proj**2).sum(dim=0).sqrt()
        else:
            proj = None
            proj_norm = torch.Tensor(0)
        return sig, vec, proj, proj_norm
    except:
        return [0], 0, 0, 0
        

# import numpy as np
# from munkres import Munkres
# m = Munkres()

# def track_SVD_order(sig, vec, sig_prev, vec_prev):
#     # compute distance between systems
#     D1, D2 = sig_prev, sig
#     V1, V2 = vec_prev, vec
#     dist = (1 - np.abs(V1.T @ V2)) * distancematrix(D1, D2)

#     # Is there a best permutation? use munkres.
#     reorder = m.compute(np.transpose(dist))
#     reorder = [coord[1] for coord in reorder]

#     V2 = V2[:, reorder]
#     D2 = D2[reorder]

#     # also ensure the signs of each eigenvector pair were consistent if possible
#     S = np.squeeze( np.sum( V1 * V2, axis=0 ).real ) 
#     V2[:, S<0] *= -1
#     # V2 = V2 * (~S * 2 - 1)
    
#     return D2, V2


# def distancematrix(vec1, vec2):
#     """simple interpoint distance matrix"""
#     v1, v2 = np.meshgrid(vec1, vec2)
#     return np.abs(v1 - v2)
