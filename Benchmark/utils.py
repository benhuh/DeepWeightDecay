import numpy as np
from scipy import linalg
import torch

def operator_update_U(S, V, d, r):
  operator = np.einsum('ijk,ir,il->jrkl', S, V, V).reshape(d*r,d*r)
  return operator

def update_U(V, U_opt, V_opt, S, Z=None):
    d, r = S.shape[1], V_opt.shape[1]
    U_hat = np.einsum('ijk,kr,ir,il->jl', S, U_opt, V_opt, V)
    if Z is not None:
        U_hat += np.einsum('ij,ik->jk', Z, V)

    # operator = np.einsum('ijk,ir,il->jrkl', S, V, V).reshape(d*r,d*r)
    operator = operator_update_U(S, V, d, r)
    operator_inv = np.linalg.inv(operator)
    U_hat = operator_inv.dot(U_hat.reshape(d*r)).reshape(d, r)
    U, R = np.linalg.qr(U_hat, mode='reduced')

    return U

def update_V(U, U_opt, V_opt, S, Z=None):
    V = np.einsum('jk,ijl,lr,ir->ik', U, S, U_opt, V_opt)
    if Z is not None:
        V += np.einsum('jk,ij->ik', U, Z)

    operators = np.einsum('jk,ijl,lr->ikr', U, S, U)
    operator_invs = np.zeros_like(operators)
    for i in range(operators.shape[0]):
        operator_invs[i] = np.linalg.inv(operators[i])

    V = np.einsum('ikr,ir->ik', operator_invs, V)

    return V

def gradients_UV(U, V, U_opt, V_opt, S, Z=None):
    d, r, t = S.shape[1], V_opt.shape[1], V_opt.shape[0]
    regs_diff = U.dot(V.T) - U_opt.dot(V_opt.T)
    grad_U = np.einsum('idk,ki,ir->dr', S, regs_diff, V)/t
    grad_V = np.einsum('idk,ki,dr->ir', S, regs_diff, U)
    if Z is not None:
        grad_U = grad_U + np.einsum('id,ir->dr', Z, V)/t
        grad_V = grad_V + np.einsum('id,dr->ir', Z, U)

    return grad_U, grad_V



def distance_U(U, U_ref):
    r = U.shape[1]
    if not U.shape[1] == U_ref.shape[1]:
      return 0
    else:
      return np.linalg.norm(
        U - U_ref.dot(U_ref.T.dot(U))
        )/(r**0.5)

def get_surrogate_loss2(W, W0, eps = 0.0):
    if not type(W) == torch.Tensor:
        W = torch.from_numpy(W).float()
        W0 = torch.from_numpy(W0).float()
        W = W.t() # match the dimensions
        W0 = W0.t()
    _, sig, v = W.svd()
    sig_PW = sig**2 / (sig**2 + eps)  
    WW_normalized = v * sig_PW @ v.T
    err = WW_normalized - W0.T@W0
    surrogate_loss = (err.norm()**2/  err.numel()).sqrt()
    return surrogate_loss

def get_true_loss2(W, W0, eps = 0.0001):  # agress with get_true_loss0 but noisier
    if not type(W) == torch.Tensor:
        W = torch.from_numpy(W).float()
        W0 = torch.from_numpy(W0).float()
    W = W.t() # match the dimensions
    W0 = W0.t()
    Q = W@W0.t()
    if Q.norm()**2>eps/100:
        u, s, v = Q.svd()  #     Q = u * s @ v.t()
        s_pinv = s/(s**2 + eps)
        Q_pinv = v * s_pinv @ u.t()
        return  (W0 - Q_pinv @ W).norm()**2 + eps* (s_pinv**2).sum()  
    else:
        Q_pinv = Q.T@(Q@Q.T + eps*torch.eye(W.shape[0])).inverse()
        return  (W0 - Q_pinv @ W).norm()**2 + eps* Q_pinv.norm()**2  # W0.shape[0] - 2*(Q_pinv@Q).trace() + (Q_pinv@W).norm()**2 + eps* 
    
def distance_U_spectral(U, U_ref):
    if not U.shape[1] == U_ref.shape[1]:
      return 0
    try: ret = linalg.svdvals(
        U - U_ref.dot(U_ref.T.dot(U)))[0]
    except ValueError:
        ret = np.nan
    return ret
