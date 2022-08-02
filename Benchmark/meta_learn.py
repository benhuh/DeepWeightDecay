import numpy as np
from scipy import linalg

import torch
from torch.utils.tensorboard import SummaryWriter

from alt_min_gd_wd import apply_alt_min_gd_wd


from utils import update_U, update_V, gradients_UV, distance_U, distance_U_spectral
from MTLR.measurements2 import get_true_loss2
from MTLR.MTLR_init import get_W0, normalize



class MetaLearnProb(object):
    def __init__(self, d, r, t, sigma=None, homogeneity=None):
        self.d = d
        self.r = r
        self.t = t

        assert d >= r
        self.U = np.eye(d)[:, :r]
        self.U = get_W0(self.r, self.d)[0].detach().numpy().transpose()
        self.V = np.random.normal(size=[t, r])
        self.V = self.init_V()

        self.homogeneity = homogeneity
        if self.homogeneity is not None:
            # import ipdb; ipdb.set_trace()

            assert 0.0 <= self.homogeneity and self.homogeneity <= 1.0

            nof_duplicates = int(self.homogeneity*self.t)
            duplicate_norm = r**0.5
            duplicate_idx = np.random.choice(self.t, size=nof_duplicates)
            duplicate_vector = np.zeros([r])
            duplicate_vector[0] = duplicate_norm
            self.V[duplicate_idx] = duplicate_vector

        self.sigma = 0 if sigma is None else sigma

        self.m = None
        self.X = None
        self.noise = None
        self.S = None
        self.Z = None
        self.moment = None

        self.m_val = None
        self.X_val = None
        self.noise_val = None
        self.S_val = None
        self.Z_val = None
    def init_V(self, t = None):
        if t == None:
            t = self.t
        V = torch.randn(t, 1, self.r)
        V = normalize(V, dim=-1).detach().numpy()
        V = np.mean(V, 1)
        return V
    def generate_data(self, m, noise=True, dist_x=None, dist_y=None):
        dist_x = dist_x or 'normal'
        dist_y = dist_y or 'normal' 
        self.m = m

        if dist_x == 'normal':
            X = np.random.normal(
                size=[self.t, self.m, self.d])
        elif dist_x == 'uniform':
            X = torch.randn(self.t, self.m, self.d)
            X = normalize(self.X, dim=2)
            X = self.X.detach().numpy()
        elif dist_x == 'exp':
            X = np.random.standard_exponential(
                size=[self.t, self.m, self.d])
        elif dist_x == 'laplace':
            X = np.random.laplace(
                scale=(2.0)**-0.5, size=[self.t, self.m, self.d])
        else:
            raise ValueError('Unknown distribution={}!'.format(dist_x))

        if noise or self.sigma == 0:
            if dist_y == 'normal':
                noise = self.sigma*np.random.normal(
                    size=[self.t, self.m])
            elif dist_y == 'exp':
                noise = self.sigma*np.random.standard_exponential(
                    size=[self.t, self.m])
            elif dist_y == 'laplace':
                noise = np.random.laplace(
                    scale=(2.0)**-0.5, size=[self.t, self.m])
            else:
                raise ValueError('Unknown distribution={}!'.format(dist_y))
        else:
            noise = np.zeros(shape=[1, 1])

        self.X = X
        self.noise = noise
        return X, noise

    def generate_val_data(self, m_val, t = None, noise=True):
        self.m_val = m_val
        if t == None:
            t = self.t
        self.X_val = np.random.normal(
            size=[t, self.m_val, self.d])

        if noise or self.sigma == 0:
            self.noise_val = self.sigma*np.random.normal(
                size=[t, self.m_val])
        else:
            self.noise_val = np.zeros(shape=[1, 1])

        return self.X_val, self.noise_val

    def get_altmin_data(self):
        S = np.einsum('ijk,ijl->ikl', self.X, self.X)/self.m #covariance matrix?
        Z = np.einsum('ij,ijk->ik', self.noise, self.X)/self.m # not sure
        return S, Z

    def get_method_of_moments_data(self):
        y = np.einsum(
            'ir,dr,ijd->ij', self.V, self.U, self.X)
        y = y + self.noise
        self.moment = np.einsum(
            'ij,ij,ijk,ijl->kl', y, y, self.X, self.X)/self.m/self.t

        return self.moment

    def get_2nd_ord_method_of_moments_data(self):
        y = np.einsum(
            'ir,dr,ijd->ij', self.V, self.U, self.X)
        y = y + self.noise

        second_moment = np.einsum(
            'ij,ijd->id', y, self.X)/self.m

        second_ord_moment = np.einsum(
            'ik,il->ikl', second_moment, second_moment)
        second_ord_self_moment = np.einsum(
            'ij,ij,ijk,ijl->ikl', y, y, self.X, self.X)/self.m

        self.second_ord_moment = (
            (self.m/(self.m-1))*second_ord_moment - second_ord_self_moment/(self.m-1)).mean(axis=0)
        return self.second_ord_moment

    def mse_loss(self, U, V, average=None):
        y = np.einsum(
            'ir,dr,ijd->ij', self.V, self.U, self.X)
        y = y + self.noise
        y_est = np.einsum(
            'ir,dr,ijd->ij', V, U, self.X)
        if average is None or average:
            return ((y - y_est)**2).mean()
        else:
            return ((y - y_est)**2).mean(axis=1)
    def avg_mse_loss_val(self, U, V, average=None):
        regressors = np.einsum(
            'ir,dr->id', self.V, self.U)
        regressors_est = np.einsum(
            'ir,dr->id', V, U)
        return ((regressors - regressors_est)**2).mean()
    def test_mse_loss(self, U, m = None, get_opt_V = None, t = None):
        return 0 # not used anymore
        
    def get_torch_data(self, Xs, Ys):
        Xs = Xs.swapaxes(1, 2)
        Ys = np.expand_dims(Ys, 1)
        Ys = torch.from_numpy(Ys)
        Xs = torch.from_numpy(Xs)
        Xs,Ys=Xs.float(),Ys.float()
        return Xs, Ys

def apply_alt_min(
    prob, N_step, U_init=None,
    partition=False,
    init_mom=False,
    writer = None
    ):
    if U_init is None:
        if init_mom:
            U_init = apply_method_of_moments(prob)['U']
        else:
            U_init = np.linalg.qr(np.random.normal(
                size=[prob.d, prob.d]))[0][:, :prob.r]
    S, Z = prob.get_altmin_data()

    t = S.shape[0]
    if N_step <= t and partition:
        partitions = np.array_split(np.random.permutation(t), N_step)
    else:
        partitions = [slice(None)]*N_step
    U = U_init
    U_list = [U]
    dist_U_list = [distance_U(U, prob.U)]
    true_loss_list = [get_true_loss2(U, prob.U)]
    dist_U_spectral_list = [distance_U_spectral(U, prob.U)]
    avg_mse_loss_list = []
    for step in range(N_step):
        V = update_V(U, prob.U, prob.V, S=S, Z=Z)
        avg_mse_loss_list.append(prob.avg_mse_loss_val(U, V))
        U = update_U(V[partitions[step]], prob.U, prob.V[partitions[step]],
                S=S[partitions[step]], Z=Z[partitions[step]])
        U_list.append(U)
        dist_U_list.append(distance_U(U, prob.U))
        true_loss_list.append(get_true_loss2(U, prob.U))
        dist_U_spectral_list.append(distance_U_spectral(U, prob.U))
        if writer:
            writer.add_scalar('U_dist', dist_U_list[-1], step)
            writer.add_scalar('true_loss', true_loss_list[-1], step)
            writer.add_scalar('MSE', avg_mse_loss_list[-1], step)
    if writer:
        writer.close()
    V = update_V(U, prob.U, prob.V, S=S, Z=Z)
    avg_mse_loss_list.append(prob.avg_mse_loss_val(U, V))
    output = {
        'true_loss_list': true_loss_list,
        'U_init': U_init,
        'U_list': U_list,
        'surrogate_loss_list': dist_U_list,
        'dist_U_spectral_list': dist_U_spectral_list,
        'avg_mse_loss_list': avg_mse_loss_list,
        'U': U,
        }
    return output

####
def apply_method_of_moments(prob, writer = None):
    moment = prob.get_method_of_moments_data()

    U_list = []
    dist_U_list = []
    dist_U_spectral_list = []
    avg_mse_loss_list = []
    test_mse_loss_list = []

    _, U = linalg.eigh(moment, eigvals=(prob.d - prob.r, prob.d-1))

    U_list.append(U)
    dist_U_list.append(distance_U(U, prob.U))
    dist_U_spectral_list.append(distance_U_spectral(U, prob.U))
    S, Z = prob.get_altmin_data()
    V = update_V(U, prob.U, prob.V, S=S, Z=Z)
    avg_mse_loss_list.append(prob.avg_mse_loss_val(U, V))
    test_mse_loss_list.append(prob.test_mse_loss(U, prob.m, t = 30))

    output = {
        'surrogate_loss_list': dist_U_list,
        'dist_U_spectral_list': dist_U_spectral_list,
        'avg_mse_loss_list': avg_mse_loss_list,
        'test_mse_loss_list': test_mse_loss_list,
        'U': U,
        }
    return output

####
def apply_2nd_ord_method_of_moments(prob, writer = None):
    second_ord_moment = prob.get_2nd_ord_method_of_moments_data()

    U_list = []
    dist_U_list = []
    dist_U_spectral_list = []
    avg_mse_loss_list = []
    test_mse_loss_list = []

    _, U = linalg.eigh(second_ord_moment, eigvals=(prob.d - prob.r, prob.d-1))

    U_list.append(U)
    dist_U_list.append(distance_U(U, prob.U))
    dist_U_spectral_list.append(distance_U_spectral(U, prob.U))
    S, Z = prob.get_altmin_data()
    V = update_V(U, prob.U, prob.V, S=S, Z=Z)
    avg_mse_loss_list.append(prob.avg_mse_loss_val(U, V))
    test_mse_loss_list.append(prob.test_mse_loss(U, prob.m, t = 30))

    output = {
        'dist_U_list': dist_U_list,
        'dist_U_spectral_list': dist_U_spectral_list,
        'avg_mse_loss_list': avg_mse_loss_list,
        'test_mse_loss_list': test_mse_loss_list,
        'U': U,
        }
    return output

####
def apply_grad_descent(
    prob, N_step,
    stepsize,
    regularizer=False,
    U_init=None,
    init_mom=False,
    qr_decomp=True,
    writer = None
    ):
    if U_init is None:
        if init_mom:
            U_init = apply_method_of_moments(prob)['U']
        else:
            U_init = np.linalg.qr(np.random.normal(
                size=[prob.d, prob.d]))[0][:, :prob.r]
    S, Z = prob.get_altmin_data()

    t = S.shape[0]
    U = U_init
    U_list = [U]
    dist_U_list = [distance_U(U, prob.U)]
    grad_U_norm_list = []
    grad_V_norm_list = []
    dist_U_spectral_list = [distance_U_spectral(U, prob.U)]
    avg_mse_loss_list = []
    V = update_V(
        U, prob.U, prob.V,
        S=S, Z=Z)
    for step in range(N_step):
        avg_mse_loss_list.append(prob.avg_mse_loss_val(U, V))
        grad_U, grad_V = gradients_UV(
            U, V, prob.U, prob.V,
            S=S, Z=Z)
        grad_U_norm_list.append((np.sum(grad_U**2))**0.5)
        grad_V_norm_list.append((np.sum(grad_V**2))**0.5)
        if regularizer:
            raise NotImplementedError('regularizer not implemented')
        U = U - stepsize*grad_U
        V = V - stepsize*grad_V
        _U, R = np.linalg.qr(U, mode='reduced')
        if qr_decomp:
            U = _U
        U_list.append(_U)
        dist_U_list.append(distance_U(_U, prob.U))
        dist_U_spectral_list.append(distance_U_spectral(_U, prob.U))
        if writer:
            writer.add_scalar('U_dist', dist_U_list[-1], step)
            writer.add_scalar('MSE', avg_mse_loss_list[-1], step)
    
    avg_mse_loss_list.append(prob.avg_mse_loss_val(U, V))
    output = {
        'U_init': U_init,
        'U_list': U_list,
        'dist_U_list': dist_U_list,
        'dist_U_spectral_list': dist_U_spectral_list,
        'avg_mse_loss_list': avg_mse_loss_list,
        'U': U,
        'grad_U_norm_list': grad_U_norm_list,
        'grad_V_norm_list': grad_V_norm_list,
        }
    return output

####
def apply_alt_min_gd(
    prob, N_step, U_init=None,
    partition=False,
    init_mom=False,
    U_gd=True, stepsize=None, 
    writer = None
    ):
    if U_init is None:
        if init_mom:
            U_init = apply_method_of_moments(prob)['U']
        else:
            U_init = np.linalg.qr(np.random.normal(
                size=[prob.d, prob.d]))[0][:, :prob.r]
    S, Z = prob.get_altmin_data()

    t = S.shape[0]
    if N_step <= t and partition:
        partitions = np.array_split(np.random.permutation(t), N_step)
    else:
        partitions = [slice(None)]*N_step

    U = U_init
    U_list = [U]
    dist_U_list = [distance_U(U, prob.U)]
    # true_loss_list = [get_true_loss2(U, prob.U)]
    dist_U_spectral_list = [distance_U_spectral(U, prob.U)]
    avg_mse_loss_list = []
    grad_U_norm_list = []
    test_loss_list = []
    stepsize_list = []
    for step in range(N_step):
        ## All the V's are updated
        V = update_V(
            U, prob.U, prob.V,
            S=S, Z=Z)
        avg_mse_loss_list.append(prob.avg_mse_loss_val(U, V))
        if U_gd:
            if stepsize is None:
                W = V[partitions[step]].T.dot(V[partitions[step]])
                W_eig_max = linalg.eigh(W, eigvals_only=True, eigvals=(W.shape[0]-1, W.shape[0]-1))
                _stepsize = 1/W_eig_max
            else:
                _stepsize = stepsize
                stepsize_list.append(_stepsize)
                # possibly using the whole dataset.
                grad_U, _ = gradients_UV(U, V[partitions[step]], prob.U, 
                                         prob.V[partitions[step]], S=S[partitions[step]],
                                         Z=Z[partitions[step]])
                grad_U_norm_list.append((np.sum(grad_U**2))**0.5)
                U = U - _stepsize*grad_U
                U, R = np.linalg.qr(U, mode='reduced')
        else:
            U = update_U(
                    V, prob.U, prob.V[partitions[step]],
                    S=S[partitions[step]], Z=Z[partitions[step]])
        U_list.append(U)
        dist_U_list.append(distance_U(U, prob.U))
        #true_loss_list.append(get_true_loss2(U, prob.U))
        dist_U_spectral_list.append(distance_U_spectral(U, prob.U))
        test_loss_list.append(prob.test_mse_loss(U, prob.m, t = 30))
        if writer:
            # writer.add_scalar('U_dist', dist_U_list[-1], step)
            writer.add_scalar('Losses/MSE', avg_mse_loss_list[-1], step)
            writer.add_scalar('Losses/test_loss', test_loss_list[-1], step)
            # writer.add_scalar('true_loss', true_loss_list[-1], step)
    if writer:
        writer.close()
    V = update_V(U, prob.U, prob.V, S=S, Z=Z)
    avg_mse_loss_list.append(prob.avg_mse_loss_val(U, V))
    output = {
        # 'true_loss_list': true_loss_list,
        'U_init': U_init,
        'U_list': U_list,
        'surrogate_loss_list': dist_U_list,
        'dist_U_spectral_list': dist_U_spectral_list,
        'avg_mse_loss_list': avg_mse_loss_list,
        'test_mse_loss_list': test_loss_list,
        'U': U,
        'grad_U_norm_list': grad_U_norm_list,
        'stepsize_list': stepsize_list,
        }
    return output


