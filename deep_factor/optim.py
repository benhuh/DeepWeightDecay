import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer


def get_optim(model, config):
    optim_ = config.optimizer
    lr  = config.lr / config.depth
    momentum = config.momentum
    weight_decay = 0 #config.weight_decay # * config.lr 
    eps = config.eps
    
    params = model.params(task_lr = config.task_lr)
    
    if optim_ == 'SGD':
        optimizer = optim.SGD(params, lr, weight_decay=weight_decay, momentum=momentum)
    elif optim_ == 'GroupRMSprop':
        optimizer = GroupRMSprop(params, lr, eps=eps, weight_decay=weight_decay, momentum=momentum)
    elif optim_ == 'RMSprop':
        optimizer = optim.RMSprop(params, lr, eps=eps, weight_decay=weight_decay, momentum=momentum)
    elif optim_ == 'Adam':
        optimizer = optim.Adam(params, lr, weight_decay=weight_decay, betas=(momentum,0.999))
    elif optim_ == 'AdamW':
        optimizer = optim.AdamW(params, lr, weight_decay=weight_decay, betas=(momentum,0.999))
    # elif optim_ == 'cvxpy':
    #     cvx_opt(prob)
    #     return
    else:
        raise ValueError(f'Invalid optimizer: {optim_}')
    
    def adjust_lr_wd(self, lr_factor, wd_factor):
        for group in self.param_groups:
            group['weight_decay'] = self.wd0*wd_factor
            group['lr'] = self.lr0*lr_factor
    
    optimizer.adjust_lr_wd = adjust_lr_wd
    optimizer.lr0 = lr
    optimizer.wd0 = weight_decay
    return optimizer


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