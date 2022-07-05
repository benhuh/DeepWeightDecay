# Adapted from https://github.com/roosephu/deep_matrix_factorization/
# import lunzi as lz
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from model import Deep_Linear #, init_model 
from optim import get_optim, GroupRMSprop
from save_load import set_logging, get_db, get_base_path, make_0_plot, make_1_plot, make_2_plot, save_config, get_run_path
from config import CONFIG, PROBLEM_CONFIG #, Multitask_CONFIG # default_problem  # OPTIM_CONFIG, default_optim , default_n_dict
############################


################################

def main(config=None, **kwargs): 
    if config is None:
        config = CONFIG(**kwargs)
    else:
        config.set_params(**kwargs)
    
    _writer, _log, tensorboard_path = set_logging(config, kwargs)
    save_config(config, tensorboard_path, filename = 'config.json', pop_list=['loop_params'])
    model = get_model(config)
    _log.info(model)
    
    # before = model.task_layer.weight.clone()
    train_loss, test_loss = train(config, model, _writer, _log)

    _log.info(f"train loss = {train_loss}. test loss = {test_loss}")
    _writer.close()
        
    # save_checkpoint(f'model{algorithm.config_num}.pkl')
    
    # after = model.task_layer.weight.clone()
    # print('before-after:', (before-after).norm().item())
        
def train(config, model, _writer, _log):
    
    optimizer = get_optim(model, config)
    test_loss=None
    model.step = getattr(model,'step',0)
    
    for step in range(model.step, model.step+config.n_iters):
        
        train_loss = model.train_loss()
        optimizer.zero_grad()
        train_loss.backward()
        
        break_flag = train_loss.item() <= config.train_thres
        
        with torch.no_grad():
            if step % config.n_dev_iters == 0 or break_flag: 
                test_loss = evaluate(config, model, optimizer, _writer, _log, step, break_flag, train_loss.item(), test_loss)
        if break_flag:
            break
            
        if config.clip_val_>0:
            clip_grad_norm_(model.params(), config.clip_val_)
            # clip_grad_norm_(model.params(), config.clip_val_/model.lr_factor*10000)


        optimizer.step()
#         optimizer.adjust_lr_wd(optimizer, lr_factor = model.lr_factor, wd_factor = model.wd_factor)

    model.step=step+1
    return train_loss, test_loss

    
def get_model(config):
    model = Deep_Linear(config, **vars(config.problem))
    model.to(config.device)
    return model


################################

def evaluate(config, model, optimizer, _writer, _log, step, break_flag, train_loss, test_loss_prev):
    test_loss = model.test_loss().item()
    if config.problem.name=='multi-task': # and config.optimize_task:
        weight = model.shared_weight
    else:
        weight = model.e2e
        
    # try:
    if len(weight.shape)==1:
        sig = weight
    else:
        U, sig, V = weight.svd() 
    
    # except:
    
    schatten_norm = sig.pow(2. / config.depth).sum()

    # d_e2e = model.d_e2e()
    # full = U.t().mm(d_e2e).mm(V).abs()  # we only need the magnitude.
    # n, m = full.shape
    # diag = full.diag()
    # mask = torch.ones_like(full, dtype=torch.int)
    # mask[np.arange(min(n, m)), np.arange(min(n, m))] = 0
    # off_diag = full.masked_select(mask > 0)
    # _writer.add_scalar('diag/mean', diag.mean().item(), global_step=step)
    # _writer.add_scalar('diag/std', diag.std().item(), global_step=step)
    # _writer.add_scalar('off_diag/mean', off_diag.mean().item(), global_step=step)
    # _writer.add_scalar('off_diag/std', off_diag.std().item(), global_step=step)
    

    param_norm, grads_norm = get_param_grad_norm(model)
    adjusted_lr = optimizer.param_groups[0]['adjusted_lr']  if isinstance(optimizer, GroupRMSprop) else optimizer.param_groups[0]['lr']

    _log.info(f"Iter #{step}: train = {train_loss:.3e}, test = {test_loss:.3e}, "
              f"Schatten norm = {schatten_norm:.3e}, "
              f"grad: {grads_norm:.3e}, "
              f"lr = {adjusted_lr:.3f}")

    _writer.add_scalar('loss/train', train_loss, global_step=step)
    _writer.add_scalar('loss/test', test_loss, global_step=step)
    _writer.add_scalar('norm/Schatten', schatten_norm, global_step=step)
    _writer.add_scalar('norm/grads', grads_norm, global_step=step)
    _writer.add_scalar('norm/params', param_norm, global_step=step)

    for i in range(min(config.n_singulars_save, len(sig))):
        _writer.add_scalar(f'singular_values/{i}', sig[i], global_step=step)
        
    # if hasattr(model.problem,'distance_U'):
    #     _writer.add_scalar(f'distance_U', model.problem.distance_U(model.shared_weight), global_step=step)
    if hasattr(model.problem,'get_surrogate_loss2'):
        _writer.add_scalar(f'loss/surrogate', model.problem.get_surrogate_loss2(model.shared_weight, model.wd_task), global_step=step)
        
    # torch.save(e2e, _fs.resolve("$LOGDIR/final.npy"))
            
    return test_loss
             
def get_param_grad_norm(model):
    grads_all = torch.cat([param.grad.data.reshape(-1) for param in model.params()])
    param_all = torch.cat([param.data.reshape(-1) for param in model.params()])
    grads_norm = grads_all.pow(2).sum().sqrt().item()
    param_norm = param_all.pow(2).sum().sqrt().item()
    return param_norm, grads_norm

####################

from collections import OrderedDict

def run_loops_plot(db, loop_params_in :OrderedDict, loss_or_svd, params_in = None):
    
    if params_in is None:
        loop_params, params = remove_len1(loop_params_in)
    else:
        loop_params = loop_params_in.copy()
        params = params_in.copy()
        
    if len(loop_params)>2:
        key, vals = loop_params.popitem()
        for val in vals:
            params[key] = val
            run_loops_plot(db, loop_params, loss_or_svd, params)
    else: # make a new figure
        if len(loop_params)==0:
            make_0_plot(db, loss_or_svd, params)
        elif len(loop_params)==2:
            make_2_plot(db, loop_params, params, loss_or_svd)
        elif len(loop_params)==1:
            make_1_plot(db, loop_params, params, loss_or_svd)

def remove_len1(loop_params):
    params = OrderedDict()
    loop_params_ = loop_params.copy()
    for key, vals in loop_params.items():
        if len(vals)==1:
            params[key]=vals[0]
            loop_params_.pop(key)
    return loop_params_, params
    
###################


def run_loops_main(config=None, **kwargs): 
    if config is None:
        config = CONFIG(**kwargs)
    else:
        config.set_params(**kwargs)
    run_loops_recur(config)

def run_loops_recur(config, loop_params_in:OrderedDict = None, params_in = None):
    
    if loop_params_in is None:
        assert params_in is None
        base_path = get_base_path(config)
        _, file_name = get_run_path(base_path, run_str = 'config_all', suffix='.json')
        save_config(config, base_path, filename = file_name) 
        
        loop_params, params = remove_len1(config.loop_params)
        
    else:
        loop_params = loop_params_in.copy()  
        params = params_in.copy()
    
    if len(loop_params)>0:
        key, vals = loop_params.popitem(last=False) #FIFO
        for val in vals:
            params[key] = val
            run_loops_recur(config, loop_params, params)
    else:
        model = main(config, **params)
        # return model
