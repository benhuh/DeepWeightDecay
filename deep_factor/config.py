import types
import numpy as np

default_problems = {  'matrix-completion':  dict(name = 'matrix-completion', n_in = 100,   n_out = None,   n_feature = 5, n_data = 2000  ),
                      'matrix-sensing':     dict(name = 'matrix-sensing',    n_in = 100,   n_out = None,   n_feature = 5, n_data = 2000  ),
                      'multi-task':         dict(name = 'multi-task',       n_in = 36,    n_out = None,   n_feature = 4, n_data = 12,  rho=1.25 ),
                      'multi-task2':        dict(name = 'multi-task',       n_in = 68,    n_out = None,   n_feature = 4, n_data = 12,  rho=1.25 ),
                      'SVD':                dict(name = 'SVD',       n_in = 20,    n_out = None,   n_feature = None, ),
                 }


class BASE_CONFIG():
    
    def __init__(self, **kwargs):
        self.set_params(**kwargs)
        
    def set_params(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self.__class__, key):
                if key.startswith('_') or type(getattr(self.__class__, key)) in [types.MethodType, property]: 
                    continue
                else:
                    if key == 'problem':
                        if isinstance(val,str):
                            val = default_problems[val]
                        val = PROBLEM_CONFIG(**val)
                    setattr(self, key, val)   
            elif hasattr(self.problem.__class__, key):
                self.problem.set_params( **{str(key): val})   
            else:
                raise ValueError(f'Invalid CONFIG key: {key}')
                


class CONFIG(BASE_CONFIG):
    experiment = 'temp'
    problem = None 
    optim = None
    loop_params= None    
    
    optimize_task = False
    skip_task_layer = True
    normalize = False
    
    task_lr = None

    device = 'cuda'    
    depth  = None
    wide   = None
    init_scale = 1  
    initialization = 'gaussian'  # `orthogonal` or `identity` or `gaussian`
    
    n_iters = 5000 #20000 
    n_record = 200
    n_singulars_save = 10
    n_dev_iters = None

    optimizer = 'GroupRMSprop'
    lr, momentum = 0.1, 0
    clip_val_ = 100.0
    eps=1e-4
    weight_decay = 0    #  0.01
    # adaptive_decay=False
    train_thres = 0 #1e-6
    
    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        self.n_dev_iters = max(1, self.n_iters // self.n_record)
    


class PROBLEM_CONFIG(BASE_CONFIG):
    name = None
    n_in = None
    n_out = None
    n_feature = None
    n_data = None
    
    rho = None  # task_factor
    noise = 0  # 

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        if any(_ is None for _ in [self.rho, self.n_in, self.n_feature, self.n_data]):
            pass
        else:
            self.n_out=self.n_task
        
    @property
    def n_task(self):
        n_task_min = get_n_task_min(self.n_in, self.n_feature, self.n_data)
        return int(np.floor(self.rho*n_task_min))
        
#     @staticmethod
def get_n_task_min(n_in, n_feature, n_data, Ny=1):
    if isinstance(n_data,list):
        n_data=np.array(n_data)
    n_task_min = (n_feature/Ny) *(n_in-n_feature) / (n_data - n_feature)  
    return n_task_min
    
    
    

import json

def load_config(file_path):
    with open(file_path) as f:
        params = json.load(f)
    return CONFIG(**params)