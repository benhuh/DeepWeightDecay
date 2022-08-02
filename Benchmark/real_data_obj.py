from dataclasses import replace
import numpy as np
import sys
import os
from scipy import stats
# sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'MTLR'))

from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
sys.path.append(os.path.join(str(root), 'MTLR'))

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from MTLR.MTLR_init import get_W0, normalize
from meta_learn import MetaLearnProb


data_config = {}
data_config['school'] = {
    'data_dir': "../Data/school/data_dummpy.csv",
    'task_list': list(range(47)),
    'num_task': 47,
    'max_per_task_train': 100,
    'max_per_task_val': 45,
}

class RealDataProb(MetaLearnProb):
    def __init__(self, name = "school", sigma=None, homogeneity=None):
        self.config = data_config[name]
        self.dat = np.loadtxt(self.config['data_dir'], skiprows = 1, delimiter = ',')
        t = self.config['num_task']
        d = 27
        r = 2
        super().__init__(d, r, t)
        self.m = self.config['max_per_task_train']
        self.m_val = self.config['max_per_task_val']
        self.U = None 
        
    def generate_data(self, m_train = None, t_train = None, m_val = None, t_val = None, seed = 3):
        # self.X = np.zeros((self.t, self.m, self.d))
        # self.X_val = np.zeros((self.t, self.m_val, self.d))
        # self.Y = np.zeros((self.t, self.m))
        # self.Y_val = np.zeros((self.t, self.m_val))
        
        self.X = []
        self.X_val = []
        self.Y = []
        self.Y_val = []
        
        if m_train is not None:
            self.m = m_train
        if t_train is not None:
            self.t = t_train
        if m_val is not None:
            self.m_val = m_val
        if t_val is not None:
            self.t_val = t_val
        
        self.dat[:, 2:] = stats.zscore(self.dat[:, 2:], axis = 0)
        
        # counting from begining
        np.random.seed(seed)
        task_list = list(np.random.choice(range(self.config['num_task']), self.config['num_task'], replace = False))
        for t in task_list:
            sub_dat = self.dat[self.dat[:, 0] == (t+1), 2:]
            if sub_dat.shape[0] > self.m and len(self.X) < self.t:
                self.X.append(sub_dat[:self.m, :])
                self.Y.append(sub_dat[:self.m, 1])
        # counting from end
        task_list.reverse()
        for t in task_list:
            # t = self.config['num_task'] - t1 - 1
            sub_dat = self.dat[self.dat[:, 0] == (t+1), 2:]
            if sub_dat.shape[0] > self.m_val and len(self.X_val) < self.t_val:
                self.X_val.append(sub_dat[:self.m_val, :])
                self.Y_val.append(sub_dat[:self.m_val, 1])
        self.t = len(self.X)
        self.t_val = len(self.X_val)
        
        self.X = np.concatenate(np.expand_dims(self.X, axis=0))
        self.X_val = np.concatenate(np.expand_dims(self.X_val, axis = 0))
        self.Y = np.concatenate(np.expand_dims(self.Y, axis = 0))
        self.Y_val = np.concatenate(np.expand_dims(self.Y_val, axis = 0))
        
        
        
        
        return self.X, None
        
                


