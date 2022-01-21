import numpy as np
import torch as tr
from utils import get_theta, pi_dist
from training_nn import train_ac_distill
from utils import node2obs, softmax
import torch.nn.functional as F
import copy
import pdb

device = tr.device("cuda:0" if tr.cuda.is_available() else "cpu")


class Runner(object):
    
    def __init__(self, T_dist, A):
        
        self.T_dist = T_dist
        self.A = A
        self.nS = 24
        self.pi0 = 0.5 * tr.ones([self.nS, 2], requires_grad=False)
        
    def beta(self, k):
        return np.exp(-k/10.)
        
    def run(self, M_tasks, args, task_list=None, init_params=None):
        return_hists, return_hists_d, pi0_hist = [], [], []
        T_dist = self.T_dist
        pi0 = self.pi0
        if args['reg'] not in ['TV', 'log-barrier', 'maxent']:
            pi0 = get_theta([15, 1])
        
        for m in range(1, M_tasks + 1):
            # draw a task 
            T_m = T_dist.sample() if task_list is None else task_list[m-1]
            args['decisions'] = T_m.decisions
            # randomly init. parameters
            n_s = len(args['decisions'])
            args['pi0'] = pi0
            policy, value_fn = get_theta([n_s, 1]), get_theta([n_s, 1])
            
            # train policy 
            policy, pi0, returns = self.A(policy, value_fn, T_m, args, task_id=m, display_iter=0)
                    
            # update pi0
            if args['reg'] == 'TV':
                # update pi0
                for s in T_m.decisions:
                    state = node2obs(T_m, s) # make the states into a single tensor 
                    pi = tr.dot(softmax(policy).flatten(), state) 
                    pi_star = F.one_hot(tr.argmax(tr.tensor([1 - pi, pi])), num_classes=2) 
                    pi0[s] += (pi_star - pi0[s]) / m 
                    pi0[s] = softmax(pi0[s] / self.beta(m)) 
            
            # save returns
            return_hists.append(returns)
            pi0_hist.append(copy.copy(pi0))
            self.pi = pi0

        return return_hists, pi0_hist

