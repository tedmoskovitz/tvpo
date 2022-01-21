import numpy as np
from typing import Callable
import torch as tr
from torch.nn import init
import pandas as pd
import matplotlib.pyplot as plt
import random

def KL(p: tr.Tensor, q: tr.Tensor) -> tr.Tensor:
  """compute KL(p||q)"""
  return tr.sum(p * tr.log(p + 1e-5) - p * tr.log(q + 1e-5))

def TV(p: tr.Tensor, q: tr.Tensor) -> tr.Tensor: 
  """compute TV(p, q)"""
  return tr.max(tr.abs(p - q))

def entropy(p: tr.Tensor, axis: int = None) -> tr.Tensor:
  """compute H[p]"""
  assert tr.all((tr.min(p) >= 0 and tr.sum(p, axis=axis)).bool()), "bad distribution"
  return -tr.sum(p * tr.log(p + 1e-5), axis=axis)


def policy_similarity(state: tr.Tensor, Pi: list, d: Callable, omega: float = 1.0) -> float:
  """compute the policy agreement function"""
  n = len(Pi)
  div = 0
  for i in range(n):
    for j in range(i + 1, n):
      div += d(Pi[i](state), Pi[j](state))
  return tr.exp(-omega * div)


def softmax(policy):
  return(tr.exp(policy) / (1 + tr.exp(policy)))


def set_seed_everywhere(seed):
    tr.manual_seed(seed)
    if tr.cuda.is_available():
        tr.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Get policy: n x 2 matrix, where n is the number of nodes
def get_theta(dims, std_val=True):
  n_states, n_actions = dims
  if std_val >0:
    return init.normal_(tr.zeros(n_states, n_actions, requires_grad=True), std=std_val)
  return tr.zeros(n_states, n_actions, requires_grad=True)


# Get the current path taken by the policy (argmax)
def get_path(env, policy, decisions, print_path=True):
  node = env.root
  path = [node.value]
  with tr.no_grad():
    while (node.left is not None) or (node.right is not None):
      if (node.left is not None) and (node.right is None):
        node = node.left
      elif (node.left is None) and (node.right is not None):    
        node = node.right
      else:
        pred = softmax(policy)[decisions.index(node.value)].detach().numpy().astype(np.float64)
        print (node.value, node.reward, pred)
        if pred >= 0.5:
          node = node.right
        else:
          node = node.left
      path.append(node.value)
  if print_path:
    print("Reward: {} \nPath: {}".format(str(node.reward), path))
  return(node.value, node.reward)

def get_df(results):
  rtns = [r[1] for r in results]
  seeds = [[idx]*len(rtns[0]) for idx, x in enumerate(rtns)]
  timesteps = [x+1 for x in range(len(rtns[0]))] * len(rtns)
  df = pd.DataFrame({'Timesteps': timesteps,
                          'Return': [x for rtn in rtns for x in rtn],
                          'Seed': [x for seed in seeds for x in seed]})
  return df


def pi_dist(policy, state):
  """get the action distribution for the given policy and state"""
  p_right = tr.dot(softmax(policy).flatten(), state)# for s in state]
  return tr.tensor([1 - p_right, p_right])

def node2obs(env, node_num):
  # each observation is structured as a one-hot vector of length |S'|, where S' is the set of non-leaf nodes
  # the obs has a 1 in the location corresponding to the current state; the observation is all zeros at termination
  n = len(env.decisions)
  return tr.eye(n)[env.decisions.index(node_num)]

def obs2node(env, obs):
  """convert a one-hot state representation to the corresponding node"""
  return env.decisions[tr.argmax(obs)]


class dumb_policy(object):

  def __init__(self, diff_action: int, branch_depth: int = 2) -> None:
    self.diff_action = diff_action # the action that's taken after the "shared" part of the state space
    self.branch_depth = branch_depth # depth delimiting the "shared" part
    self.action_probs = {0: tr.Tensor([0.99, 0.01]), 1: tr.Tensor([0.01, 0.99])}

  def __call__(self, state: int) -> np.ndarray:
    # returns distribution over actions
    depth = state // 2 + 1
    if depth <= self.branch_depth:
      action_dist = tr.Tensor([0.01, 0.99]) # go right
    else:
      action_dist = self.action_probs[self.diff_action]

    return action_dist

  def get_action(self, state: int) -> int:
    depth = state // 2 + 1
    if depth <= self.branch_depth:
      action_dist = tr.Tensor([0.01, 0.99]) # go right
    else:
      action_dist = self.action_probs[self.diff_action]

    # sample action
    action = np.random.choice(2, p=action_dist.detach().numpy()) 
    
    return action 

def plot_runs(ax, n_parallel, returns, label):
  mean_rtns = np.mean(returns, axis=0)
  std_rtns = np.std(returns, axis=0) / 2
  timesteps = n_parallel * np.arange(len(mean_rtns))  
  h, = ax.plot(timesteps, mean_rtns, label=label)
  ax.fill_between(timesteps, mean_rtns - std_rtns, mean_rtns + std_rtns, alpha=0.2)
  return h 

def errorfill(x, y, yerr, color='C0', alpha_fill=0.3, ax=None, label=None, lw=1, ls='-'):
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    l, = ax.plot(x, y, color=color, label=label, lw=lw, ls=ls)

    ax.tick_params(axis='both', labelsize=20)
    ax.grid(alpha=0.7)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
    return l