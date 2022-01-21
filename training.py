import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import SGD
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm, trange
import copy 
import seaborn as sns
import pickle
from typing import Callable
from binarytree import tree, build, Node
import pdb

from envs import *
from utils import *

### training funcs for softmax tabular policy
  

def get_exp(policy, root, decisions):
  # compute expected return (ground truth value of policy)
  reward = tr.zeros(1)
  root.prob = 1
  for i in range(len(root.levels)):
    for node in root.levels[i]:
      if (node.left is None) and (node.right is None):
        reward += node.reward * node.prob
      elif (node.left is not None) and (node.right is None):
        node.left.prob = node.prob
      elif (node.left is None) and (node.right is not None):
        node.right.prob = node.prob
      else:
        pred = softmax(policy)[decisions.index(node.value)]
        node.left.prob = node.prob * (1-pred)
        node.right.prob = node.prob * pred
  return reward * -1



def get_exp_rl(policy, val, env, n_parallel, device):
    # batch data collection, ultimately we will have t rows, of n_parallel entries
    # returns:
    # > log_pi: the log prob of the action taken
    # > all_v: the value function for the s,a pairs
    # > all_r: the single step reward for the s,a pairs
    # Once episodes have terminated, we set the reward to zero, this is the done vector
    t=0
    vec_env = [copy.copy(env) for _ in range(n_parallel)] # n environments
    done = [False for _ in range(n_parallel)] # nx1 vec of done = False
    state = [env.reset() for env in vec_env] # n x state dim 
    zerostate = tr.zeros_like(state[0]) # placeholder state for when done
    placeholder = [zerostate, 0, True] # placeholder for when envs are done

    while not all(done):
      state = tr.stack((state)).to(device) # make the states into a single tensor 
      probs = tr.stack([tr.dot(softmax(policy).flatten(), s) for s in state])
      probs = tr.stack((1-probs,probs),dim=1) # 
      value = tr.stack([tr.dot(val.flatten(), s) for s in state]) # compute value
      m = Categorical(probs)
      action = m.sample() 
      action_vec = tr.stack([m.log_prob(action[i])[0].double() if not done[i] else tr.zeros_like(m.log_prob(action[i])[0].double()).to(device) for i in range(n_parallel)]) # 1 x n 
      value_vec = tr.stack([value[i] if not done[i] else tr.zeros_like(value[i]).to(device) for i in range(len(value))]) # 1 x n                                                                           
      step_results = [vec_env[i].step(action[i]) if not done[i] else placeholder for i in range(len(vec_env))]
      rewards = tr.stack([tr.tensor(x[1]).long() for x in step_results])
      new_state = [x[0] for x in step_results]
      done = [x[2] for x in step_results]
      state = [new_state[i] if not done[i] else zerostate for i in range(len(new_state))]
      vec_env = [vec_env[i] if not done[i] else zerostate for i in range(len(vec_env))]

      if t == 0:
        log_pi = action_vec.clone().reshape(1,-1)
        all_v = value_vec.clone().reshape(1,-1)
        all_r = rewards.clone().reshape(1,-1)
      else:
        log_pi = tr.cat((log_pi, action_vec.reshape(1,-1))) # each timestep is a new ROW of these
        all_v = tr.cat((all_v, value_vec.reshape(1,-1)))
        all_r = tr.cat((all_r, rewards.reshape(1,-1)))
      t += 1

    rewards = all_r.float() # single_step rewards, t x n
    reward_to_go = tr.zeros_like(rewards) # t x n
    reward_to_go[-1, :] = rewards[-1, :] # bottom row is final single step reward
    for i in range(2, reward_to_go.shape[0]+1): 
        reward_to_go[-i, :] = 0.99 * reward_to_go[-(i-1), :] + rewards[-i, :] # build up so each R_i = r_i + \sum_j=1^{i-1} r_j * \gamma^j

    returns = -reward_to_go.to(device)

    v_advantages = returns - all_v # t x n. Advantages with gradients through V
    value_loss = (v_advantages**2).mean(1).sum() # positive loss, want to minimize, must take negative

    advantages = returns - all_v.detach() # t x n. 
    pg_loss = (advantages * log_pi).sum()


    return pg_loss, value_loss, rewards.sum(0).mean()
 
def train_vanilla(policy, val, env, args, plot=False, show_path=False, display_iter=0):
  
  max_t = int(args['max_timesteps'] / args['n_parallel'])
  device = args['device']
  returns = []
  iterator = range(1, max_t+1)
  with trange(1, max_t+1) as iterator:
    for t in iterator:

      if args['exact']:
          loss = get_exp(policy, env.root, args['decisions'])
      else:
          loss, val_loss, reward = get_exp_rl(policy, val, env, args['n_parallel'], device)     

      gradient = tr.autograd.grad(loss, policy, create_graph=True)[0]
      gradient = gradient.detach()
      policy = policy - gradient * args['lr'] # regular gradient descent
      
      if not args['exact']:
          val_grad = tr.autograd.grad(val_loss, val, create_graph=True)[0]
          val_grad = val_grad.detach()
          val = val - val_grad * args['lr']

      ep_reward_mean = evaluate(policy, env, device)
      if display_iter == 0: iterator.set_description(f'r = {ep_reward_mean}');
      returns.append(ep_reward_mean)

      if display_iter > 0 and t % display_iter == 0:
        print (f"iter {t}: loss = {loss}, reward = {ep_reward_mean}, policy: {policy.detach().numpy().flatten()[:3]}, norm = {tr.norm(gradient)}")

        get_path(env, policy, args['decisions'], print_path=True)


  if plot:
    fig, ax = plt.subplots(figsize=(4,3))

    ax.plot(returns)
    ax.set_ylabel('Expected Reward')

  return policy, returns



def evaluate(policy, env, device):
    t=0
    n_parallel = 10
    vec_env = [copy.copy(env) for _ in range(n_parallel)] # n environments
    done = [False for _ in range(n_parallel)] # nx1 vec of done = False
    state = [env.reset() for env in vec_env] # n x state dim 
    zerostate = tr.zeros_like(state[0]) # placeholder state for when done
    placeholder = [zerostate, 0, True] # placeholder for when envs are done
    paths = []

    while not all(done):
      state = tr.stack((state)).to(device) # make the states into a single tensor
      paths.append(state)
      with tr.no_grad():
        probs = tr.stack([tr.dot(softmax(policy).flatten(), s) for s in state])
      probs = tr.stack((1-probs,probs),dim=1)
      m = Categorical(probs)
      action = m.sample()     
      action_vec = tr.stack([m.log_prob(action[i])[0].double() if not done[i] else tr.zeros_like(m.log_prob(action[i])[0].double()).to(device) for i in range(n_parallel)]) # 1 x n 
      step_results = [vec_env[i].step(action[i]) if not done[i] else placeholder for i in range(len(vec_env))]
      rewards = tr.stack([tr.tensor(x[1]).long() for x in step_results])
      new_state = [x[0] for x in step_results]
      done = [x[2] for x in step_results]
      state = [new_state[i] if not done[i] else zerostate for i in range(len(new_state))]
      vec_env = [vec_env[i] if not done[i] else zerostate for i in range(len(vec_env))]

      if t == 0:
        all_r = rewards.clone().reshape(1,-1)
      else:
        all_r = tr.cat((all_r, rewards.reshape(1,-1)))
      t += 1
    rewards = all_r.float() # single_step rewards, t x n
    return rewards.sum(0).mean().cpu().numpy().item()



def train_reg(policy, val, env, args, task_id=-1, plot=False, display_iter=0):

  max_t = int(args['max_timesteps'] / args['n_parallel'])
  return_hist = []
  pi0 = args['pi0']
  best_r = -np.inf
  best_pi = pi0

  iterator = range(1, max_t+1)
  iterator = iterator if display_iter > 0 else tqdm(iterator, position=0, leave=True)

  for t in iterator:


    loss, val_loss, reward = get_exp_rl_reg(
        policy, val, env, pi0, args
    )     

    gradient = tr.autograd.grad(loss, policy, create_graph=True)[0]
    gradient = gradient.detach()
    policy = policy - gradient * args['lr'] # regular gradient descent
    if args['reg'] not in ['TV', 'log-barrier', 'maxent']:
      pi0_gradient = tr.autograd.grad(loss, pi0, create_graph=True)[0]
      pi0_gradient = pi0_gradient.detach()
      pi0 = pi0 - pi0_gradient * 0.5 * args['lr']  # regular gradient descent
    
    val_grad = tr.autograd.grad(val_loss, val, create_graph=True)[0]
    val_grad = val_grad.detach()
    val = val - val_grad * args['lr']

    ep_reward_mean = evaluate(policy, env, args['device'])
    return_hist.append(ep_reward_mean)

    if display_iter == 0: iterator.set_description(f'task {task_id}, r = {ep_reward_mean}');

    if display_iter > 0 and t % display_iter == 0:
      print (f"iter {t}: loss = {loss}, reward = {ep_reward_mean}, policy: {policy.detach().numpy().flatten()[:3]}, norm = {tr.norm(gradient)}")
      get_path(env, policy, args['decisions'], print_path=True)

    if ep_reward_mean == 3.0:
      return_hist += (max_t - t) * [3.0]
      print ("\n")
      return policy, pi0, return_hist

    if ep_reward_mean > best_r:
      best_policy = policy 
      best_r = ep_reward_mean

  if plot:
    fig, ax = plt.subplots(figsize=(4,3))

    ax.plot(return_hist)
    ax.set_ylabel('Expected Reward')

  return best_policy, pi0, return_hist



def get_exp_rl_reg(policy, val, env, pi0, args):
    # batch data collection, ultimately we will have t rows, of n_parallel entries
    # returns:
    # > log_pi: the log prob of the action taken
    # > all_v: the value function for the s,a pairs
    # > all_r: the single step reward for the s,a pairs
    # Once episodes have terminated, we set the reward to zero, this is the done vector
    n_parallel, device, beta_ee, omega = args['n_parallel'], args['device'], args['beta_ee'], args['omega']
    t=0
    vec_env = [copy.copy(env) for _ in range(n_parallel)] # n environments
    done = [False for _ in range(n_parallel)] # nx1 vec of done = False
    state = [env.reset() for env in vec_env] # n x state dim 
    zerostate = tr.zeros_like(state[0]) # placeholder state for when done
    placeholder = [zerostate, 0, True] # placeholder for when envs are done
    
    while not all(done):
      state = tr.stack((state)).to(device) # make the states into a single tensor 
      probs = tr.stack([tr.dot(softmax(policy).flatten(), s) for s in state])
      probs = tr.stack((1-probs,probs), dim=1) # 
      if pi0 is not None:
          if pi0.requires_grad: 
            probs0 = tr.stack([tr.dot(softmax(pi0).flatten(), s) for s in state])
            probs0 = tr.stack((1-probs0, probs0), dim=1)
          elif args['reg'] == 'TV':
            probs0 = tr.stack([pi0[obs2node(env, s), :] for s in state], dim=0)
          elif args['reg'] == 'log-barrier':
            probs0 = 0.5 * tr.ones_like(probs)
      value = tr.stack([tr.dot(val.flatten(), s) for s in state]) # compute value

      # compute similarity scores
      if pi0 is not None:
        forward_kl, reverse_kl = [], []
        for i,s in enumerate(state):
          forward_kl.append(KL(probs0[i], probs[i]))
          reverse_kl.append(KL(probs[i], probs0[i]))
        forward_kl = tr.stack(forward_kl)
        reverse_kl = tr.stack(reverse_kl)
      else:
        forward_kl = tr.zeros_like(probs)
        reverse_kl = tr.zeros_like(probs)


      m = Categorical(probs)
      action = m.sample()     
      action_vec = tr.stack([m.log_prob(action[i])[0].double() if not done[i] else tr.zeros_like(m.log_prob(action[i])[0].double()).to(device) for i in range(n_parallel)]) # 1 x n 
      value_vec = tr.stack([value[i] if not done[i] else tr.zeros_like(value[i]).to(device) for i in range(len(value))]) # 1 x n                                                                           
      step_results = [vec_env[i].step(action[i]) if not done[i] else placeholder for i in range(len(vec_env))]
      rewards = tr.stack([tr.tensor(x[1]).long() for x in step_results])
      new_state = [x[0] for x in step_results]
      done = [x[2] for x in step_results]
      state = [new_state[i] if not done[i] else zerostate for i in range(len(new_state))]
      vec_env = [vec_env[i] if not done[i] else zerostate for i in range(len(vec_env))]

      if t == 0:
        log_pi = action_vec.clone().reshape(1,-1)
        all_v = value_vec.clone().reshape(1,-1)
        all_r = rewards.clone().reshape(1,-1)
        all_probs = [probs]
        all_reverse_kl = reverse_kl.clone().reshape(1, -1)
        all_forward_kl = forward_kl.clone().reshape(1, -1)
      else:
        log_pi = tr.cat((log_pi, action_vec.reshape(1,-1))) # each timestep is a new ROW of these
        all_v = tr.cat((all_v, value_vec.reshape(1,-1)))
        all_r = tr.cat((all_r, rewards.reshape(1,-1)))
        all_probs.append(probs)
        all_reverse_kl = tr.cat((all_reverse_kl, reverse_kl.reshape(1, -1)))
        all_forward_kl = tr.cat((all_forward_kl, forward_kl.reshape(1, -1)))
      t += 1

    # policy and value losses
    rewards = all_r.float() # single_step rewards, t x n
    reward_to_go = tr.zeros_like(rewards) # t x n
    reward_to_go[-1, :] = rewards[-1, :] # bottom row is final single step reward
    for i in range(2, reward_to_go.shape[0]+1): 
        reward_to_go[-i, :] = 0.99 * reward_to_go[-(i-1), :] + rewards[-i, :] # build up so each R_i = r_i + \sum_j=1^{i-1} r_j * \gamma^j

    returns = -reward_to_go.to(device)

    v_advantages = returns - all_v # t x n. Advantages with gradients through V
    value_loss = (v_advantages**2).mean(1).sum() # positive loss, want to minimize, must take negative

    advantages = returns - all_v.detach() # t x n. 
    pg_loss = (advantages * log_pi).sum()

    # exploration/exploitation loss
    all_probs = tr.stack(all_probs)
    H_pi = entropy(all_probs, axis=-1) # everything is (num time steps) x (num trajectories)
    if args['reg'] == 'maxent': Omega = -H_pi; 
    elif args['reg'] == 'kl-reverse': Omega = -all_reverse_kl; 
    elif args['reg'] == 'distral': Omega = -all_reverse_kl - H_pi; 
    elif args['reg'] in ['kl-forward', 'TV', 'log-barrier']: Omega = -all_forward_kl;
    else: Omega = tr.zeros_like(H_pi); 
    Omega = tr.mean(tr.sum(Omega, axis=0)) # sum over time steps, average over trajectories
    
    pg_loss = pg_loss - beta_ee * Omega
    

    
    return pg_loss, value_loss, rewards.sum(0).mean()





