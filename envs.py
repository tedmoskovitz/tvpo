import binarytree
from binarytree import tree, build, Node
import numpy as np
import torch

def create_tree_robust(depth=5, sparsity=0.5, reverse=False, max_reward=10, fixed=False, good_idxs=[15], bad_idxs=[], doors=[17]):
  while True: 
    try:
      if fixed: root = create_fixed_tree(good_idxs=good_idxs, bad_idxs=bad_idxs, doors=doors)
      else: root = create_tree(depth, sparsity, reverse, max_reward)
      return(root)
    except:
      IndexError


def create_small_tree(base=17):
  root = Node(base + 1)
  root.left = Node(base + 2)
  root.right = Node(base + 3)

  root.left.left = Node(base + 4)
  root.left.right = Node(base + 5)
  
  root.right.left = Node(base + 6)
  root.right.right = Node(base + 7)

  return root


def create_fixed_tree(good_idxs=[15], bad_idxs=[], base=0, doors=[17], recurse=True):
  root = Node(base + 1)
  root.left = Node(base + 2)
  root.right = Node(base + 3)


  # let's first create the left subtree
  root.left.left = Node(base + 4)
  root.left.right = Node(base + 5)
  root.left.right.reward = -1
  root.left.left.left = Node(base + 8)
  root.left.left.left.reward = -1
  root.left.left.right = Node(base + 9)
  root.left.left.right.reward = -1
  nine = root.left.left.right 
  nine.left = Node(base + 12)

  nine.right = Node(base + 13)

  # now we'll create the right side
  root.right.left = Node(base + 6)

  root.right.right = Node(base + 7)
  seven = root.right.right
  seven.left = Node(base + 10)
  seven.left.left = Node(base + 14)
  seven.left.right = Node(base + 15)
  seven.right = Node(base + 11)
  seven.right.left = Node(base + 16)
  seven.right.right = Node(base + 17)

  door_list = [
    nine.left, nine.right, seven.left.left, seven.left.right, seven.right.left, seven.right.right
  ] 
  num_list = [12, 13, 14, 15, 16, 17]
  num2door = dict(zip(num_list, door_list))

  addon = create_small_tree(base=17)
  for door_num in doors:
    num2door[door_num].right = addon
    num2door[door_num].left = addon


  for level in root.levels:
    for x in level:
      if x.value in good_idxs: x.reward = 1;
      elif x.value in bad_idxs: x.reward = 0; 
      else: x.reward = 0;
  
  return root


def create_tree(depth=5, sparsity=0.5, reverse=False, max_reward=10, start_idxs=None):

  if start_idxs is None:
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
  else: 
    assert len(start_idxs) == 3, "incorrect starting indices"
    i, j, k = start_idxs
    root = Node(i)
    root.left = Node(j)
    root.right = Node(k)
  l = 1 #if start_idxs is None else # current depth level
  curr = 4 if start_idxs is None else 12# tracks number of nodes in tree
  while l <= depth:
    level = root.levels[l]
    for node in level:
      # add a child w.p. 1 - sparsity 
      p = np.random.rand()
      if p > sparsity:
        node.left = Node(curr)
        curr += 1
      p = np.random.rand()
      if p > sparsity:  
        node.right = Node(curr)
        curr += 1
    l += 1

  # set rewards
  depth = 0
  for level in root.levels:
    depth += 1
    for node in level:
      if (node.left is None) and (node.right is None):
        # set reward if node is a leaf node
        if reverse:
          node.reward = max_reward # if reverse, just use max reward
        else:
          node.reward = depth # default reward is depth of node in tree
      elif (node.left is not None) and (node.right is None):
        # if node missing a right child, add one and give it a reward of -1
        node.right = Node(curr)
        node.right.reward = -1
        curr += 1
      elif (node.left is None) and (node.right is not None):
        # if node missing a right child, add one and give it a reward of -1
        node.left = Node(curr)
        node.left.reward = -1
        curr += 1
  return(root)


class TreeEnv(object):
  def __init__(
      self, depth=5, sparsity=0.8, reverse=True, max_reward=10, subtree_depth=1, structured=False, fixed=False, good_idxs=[15], bad_idxs=[], doors=[17], dense_r=False):

    self.depth = depth
    self.sparsity = sparsity
    self.reverse = reverse
    self.dense_r = dense_r
    self.max_reward = max_reward 
    self.root = create_tree_robust(
        depth=self.depth,
        sparsity=self.sparsity,
        reverse=self.reverse,
        max_reward=max_reward,
        subtree_depth=subtree_depth,
        fixed=fixed,
        good_idxs=good_idxs,
        bad_idxs=bad_idxs,
        doors=doors
    )
    self.curr_node = self.root
    # values for non-leaf nodes
    self.decisions = [x.value for level in self.root.levels for x in level if (x.left is not None) and (x.right is not None)]

  def print(self):
    self.root.pprint()
    print("Rewards:\n")
    for x in self.root.leaves:
      print("* Node: {}, Reward: {}".format(str(x.value), str(x.reward)))

  def reset(self):
    self.done = False
    self.curr_node = self.root
    obs = torch.zeros(len(self.decisions))
    obs[self.decisions.index(self.curr_node.value)] = 1 # observation is a 1-hot encoding of the number of non-leaf nodes
    return obs
  
  def step(self, action):
    #pdb.set_trace()
    if self.done:
      raise Exception("Done, cannot step.")
      pass
    if action == 1:
      new_node = self.curr_node.right
    elif action == 0:
      new_node = self.curr_node.left
    else:
      raise Exception("Action must be 0 or 1")
    self.curr_node = new_node

    obs = torch.zeros(len(self.decisions))
    if (self.curr_node.left is not None) and (self.curr_node.right is not None):
      reward = 0 if not self.dense_r else 0.2 * self.curr_node.reward

      obs[self.decisions.index(self.curr_node.value)] = 1 # only if not done
    else:
      reward = self.curr_node.reward
      self.done = True
    
    info = self.curr_node.value
    return(obs, reward, self.done, info)


class TreeDistribution(object):
    
    def __init__(self, type1=True):
        # type1 = shared structure in 1st room; type2 = shared structure in 2nd
        
        if type1:
            self.reward_options = [21, 22, 23, 24]
            self.door_options = [17]
        else:
            self.reward_options = [24]
            self.door_options = [12, 13, 14, 15]
        
    def sample(self):
        # sample number of rewarded nodes; use geometric dist to encourage sparsity
        n_r = min(np.random.geometric(p=0.5), len(self.reward_options))
        # sample number of doors to second room (maybe forget this...)
        n_d = min(np.random.geometric(p=0.5), len(self.door_options))
        
        # sample rewarded nodes
        rewarded_nodes = np.random.choice(self.reward_options, size=n_r)
        # sample doors
        doors = np.random.choice(self.door_options, size=n_d)
        # create tree env
        task = TreeEnv(fixed=True, good_idxs=rewarded_nodes, doors=doors)
        
        return task 
        
    @property
    def base_env(self):
        """an environment with no reward"""
        return TreeEnv(fixed=True, good_idxs=[], doors=self.door_options)