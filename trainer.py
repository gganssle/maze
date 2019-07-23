import numpy as np 
import torch
import torch.nn as nn
from agnt import ffnn 
from env import lumpy, reward, discount
from tqdm import tqdm

#model = ffnn.net()
#model.train()
actor = ffnn.agent(train=True)

plot = True
num_games = 100
max_iter = 40
model_checkpoint = f'/Users/gram/maze/dat/checkpoints/temp'

rwrd = reward.score()
dcount = discount.disc()

for game in range(num_games):
  env = lumpy.world(gridsize=20)        

  state = env.initial_state
  cursor = env.cursor
  local_reward = rwrd.initial_reward

  running_reward = [0]
  discounted_reward = dcount.expcoef(running_reward)

  for i in range(max_iter):
    actions0 = actor.decision(state, cursor, env.end, [local_reward]) 
    
    state, cursor = env.get_state(state, cursor, action)

    local_reward = rwrd.endonly(cursor, env.end)
    



torch.save(model.state_dict(), model_checkpoint)
