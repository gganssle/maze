import numpy as np 
import torch
import torch.nn as nn
from agnt import ffnn 
from env import lumpy, reward, discount
from tqdm import trange

actor = ffnn.agent(train=True)
actor.model.train()

optionswords = ['left', 'right', 'up', 'down']
plot = True
num_games = 1000
max_iter = 40
discount_factor = 0.9
learning_rate = 0.001
model_checkpoint = f'/Users/gram/maze/dat/checkpoints/temp'

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(actor.model.parameters(), lr=learning_rate)

rwrd = reward.score()
dcount = discount.disc()

for game in trange(num_games):
  env = lumpy.world(gridsize=20)        

  state = env.initial_state
  cursor = env.cursor
  local_reward = rwrd.initial_reward

  for i in range(max_iter):
    # for time step 0
    q_vals = actor.decision(state, cursor, env.end, [local_reward])
    action = np.argmax(q_vals.data.numpy())
    
    # for time step 1
    state1, cursor1 = env.get_state(state, cursor, optionswords[action])
    local_reward1 = rwrd.endonly(cursor1, env.end)

    q_vals1 = actor.decision(state1, cursor1, env.end, [local_reward1])
    action1 = np.argmax(q_vals1.data.numpy())

    # calculate target
    targetele = local_reward + discount_factor * q_vals[action1]
    target = q_vals.clone()
    target[action] = targetele

    # backpropagate errors
    loss = criterion(q_vals, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # step the game
    state = state1
    cursor = cursor1 
    local_reward = local_reward1

torch.save(actor.model.state_dict(), model_checkpoint)
